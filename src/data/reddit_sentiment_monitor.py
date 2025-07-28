#!/usr/bin/env python3
"""
Reddit Sentiment Monitoring Service

Real-time Reddit monitoring and sentiment analysis service for the Alpha Discovery platform.
Tracks mentions, sentiment, and engagement metrics for stock symbols across multiple subreddits.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import aiohttp
import asyncpraw
from textblob import TextBlob
import pandas as pd
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from configs.config_loader import get_config
from src.utils.database import DatabaseManager
from src.utils.metrics import MetricsCollector
from src.utils.text_processing import TextProcessor

logger = logging.getLogger(__name__)

@dataclass
class RedditPost:
    """Reddit post data structure."""
    id: str
    title: str
    content: str
    author: str
    subreddit: str
    score: int
    num_comments: int
    created_utc: datetime
    url: str
    symbols_mentioned: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    engagement_score: float = 0.0

@dataclass
class RedditComment:
    """Reddit comment data structure."""
    id: str
    post_id: str
    content: str
    author: str
    subreddit: str
    score: int
    created_utc: datetime
    parent_id: Optional[str] = None
    symbols_mentioned: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"

@dataclass
class SentimentMetrics:
    """Sentiment metrics for a symbol."""
    symbol: str
    timestamp: datetime
    total_mentions: int
    positive_mentions: int
    negative_mentions: int
    neutral_mentions: int
    average_sentiment: float
    sentiment_trend: str
    volume_trend: str
    engagement_score: float
    top_posts: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

class RedditSentimentMonitor:
    """Reddit sentiment monitoring and analysis service."""
    
    def __init__(self):
        self.config = get_config('monitoring')['reddit']
        self.db_manager = DatabaseManager(self.config)
        self.metrics_collector = MetricsCollector(self.config)
        self.text_processor = TextProcessor()
        
        # Reddit API client
        self.reddit = None
        self.reddit_config = self.config.get('api', {})
        
        # Monitoring state
        self.is_running = False
        self.monitored_subreddits = self.config.get('subreddits', [])
        self.tracked_symbols = set()
        
        # Data storage
        self.posts_cache = deque(maxlen=10000)
        self.comments_cache = deque(maxlen=50000)
        self.sentiment_cache = {}
        self.symbol_mentions = defaultdict(list)
        
        # Processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_tasks = []
        
        # Rate limiting
        self.request_timestamps = deque(maxlen=60)  # Track last 60 requests
        self.max_requests_per_minute = 60
        
        # Symbol extraction patterns
        self.symbol_patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL format
            r'\b([A-Z]{1,5})\b(?=\s|$|[^A-Z])',  # Standalone tickers
            r'(?:ticker|stock|symbol):\s*([A-Z]{1,5})',  # Explicit mentions
        ]
        
        logger.info("Reddit Sentiment Monitor initialized")
    
    async def initialize_reddit_client(self) -> None:
        """Initialize Reddit API client."""
        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.reddit_config.get('client_id'),
                client_secret=self.reddit_config.get('client_secret'),
                user_agent=self.reddit_config.get('user_agent', 'AlphaDiscovery/1.0'),
                username=self.reddit_config.get('username'),
                password=self.reddit_config.get('password')
            )
            
            # Test connection
            await self.reddit.user.me()
            logger.info("Reddit API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            raise
    
    async def start_monitoring(self, symbols: List[str]) -> None:
        """Start Reddit monitoring for given symbols."""
        logger.info(f"Starting Reddit monitoring for {len(symbols)} symbols")
        
        try:
            self.is_running = True
            self.tracked_symbols = set(symbols)
            
            # Initialize Reddit client
            await self.initialize_reddit_client()
            
            # Start monitoring tasks
            for subreddit in self.monitored_subreddits:
                # Monitor new posts
                post_task = asyncio.create_task(self._monitor_subreddit_posts(subreddit))
                self.processing_tasks.append(post_task)
                
                # Monitor comments
                comment_task = asyncio.create_task(self._monitor_subreddit_comments(subreddit))
                self.processing_tasks.append(comment_task)
            
            # Start sentiment analysis task
            sentiment_task = asyncio.create_task(self._process_sentiment_analysis())
            self.processing_tasks.append(sentiment_task)
            
            # Start metrics calculation task
            metrics_task = asyncio.create_task(self._calculate_sentiment_metrics())
            self.processing_tasks.append(metrics_task)
            
            logger.info("Reddit monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Reddit monitoring: {e}")
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop Reddit monitoring."""
        logger.info("Stopping Reddit monitoring...")
        
        self.is_running = False
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()
        
        # Close Reddit client
        if self.reddit:
            await self.reddit.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Reddit monitoring stopped")
    
    async def _monitor_subreddit_posts(self, subreddit_name: str) -> None:
        """Monitor new posts in a subreddit."""
        logger.info(f"Starting post monitoring for r/{subreddit_name}")
        
        try:
            subreddit = await self.reddit.subreddit(subreddit_name)
            
            # Track processed posts to avoid duplicates
            processed_posts = set()
            
            while self.is_running:
                try:
                    # Check rate limit
                    if not self._check_rate_limit():
                        await asyncio.sleep(1)
                        continue
                    
                    # Get new posts
                    async for submission in subreddit.new(limit=25):
                        if submission.id in processed_posts:
                            continue
                        
                        processed_posts.add(submission.id)
                        
                        # Process post
                        post = await self._process_reddit_post(submission)
                        if post and post.symbols_mentioned:
                            self.posts_cache.append(post)
                            
                            # Update symbol mentions
                            for symbol in post.symbols_mentioned:
                                self.symbol_mentions[symbol].append({
                                    'type': 'post',
                                    'data': post,
                                    'timestamp': post.created_utc
                                })
                    
                    # Clean old processed posts
                    if len(processed_posts) > 1000:
                        # Keep only recent posts
                        processed_posts = set(list(processed_posts)[-500:])
                    
                    await asyncio.sleep(self.config.get('post_check_interval', 30))
                    
                except Exception as e:
                    logger.error(f"Error monitoring posts in r/{subreddit_name}: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"Fatal error in post monitoring for r/{subreddit_name}: {e}")
    
    async def _monitor_subreddit_comments(self, subreddit_name: str) -> None:
        """Monitor new comments in a subreddit."""
        logger.info(f"Starting comment monitoring for r/{subreddit_name}")
        
        try:
            subreddit = await self.reddit.subreddit(subreddit_name)
            
            # Track processed comments to avoid duplicates
            processed_comments = set()
            
            while self.is_running:
                try:
                    # Check rate limit
                    if not self._check_rate_limit():
                        await asyncio.sleep(1)
                        continue
                    
                    # Get new comments
                    async for comment in subreddit.comments(limit=100):
                        if comment.id in processed_comments:
                            continue
                        
                        processed_comments.add(comment.id)
                        
                        # Process comment
                        comment_data = await self._process_reddit_comment(comment)
                        if comment_data and comment_data.symbols_mentioned:
                            self.comments_cache.append(comment_data)
                            
                            # Update symbol mentions
                            for symbol in comment_data.symbols_mentioned:
                                self.symbol_mentions[symbol].append({
                                    'type': 'comment',
                                    'data': comment_data,
                                    'timestamp': comment_data.created_utc
                                })
                    
                    # Clean old processed comments
                    if len(processed_comments) > 5000:
                        # Keep only recent comments
                        processed_comments = set(list(processed_comments)[-2500:])
                    
                    await asyncio.sleep(self.config.get('comment_check_interval', 15))
                    
                except Exception as e:
                    logger.error(f"Error monitoring comments in r/{subreddit_name}: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"Fatal error in comment monitoring for r/{subreddit_name}: {e}")
    
    async def _process_reddit_post(self, submission) -> Optional[RedditPost]:
        """Process a Reddit post."""
        try:
            # Extract text content
            title = submission.title or ""
            content = submission.selftext or ""
            full_text = f"{title} {content}"
            
            # Extract symbols
            symbols = self._extract_symbols(full_text)
            
            # Only process if relevant symbols are mentioned
            if not any(symbol in self.tracked_symbols for symbol in symbols):
                return None
            
            # Filter symbols to only tracked ones
            relevant_symbols = [s for s in symbols if s in self.tracked_symbols]
            
            # Calculate sentiment
            sentiment_score, sentiment_label = self._analyze_sentiment(full_text)
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(
                submission.score,
                submission.num_comments,
                'post'
            )
            
            post = RedditPost(
                id=submission.id,
                title=title,
                content=content,
                author=str(submission.author) if submission.author else "[deleted]",
                subreddit=submission.subreddit.display_name,
                score=submission.score,
                num_comments=submission.num_comments,
                created_utc=datetime.fromtimestamp(submission.created_utc),
                url=f"https://reddit.com{submission.permalink}",
                symbols_mentioned=relevant_symbols,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                engagement_score=engagement_score
            )
            
            return post
            
        except Exception as e:
            logger.error(f"Error processing Reddit post: {e}")
            return None
    
    async def _process_reddit_comment(self, comment) -> Optional[RedditComment]:
        """Process a Reddit comment."""
        try:
            # Extract text content
            content = comment.body or ""
            
            # Extract symbols
            symbols = self._extract_symbols(content)
            
            # Only process if relevant symbols are mentioned
            if not any(symbol in self.tracked_symbols for symbol in symbols):
                return None
            
            # Filter symbols to only tracked ones
            relevant_symbols = [s for s in symbols if s in self.tracked_symbols]
            
            # Calculate sentiment
            sentiment_score, sentiment_label = self._analyze_sentiment(content)
            
            comment_data = RedditComment(
                id=comment.id,
                post_id=comment.submission.id,
                content=content,
                author=str(comment.author) if comment.author else "[deleted]",
                subreddit=comment.subreddit.display_name,
                score=comment.score,
                created_utc=datetime.fromtimestamp(comment.created_utc),
                parent_id=comment.parent_id,
                symbols_mentioned=relevant_symbols,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label
            )
            
            return comment_data
            
        except Exception as e:
            logger.error(f"Error processing Reddit comment: {e}")
            return None
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        symbols = set()
        
        # Convert to uppercase for pattern matching
        text_upper = text.upper()
        
        # Apply symbol patterns
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, text_upper)
            symbols.update(matches)
        
        # Filter out common false positives
        false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'HAD', 'BUT', 'WHAT', 'THERE', 'WE', 'HE', 'HAVE', 'IT', 'FROM', 'THEY',
            'SHE', 'OR', 'AN', 'WILL', 'MY', 'NOW', 'BEEN', 'SAID', 'EACH', 'WHICH', 'DO',
            'THEIR', 'TIME', 'IF', 'UP', 'OUT', 'MANY', 'THEN', 'THEM', 'THESE', 'SO', 'SOME',
            'HIM', 'HAS', 'TWO', 'MORE', 'VERY', 'TO', 'OF', 'IN', 'IS', 'ON', 'AT', 'BE',
            'BY', 'THIS', 'THAT', 'WITH', 'AS', 'IT', 'HIS', 'HER', 'WHO', 'OIL', 'GAS', 'CAR',
            'NEW', 'OLD', 'BIG', 'BAD', 'GOOD', 'BEST', 'LAST', 'LONG', 'GREAT', 'LITTLE', 'OWN',
            'OTHER', 'RIGHT', 'HIGH', 'EVERY', 'ANOTHER', 'SAME', 'FEW', 'MUCH', 'WELL', 'ALSO'
        }
        
        # Remove false positives and ensure length constraints
        valid_symbols = {
            symbol for symbol in symbols
            if (
                symbol not in false_positives and
                1 <= len(symbol) <= 5 and
                symbol.isalpha()
            )
        }
        
        return list(valid_symbols)
    
    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment of text."""
        try:
            # Clean text
            cleaned_text = self.text_processor.clean_text(text)
            
            # Use TextBlob for sentiment analysis
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity
            
            # Determine sentiment label
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return polarity, label
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0, "neutral"
    
    def _calculate_engagement_score(self, score: int, num_comments: int, content_type: str) -> float:
        """Calculate engagement score for content."""
        try:
            # Base score from upvotes/downvotes
            base_score = max(0, score)
            
            # Comment weight (comments are more valuable than just upvotes)
            comment_weight = num_comments * 2
            
            # Content type multiplier
            type_multiplier = 1.5 if content_type == 'post' else 1.0
            
            # Calculate final score
            engagement_score = (base_score + comment_weight) * type_multiplier
            
            # Normalize to 0-1 scale (log scale for better distribution)
            import math
            normalized_score = math.log(engagement_score + 1) / math.log(1000)
            
            return min(1.0, normalized_score)
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.0
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        
        # Remove old timestamps
        while self.request_timestamps and now - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        # Check if we can make another request
        if len(self.request_timestamps) < self.max_requests_per_minute:
            self.request_timestamps.append(now)
            return True
        
        return False
    
    async def _process_sentiment_analysis(self) -> None:
        """Process sentiment analysis for collected data."""
        while self.is_running:
            try:
                # Process posts and comments in batches
                posts_to_process = []
                comments_to_process = []
                
                # Get unprocessed posts
                for post in list(self.posts_cache):
                    if post.symbols_mentioned:
                        posts_to_process.append(post)
                
                # Get unprocessed comments
                for comment in list(self.comments_cache):
                    if comment.symbols_mentioned:
                        comments_to_process.append(comment)
                
                # Save to database
                if posts_to_process:
                    await self.db_manager.save_reddit_posts(posts_to_process)
                
                if comments_to_process:
                    await self.db_manager.save_reddit_comments(comments_to_process)
                
                # Update metrics
                self.metrics_collector.record_reddit_data(
                    len(posts_to_process),
                    len(comments_to_process)
                )
                
                await asyncio.sleep(self.config.get('processing_interval', 60))
                
            except Exception as e:
                logger.error(f"Error in sentiment analysis processing: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_sentiment_metrics(self) -> None:
        """Calculate sentiment metrics for tracked symbols."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for symbol in self.tracked_symbols:
                    # Get recent mentions
                    recent_mentions = self._get_recent_mentions(symbol, hours=24)
                    
                    if not recent_mentions:
                        continue
                    
                    # Calculate metrics
                    metrics = self._calculate_symbol_sentiment_metrics(symbol, recent_mentions)
                    
                    # Store in cache
                    self.sentiment_cache[symbol] = metrics
                    
                    # Save to database
                    await self.db_manager.save_sentiment_metrics(metrics)
                
                await asyncio.sleep(self.config.get('metrics_interval', 300))  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error calculating sentiment metrics: {e}")
                await asyncio.sleep(300)
    
    def _get_recent_mentions(self, symbol: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent mentions for a symbol."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_mentions = []
        for mention in self.symbol_mentions.get(symbol, []):
            if mention['timestamp'] > cutoff_time:
                recent_mentions.append(mention)
        
        return recent_mentions
    
    def _calculate_symbol_sentiment_metrics(self, symbol: str, mentions: List[Dict[str, Any]]) -> SentimentMetrics:
        """Calculate sentiment metrics for a symbol."""
        if not mentions:
            return SentimentMetrics(
                symbol=symbol,
                timestamp=datetime.now(),
                total_mentions=0,
                positive_mentions=0,
                negative_mentions=0,
                neutral_mentions=0,
                average_sentiment=0.0,
                sentiment_trend="neutral",
                volume_trend="stable",
                engagement_score=0.0
            )
        
        # Count sentiment categories
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        sentiment_scores = []
        engagement_scores = []
        top_posts = []
        
        for mention in mentions:
            data = mention['data']
            
            if data.sentiment_label == 'positive':
                positive_count += 1
            elif data.sentiment_label == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
            
            sentiment_scores.append(data.sentiment_score)
            
            if hasattr(data, 'engagement_score'):
                engagement_scores.append(data.engagement_score)
            
            # Collect top posts
            if mention['type'] == 'post' and hasattr(data, 'score'):
                top_posts.append({
                    'id': data.id,
                    'title': data.title,
                    'score': data.score,
                    'url': data.url
                })
        
        # Calculate averages
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0
        
        # Determine sentiment trend
        if avg_sentiment > 0.1:
            sentiment_trend = "bullish"
        elif avg_sentiment < -0.1:
            sentiment_trend = "bearish"
        else:
            sentiment_trend = "neutral"
        
        # Determine volume trend (based on recent mention count)
        total_mentions = len(mentions)
        if total_mentions > 50:
            volume_trend = "high"
        elif total_mentions > 20:
            volume_trend = "moderate"
        else:
            volume_trend = "low"
        
        # Sort top posts by score
        top_posts.sort(key=lambda x: x['score'], reverse=True)
        top_post_ids = [post['id'] for post in top_posts[:5]]
        
        return SentimentMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            total_mentions=total_mentions,
            positive_mentions=positive_count,
            negative_mentions=negative_count,
            neutral_mentions=neutral_count,
            average_sentiment=avg_sentiment,
            sentiment_trend=sentiment_trend,
            volume_trend=volume_trend,
            engagement_score=avg_engagement,
            top_posts=top_post_ids
        )
    
    async def get_sentiment(self, symbol: str, timeframe: str = "1d", include_posts: bool = False) -> Dict[str, Any]:
        """Get sentiment data for a symbol."""
        try:
            # Get cached metrics
            metrics = self.sentiment_cache.get(symbol)
            
            if not metrics:
                # Calculate on-demand
                hours = {"1h": 1, "4h": 4, "1d": 24, "3d": 72, "1w": 168}.get(timeframe, 24)
                recent_mentions = self._get_recent_mentions(symbol, hours)
                metrics = self._calculate_symbol_sentiment_metrics(symbol, recent_mentions)
            
            result = {
                'symbol': symbol,
                'timestamp': metrics.timestamp.isoformat(),
                'total_mentions': metrics.total_mentions,
                'sentiment_breakdown': {
                    'positive': metrics.positive_mentions,
                    'negative': metrics.negative_mentions,
                    'neutral': metrics.neutral_mentions
                },
                'average_sentiment': metrics.average_sentiment,
                'sentiment_trend': metrics.sentiment_trend,
                'volume_trend': metrics.volume_trend,
                'engagement_score': metrics.engagement_score
            }
            
            if include_posts:
                # Get recent posts
                recent_posts = []
                for mention in self._get_recent_mentions(symbol, 24):
                    if mention['type'] == 'post':
                        post_data = mention['data']
                        recent_posts.append({
                            'id': post_data.id,
                            'title': post_data.title,
                            'score': post_data.score,
                            'sentiment': post_data.sentiment_label,
                            'url': post_data.url,
                            'created_utc': post_data.created_utc.isoformat()
                        })
                
                result['recent_posts'] = recent_posts[:10]  # Limit to 10 most recent
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return {'error': str(e)}
    
    async def analyze_news_sentiment(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze news sentiment for a symbol."""
        try:
            # Get recent mentions
            recent_mentions = self._get_recent_mentions(symbol, hours_back)
            
            # Filter for high-engagement posts (likely news-related)
            news_mentions = [
                mention for mention in recent_mentions
                if (mention['type'] == 'post' and 
                    hasattr(mention['data'], 'engagement_score') and
                    mention['data'].engagement_score > 0.5)
            ]
            
            if not news_mentions:
                return {
                    'symbol': symbol,
                    'news_sentiment': 'neutral',
                    'news_count': 0,
                    'confidence': 0.0
                }
            
            # Calculate news sentiment
            sentiment_scores = [mention['data'].sentiment_score for mention in news_mentions]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Determine news sentiment
            if avg_sentiment > 0.2:
                news_sentiment = 'positive'
            elif avg_sentiment < -0.2:
                news_sentiment = 'negative'
            else:
                news_sentiment = 'neutral'
            
            # Calculate confidence based on number of mentions and consistency
            confidence = min(1.0, len(news_mentions) / 10.0)
            
            return {
                'symbol': symbol,
                'news_sentiment': news_sentiment,
                'news_count': len(news_mentions),
                'average_sentiment': avg_sentiment,
                'confidence': confidence,
                'timeframe_hours': hours_back
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return {'error': str(e)}
    
    async def get_recent_sentiment(self) -> Dict[str, Any]:
        """Get recent sentiment data for all tracked symbols."""
        try:
            sentiment_data = {}
            
            for symbol in self.tracked_symbols:
                if symbol in self.sentiment_cache:
                    metrics = self.sentiment_cache[symbol]
                    sentiment_data[symbol] = {
                        'total_mentions': metrics.total_mentions,
                        'average_sentiment': metrics.average_sentiment,
                        'sentiment_trend': metrics.sentiment_trend,
                        'volume_trend': metrics.volume_trend,
                        'last_updated': metrics.timestamp.isoformat()
                    }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'symbols': sentiment_data,
                'total_symbols': len(sentiment_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting recent sentiment: {e}")
            return {'error': str(e)}
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'is_running': self.is_running,
            'tracked_symbols': len(self.tracked_symbols),
            'monitored_subreddits': len(self.monitored_subreddits),
            'posts_cached': len(self.posts_cache),
            'comments_cached': len(self.comments_cache),
            'symbol_mentions': {
                symbol: len(mentions) 
                for symbol, mentions in self.symbol_mentions.items()
            },
            'sentiment_cache_size': len(self.sentiment_cache),
            'rate_limit_status': {
                'requests_last_minute': len(self.request_timestamps),
                'max_requests_per_minute': self.max_requests_per_minute
            }
        } 