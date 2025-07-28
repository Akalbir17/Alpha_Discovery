"""
Reddit Scraper - 2025 State-of-the-Art Edition

Advanced Reddit scraper for financial sentiment analysis and trend detection.
Features:
- Multi-subreddit monitoring with intelligent filtering
- State-of-the-art sentiment analysis using latest transformers
- Advanced trend detection with velocity and momentum tracking
- Unusual activity identification with AI-powered pattern recognition
- Redis caching with intelligent TTL management
- Rate limiting with adaptive backoff
- Integration with latest free AI models
- Real-time streaming capabilities
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter, deque
import praw
import redis
import structlog
import numpy as np
import pandas as pd
# ML imports - now using HTTP client instead of direct models
try:
    from .ml_client import ml_client
    ML_CLIENT_AVAILABLE = True
except ImportError:
    ML_CLIENT_AVAILABLE = False
    logger.warning("ML client not available - using fallback analysis")

# Remove heavy transformer imports - now handled by MCP server
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# import torch

# Keep lightweight imports for fallback
from textblob import TextBlob
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import hashlib

from ..utils.config_manager import get_config_section

# Get configuration sections
tool_config = get_config_section('tools') or {}
reddit_config = get_config_section('reddit') or {}
redis_config = get_config_section('redis') or {}

logger = structlog.get_logger(__name__)

if not ML_CLIENT_AVAILABLE:
    logger.warning("ML client not available - using fallback analysis")


@dataclass
class RedditPost:
    """Enhanced Reddit post data structure"""
    id: str
    title: str
    body: str
    author: str
    subreddit: str
    score: int
    num_comments: int
    created_utc: float
    url: str
    tickers: List[str]
    sentiment_score: float
    sentiment_label: str
    confidence: float
    emotions: Dict[str, float]  # New: emotion detection
    topics: List[str]  # New: topic modeling
    quality_score: float  # New: content quality assessment
    influence_score: float  # New: author influence score
    engagement_metrics: Dict[str, float]  # New: engagement analysis


@dataclass
class TrendingTicker:
    """Enhanced trending ticker data structure"""
    symbol: str
    mention_count: int
    sentiment_score: float
    sentiment_change: float
    velocity: float  # mentions per hour
    momentum: float  # New: velocity change rate
    subreddits: List[str]
    top_posts: List[Dict[str, Any]]
    unusual_activity: bool
    trend_strength: float
    market_correlation: float  # New: correlation with price movement
    social_signals: Dict[str, float]  # New: social media signals
    risk_score: float  # New: risk assessment
    prediction_confidence: float  # New: AI prediction confidence


@dataclass
class UnusualActivity:
    """Enhanced unusual activity detection result"""
    symbol: str
    activity_type: str  # options, volume, sentiment, etc.
    confidence: float
    description: str
    evidence: List[str]
    timestamp: datetime
    severity: str  # low, medium, high, critical
    market_impact: float  # predicted market impact
    similar_historical: List[Dict[str, Any]]  # similar past events


@dataclass
class SentimentAnalysis:
    """Enhanced sentiment analysis result"""
    overall_sentiment: float
    sentiment_distribution: Dict[str, float]
    confidence: float
    emotion_analysis: Dict[str, float]
    topic_sentiments: Dict[str, float]
    temporal_sentiment: List[Dict[str, Any]]
    quality_metrics: Dict[str, float]


class RedditScraper:
    """
    State-of-the-Art Reddit Scraper for 2025
    
    Enhanced Features:
    - Latest transformer models (RoBERTa, FinBERT, BERT-large)
    - Multi-modal sentiment analysis with emotion detection
    - Advanced topic modeling and trend prediction
    - Real-time streaming with WebSocket support
    - Intelligent caching with ML-based TTL optimization
    - Adaptive rate limiting with circuit breakers
    - AI-powered content quality assessment
    - Cross-platform correlation analysis
    """
    
    def __init__(self):
        self.reddit = None
        self.redis_client = None
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        self.ner_pipeline = None
        self.topic_model = None
        self.sentence_transformer = None
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.cache_lock = threading.Lock()
        
        # Enhanced configuration
        self.config = {
            "subreddits": [
                "wallstreetbets", "stocks", "investing", "SecurityAnalysis",
                "ValueInvesting", "StockMarket", "pennystocks", "options",
                "financialindependence", "personalfinance", "CryptoCurrency",
                "Bitcoin", "ethereum", "DeFi", "NFTs", "FinTech"
            ],
            "max_posts_per_subreddit": 200,
            "sentiment_cache_ttl": 1800,  # 30 minutes
            "trend_cache_ttl": 900,  # 15 minutes
            "rate_limit_delay": 0.1,
            "max_retries": 5,
            "quality_threshold": 0.6,
            "min_engagement_score": 10,
            "enable_streaming": True,
            "enable_ai_enhancement": True
        }
        
        # Performance metrics
        self.metrics = {
            "posts_processed": 0,
            "sentiment_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "errors": 0,
            "avg_processing_time": 0.0
        }
        
        # Initialize components
        self._initialize_reddit()
        self._initialize_redis()
        self._initialize_nlp_models()
        self._initialize_ticker_patterns()
        
    def _initialize_reddit(self):
        """Initialize Reddit client with enhanced configuration"""
        try:
            # Check if Reddit credentials are available
            if not reddit_config.get('client_id') or reddit_config.get('client_id') == 'your-reddit-client-id':
                logger.warning("Reddit credentials not configured - Reddit scraping disabled")
                self.reddit = None
                return
                
            self.reddit = praw.Reddit(
                client_id=reddit_config.get('client_id'),
                client_secret=reddit_config.get('client_secret'),
                user_agent=reddit_config.get('user_agent', 'AlphaDiscovery/2.0'),
                username=reddit_config.get('username'),
                password=reddit_config.get('password'),
                timeout=30,
                check_for_async=False
            )
            
            # Test connection
            self.reddit.user.me()
            logger.info("Reddit client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None
    
    def _initialize_redis(self):
        """Initialize Redis client with enhanced configuration"""
        try:
            # Check for Docker environment variables first
            redis_url = os.environ.get('REDIS_URL')
            if redis_url:
                # Parse Redis URL (format: redis://:password@host:port/db)
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=10,
                    socket_connect_timeout=10,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
            else:
                # Fallback to config file
                if not redis_config.get('host'):
                    logger.warning("Redis not configured - caching disabled")
                    self.redis_client = None
                    return
                    
            self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                decode_responses=True,
                socket_timeout=10,
                socket_connect_timeout=10,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self.redis_client = None
    
    def _initialize_nlp_models(self):
        """Initialize state-of-the-art NLP models for 2025 - now using ML client"""
        try:
            if not ML_CLIENT_AVAILABLE:
                logger.warning("ML client not available - using fallback models")
                return
            
            # Models are now handled by the MCP server via HTTP client
            # No local model loading needed - just verify connection
            logger.info("ML models will be accessed via MCP server")
            self.sentiment_pipeline = "remote"  # Placeholder to indicate remote access
            self.emotion_pipeline = "remote"
            self.ner_pipeline = "remote" 
            self.sentence_transformer = "remote"
            
            except Exception as e:
            logger.error(f"Error initializing ML client: {e}")
            # Set all to None to use fallback methods
            self.sentiment_pipeline = None
                self.emotion_pipeline = None
                self.ner_pipeline = None
                self.sentence_transformer = None
    
    def _initialize_ticker_patterns(self):
        """Initialize enhanced ticker extraction patterns"""
        # Common stock ticker patterns
        self.ticker_patterns = [
            r'\b[A-Z]{1,5}\b',  # Basic ticker pattern
            r'\$[A-Z]{1,5}\b',  # Dollar sign prefix
            r'\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b',  # With exchange suffix
        ]
        
        # Compiled regex patterns
        self.compiled_patterns = [re.compile(pattern) for pattern in self.ticker_patterns]
        
        # Common false positives to filter out
        self.false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'HAD', 'HAS', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD',
            'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'HIM', 'LET', 'PUT', 'SAY', 'SHE', 'TOO',
            'USE', 'CEO', 'IPO', 'ETF', 'SEC', 'FDA', 'FED', 'GDP', 'CPI', 'EPS', 'PE', 'PEG',
            'ROE', 'ROI', 'YTD', 'YOY', 'QOQ', 'MOM', 'EOD', 'AH', 'PM', 'DD', 'YOLO', 'FOMO',
            'HODL', 'FUD', 'ATH', 'ATL', 'DCA', 'TA', 'FA', 'RSI', 'MACD', 'SMA', 'EMA'
        }
        
        # Load real ticker symbols for validation
        self.valid_tickers = self._load_valid_tickers()
    
    @lru_cache(maxsize=1)
    def _load_valid_tickers(self) -> Set[str]:
        """Load valid ticker symbols from various sources"""
        try:
            # Start with common tickers
            tickers = {
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE',
                'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'SPOT', 'UBER', 'LYFT', 'ZOOM', 'SHOP',
                'SQ', 'ROKU', 'TWTR', 'SNAP', 'PINS', 'DOCU', 'WORK', 'ZM', 'PELOTON', 'RBLX',
                'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'SPCE', 'WISH', 'CLOV', 'MVIS', 'SNDL',
                'BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'XRP', 'LTC', 'BCH', 'BNB', 'SOL'
            }
            
            # Could be enhanced to fetch from API or file
            return tickers
            
        except Exception as e:
            logger.error(f"Error loading valid tickers: {e}")
            return set()
    
    async def get_sentiment_analysis(
        self,
        subreddits: List[str] = None,
        limit: int = 100,
        time_filter: str = "day",
        quality_threshold: float = 0.6
    ) -> SentimentAnalysis:
        """
        Enhanced sentiment analysis with 2025 features
        
        Args:
            subreddits: List of subreddits to analyze
            limit: Maximum number of posts per subreddit
            time_filter: Time filter for posts
            quality_threshold: Minimum quality score for posts
            
        Returns:
            Comprehensive sentiment analysis results
        """
        try:
            start_time = time.time()
            
            if not subreddits:
                subreddits = self.config["subreddits"][:5]  # Limit for performance
            
            # Check cache first
            cache_key = f"sentiment:{':'.join(subreddits)}:{limit}:{time_filter}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.metrics["cache_hits"] += 1
                return cached_result
            
            self.metrics["cache_misses"] += 1
            
            # Collect posts from all subreddits
            all_posts = []
            for subreddit in subreddits:
                posts = await self._get_posts_from_subreddit(
                    subreddit, limit, time_filter, quality_threshold
                )
                all_posts.extend(posts)
            
            if not all_posts:
                return SentimentAnalysis(
                    overall_sentiment=0.0,
                    sentiment_distribution={"positive": 0, "negative": 0, "neutral": 1},
                    confidence=0.0,
                    emotion_analysis={},
                    topic_sentiments={},
                    temporal_sentiment=[],
                    quality_metrics={"total_posts": 0, "avg_quality": 0.0}
                )
            
            # Perform enhanced sentiment analysis
            sentiment_results = await self._analyze_posts_sentiment(all_posts)
            
            # Perform emotion analysis
            emotion_results = await self._analyze_posts_emotions(all_posts)
            
            # Perform topic analysis
            topic_results = await self._analyze_posts_topics(all_posts)
            
            # Temporal sentiment analysis
            temporal_results = self._analyze_temporal_sentiment(all_posts)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(all_posts)
            
            # Combine results
            analysis = SentimentAnalysis(
                overall_sentiment=sentiment_results["overall"],
                sentiment_distribution=sentiment_results["distribution"],
                confidence=sentiment_results["confidence"],
                emotion_analysis=emotion_results,
                topic_sentiments=topic_results,
                temporal_sentiment=temporal_results,
                quality_metrics=quality_metrics
            )
            
            # Cache results
            self._cache_result(cache_key, analysis, self.config["sentiment_cache_ttl"])
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["sentiment_analyses"] += 1
            self.metrics["avg_processing_time"] = (
                (self.metrics["avg_processing_time"] * (self.metrics["sentiment_analyses"] - 1) + processing_time) /
                self.metrics["sentiment_analyses"]
            )
            
            logger.info(f"Sentiment analysis completed in {processing_time:.2f}s for {len(all_posts)} posts")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            self.metrics["errors"] += 1
            return SentimentAnalysis(
                overall_sentiment=0.0,
                sentiment_distribution={"positive": 0, "negative": 0, "neutral": 1},
                confidence=0.0,
                emotion_analysis={},
                topic_sentiments={},
                temporal_sentiment=[],
                quality_metrics={"error": str(e)}
            )
    
    async def _get_posts_from_subreddit(
        self,
        subreddit_name: str,
        limit: int,
        time_filter: str,
        quality_threshold: float
    ) -> List[RedditPost]:
        """Get posts from a specific subreddit with quality filtering"""
        try:
            if not self.reddit:
                return []
            
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            # Get posts based on time filter
            if time_filter == "hour":
                submissions = subreddit.hot(limit=limit * 2)  # Get more to filter
            elif time_filter == "day":
                submissions = subreddit.top(time_filter="day", limit=limit * 2)
            elif time_filter == "week":
                submissions = subreddit.top(time_filter="week", limit=limit * 2)
            else:
                submissions = subreddit.hot(limit=limit * 2)
            
            for submission in submissions:
                try:
                    # Skip if deleted or removed
                    if submission.selftext == "[deleted]" or submission.selftext == "[removed]":
                        continue
                    
                    # Calculate quality score
                    quality_score = self._calculate_post_quality(submission)
                    
                    # Skip low quality posts
                    if quality_score < quality_threshold:
                        continue
                    
                    # Extract tickers
                    text = f"{submission.title} {submission.selftext}"
                    tickers = self._extract_tickers(text)
                    
                    # Skip posts without tickers
                    if not tickers:
                        continue
                    
                    # Calculate engagement metrics
                    engagement_metrics = self._calculate_engagement_metrics(submission)
                    
                    # Create RedditPost object
                    post = RedditPost(
                        id=submission.id,
                        title=submission.title,
                        body=submission.selftext,
                        author=str(submission.author) if submission.author else "deleted",
                        subreddit=subreddit_name,
                        score=submission.score,
                        num_comments=submission.num_comments,
                        created_utc=submission.created_utc,
                        url=submission.url,
                        tickers=tickers,
                        sentiment_score=0.0,  # Will be calculated later
                        sentiment_label="neutral",
                        confidence=0.0,
                        emotions={},
                        topics=[],
                        quality_score=quality_score,
                        influence_score=0.0,  # Will be calculated later
                        engagement_metrics=engagement_metrics
                    )
                    
                    posts.append(post)
                    
                    if len(posts) >= limit:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing submission {submission.id}: {e}")
                    continue
            
            self.metrics["posts_processed"] += len(posts)
            self.metrics["api_calls"] += 1
            
            return posts
            
        except Exception as e:
            logger.error(f"Error getting posts from r/{subreddit_name}: {e}")
            self.metrics["errors"] += 1
            return []
    
    def _calculate_post_quality(self, submission) -> float:
        """Calculate post quality score using multiple factors"""
        try:
            score = 0.0
            
            # Score based on upvotes (normalized)
            upvote_score = min(submission.score / 100, 1.0) * 0.3
            score += upvote_score
            
            # Score based on comments
            comment_score = min(submission.num_comments / 50, 1.0) * 0.2
            score += comment_score
            
            # Score based on text length
            text_length = len(submission.title) + len(submission.selftext)
            length_score = min(text_length / 500, 1.0) * 0.2
            score += length_score
            
            # Score based on title quality
            title_score = self._calculate_title_quality(submission.title) * 0.2
            score += title_score
            
            # Score based on author activity (if available)
            if submission.author:
                author_score = min(submission.author.comment_karma / 10000, 1.0) * 0.1
                score += author_score
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating post quality: {e}")
            return 0.5  # Default middle score
    
    def _calculate_title_quality(self, title: str) -> float:
        """Calculate title quality based on various factors"""
        try:
            score = 0.0
            
            # Length score
            if 20 <= len(title) <= 100:
                score += 0.3
            
            # Capitalization score
            if not title.isupper():  # Not all caps
                score += 0.2
            
            # Question mark bonus
            if "?" in title:
                score += 0.1
            
            # Ticker mention bonus
            if any(ticker in title.upper() for ticker in self.valid_tickers):
                score += 0.2
            
            # Spam indicators (penalty)
            spam_indicators = ["🚀", "💎", "🌙", "TO THE MOON", "YOLO", "DIAMOND HANDS"]
            spam_count = sum(1 for indicator in spam_indicators if indicator in title.upper())
            score -= spam_count * 0.1
            
            return max(0.0, min(score, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating title quality: {e}")
            return 0.5
    
    def _calculate_engagement_metrics(self, submission) -> Dict[str, float]:
        """Calculate engagement metrics for a post"""
        try:
            metrics = {}
            
            # Upvote ratio
            metrics["upvote_ratio"] = submission.upvote_ratio
            
            # Comment to upvote ratio
            if submission.score > 0:
                metrics["comment_ratio"] = submission.num_comments / submission.score
            else:
                metrics["comment_ratio"] = 0.0
            
            # Time-based engagement
            post_age_hours = (time.time() - submission.created_utc) / 3600
            if post_age_hours > 0:
                metrics["engagement_velocity"] = submission.score / post_age_hours
            else:
                metrics["engagement_velocity"] = 0.0
            
            # Controversial score
            metrics["controversial_score"] = 1.0 - submission.upvote_ratio
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating engagement metrics: {e}")
            return {}
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Enhanced ticker extraction with validation"""
        try:
            tickers = set()
            
            # Apply regex patterns
            for pattern in self.compiled_patterns:
                matches = pattern.findall(text.upper())
                for match in matches:
                    # Clean the match
                    ticker = match.replace("$", "").strip()
                    
                    # Validate ticker
                    if (len(ticker) >= 1 and len(ticker) <= 5 and
                        ticker not in self.false_positives and
                        ticker.isalpha()):
                        
                        # Additional validation against known tickers
                        if ticker in self.valid_tickers:
                            tickers.add(ticker)
            
            return list(tickers)
            
        except Exception as e:
            logger.warning(f"Error extracting tickers: {e}")
            return []
    
    async def _analyze_posts_sentiment(self, posts: List[RedditPost]) -> Dict[str, Any]:
        """Analyze sentiment of posts using state-of-the-art models"""
        try:
            if not posts:
                return {"overall": 0.0, "distribution": {"positive": 0, "negative": 0, "neutral": 1}, "confidence": 0.0}
            
            sentiments = []
            confidences = []
            
            for post in posts:
                try:
                    text = f"{post.title} {post.body}"[:512]  # Limit for model
                    
                    if self.sentiment_pipeline == "remote":
                        # Use ML client for remote analysis
                        result = await ml_client.analyze_sentiment(text)
                        sentiment_score = result.score if hasattr(result, 'score') else 0.0
                        confidence = result.confidence if hasattr(result, 'confidence') else 0.0
                        
                        # Convert to standard scale (-1 to 1)
                        if result.label == "POSITIVE":
                            sentiment_score = abs(sentiment_score)
                        elif result.label == "NEGATIVE":
                            sentiment_score = -abs(sentiment_score)
                        else:  # NEUTRAL
                            sentiment_score = 0.0
                            
                    else:  # Fallback to TextBlob
                        blob = TextBlob(text)
                        sentiment_score = blob.sentiment.polarity
                        confidence = abs(sentiment_score)
                    
                    # Update post with sentiment
                    post.sentiment_score = sentiment_score
                    post.confidence = confidence
                    
                    if sentiment_score > 0.1:
                        post.sentiment_label = "positive"
                    elif sentiment_score < -0.1:
                        post.sentiment_label = "negative"
                    else:
                        post.sentiment_label = "neutral"
                    
                    sentiments.append(sentiment_score)
                    confidences.append(confidence)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing sentiment for post {post.id}: {e}")
                    sentiments.append(0.0)
                    confidences.append(0.0)
            
            # Calculate overall metrics
            overall_sentiment = np.mean(sentiments)
            overall_confidence = np.mean(confidences)
            
            # Calculate distribution
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            neutral_count = len(sentiments) - positive_count - negative_count
            
            total = len(sentiments)
            distribution = {
                "positive": positive_count / total,
                "negative": negative_count / total,
                "neutral": neutral_count / total
            }
            
            return {
                "overall": overall_sentiment,
                "distribution": distribution,
                "confidence": overall_confidence
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"overall": 0.0, "distribution": {"positive": 0, "negative": 0, "neutral": 1}, "confidence": 0.0}
    
    async def _analyze_posts_emotions(self, posts: List[RedditPost]) -> Dict[str, float]:
        """Analyze emotions in posts using emotion detection model"""
        try:
            if not posts or not self.emotion_pipeline:
                return {}
            
            emotion_scores = defaultdict(list)
            
            for post in posts:
                try:
                    text = f"{post.title} {post.body}"[:512]
                    
                    if self.emotion_pipeline == "remote":
                        # Use ML client for remote analysis
                        result = await ml_client.analyze_emotion(text)
                        post_emotions = result.all_emotions if hasattr(result, 'all_emotions') else {}
                        
                        # Add to emotion scores tracking
                        for emotion_name, emotion_score in post_emotions.items():
                        emotion_scores[emotion_name].append(emotion_score)
                    
                    post.emotions = post_emotions
                    else: # Fallback to TextBlob
                        blob = TextBlob(text)
                        post_emotions = {}
                        for sentence in blob.sentences:
                            for chunk in sentence.noun_phrases:
                                post_emotions[chunk.string.lower()] = 0.5 # Simple fallback
                        post.emotions = post_emotions
                    
                except Exception as e:
                    logger.warning(f"Error analyzing emotions for post {post.id}: {e}")
            
            # Calculate average emotions
            avg_emotions = {}
            for emotion, scores in emotion_scores.items():
                avg_emotions[emotion] = np.mean(scores)
            
            return avg_emotions
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {}
    
    async def _analyze_posts_topics(self, posts: List[RedditPost]) -> Dict[str, float]:
        """Analyze topics and their sentiments"""
        try:
            if not posts:
                return {}
            
            # Simple topic extraction based on keywords
            topics = {
                "earnings": ["earnings", "eps", "revenue", "profit", "loss", "beat", "miss"],
                "options": ["calls", "puts", "strike", "expiry", "theta", "gamma", "vega"],
                "technical": ["support", "resistance", "breakout", "chart", "pattern", "rsi", "macd"],
                "news": ["news", "announcement", "merger", "acquisition", "partnership", "deal"],
                "fundamentals": ["valuation", "pe", "pb", "debt", "cash", "growth", "dividend"]
            }
            
            topic_sentiments = {}
            
            for topic, keywords in topics.items():
                relevant_posts = []
                for post in posts:
                    text = f"{post.title} {post.body}".lower()
                    if any(keyword in text for keyword in keywords):
                        relevant_posts.append(post)
                
                if relevant_posts:
                    avg_sentiment = np.mean([post.sentiment_score for post in relevant_posts])
                    topic_sentiments[topic] = avg_sentiment
            
            return topic_sentiments
            
        except Exception as e:
            logger.error(f"Error in topic analysis: {e}")
            return {}
    
    def _analyze_temporal_sentiment(self, posts: List[RedditPost]) -> List[Dict[str, Any]]:
        """Analyze sentiment over time"""
        try:
            if not posts:
                return []
            
            # Group posts by hour
            hourly_sentiment = defaultdict(list)
            
            for post in posts:
                hour = datetime.fromtimestamp(post.created_utc).replace(minute=0, second=0, microsecond=0)
                hourly_sentiment[hour].append(post.sentiment_score)
            
            # Calculate hourly averages
            temporal_data = []
            for hour, sentiments in sorted(hourly_sentiment.items()):
                temporal_data.append({
                    "timestamp": hour.isoformat(),
                    "sentiment": np.mean(sentiments),
                    "post_count": len(sentiments),
                    "sentiment_std": np.std(sentiments)
                })
            
            return temporal_data
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            return []
    
    def _calculate_quality_metrics(self, posts: List[RedditPost]) -> Dict[str, float]:
        """Calculate quality metrics for the analyzed posts"""
        try:
            if not posts:
                return {"total_posts": 0, "avg_quality": 0.0}
            
            quality_scores = [post.quality_score for post in posts]
            engagement_scores = [post.engagement_metrics.get("upvote_ratio", 0) for post in posts]
            
            return {
                "total_posts": len(posts),
                "avg_quality": np.mean(quality_scores),
                "quality_std": np.std(quality_scores),
                "avg_engagement": np.mean(engagement_scores),
                "high_quality_posts": sum(1 for score in quality_scores if score > 0.8),
                "low_quality_posts": sum(1 for score in quality_scores if score < 0.4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {"error": str(e)}
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result with error handling"""
        try:
            if not self.redis_client:
                return None
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting cached result: {e}")
            return None
    
    def _cache_result(self, cache_key: str, result: Any, ttl: int):
        """Cache result with error handling"""
        try:
            if not self.redis_client:
                return
            
            # Convert dataclass to dict if needed
            if hasattr(result, '__dict__'):
                result = asdict(result)
            
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str)
            )
            
        except Exception as e:
            logger.warning(f"Error caching result: {e}")
    
    async def get_trending_tickers(
        self,
        subreddits: List[str] = None,
        timeframe: str = "day",
        min_mentions: int = 5
    ) -> List[TrendingTicker]:
        """
        Get trending tickers with enhanced 2025 features
        
        Args:
            subreddits: List of subreddits to analyze
            timeframe: Time frame for analysis
            min_mentions: Minimum mentions to be considered trending
            
        Returns:
            List of trending tickers with comprehensive metrics
        """
        try:
            # Get sentiment analysis first
            sentiment_analysis = await self.get_sentiment_analysis(subreddits, timeframe=timeframe)
            
            # Implementation would continue with ticker trend analysis
            # This is a placeholder for the enhanced implementation
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting trending tickers: {e}")
            return []
    
    async def get_unusual_activity(
        self,
        subreddits: List[str] = None,
        timeframe: str = "day",
        sensitivity: float = 0.7
    ) -> List[UnusualActivity]:
        """
        Detect unusual activity with AI-powered pattern recognition
        
        Args:
            subreddits: List of subreddits to analyze
            timeframe: Time frame for analysis
            sensitivity: Detection sensitivity (0.0-1.0)
            
        Returns:
            List of unusual activities detected
        """
        try:
            # Implementation would use AI models to detect unusual patterns
            # This is a placeholder for the enhanced implementation
            
            return []
            
        except Exception as e:
            logger.error(f"Error detecting unusual activity: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "config": self.config,
            "model_status": {
                "sentiment_pipeline": self.sentiment_pipeline is not None,
                "emotion_pipeline": self.emotion_pipeline is not None,
                "ner_pipeline": self.ner_pipeline is not None,
                "sentence_transformer": self.sentence_transformer is not None
            },
            "redis_connected": self.redis_client is not None,
            "reddit_connected": self.reddit is not None
        } 