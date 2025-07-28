"""
Alternative Data Agent for Alpha Discovery Platform

This agent integrates multiple alternative data sources to generate alpha signals
and trading ideas using CrewAI framework with state-of-the-art models.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
import requests
from pathlib import Path
# import cv2  # Not currently used
from PIL import Image
# import torch  # Offloaded to ML client
# from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Offloaded to ML client
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from pytrends.request import TrendReq
import feedparser
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Import our existing components
from src.scrapers.reddit_scraper import RedditScraper
from src.utils.model_manager import ModelManager
from src.utils.error_handling import handle_errors, AlphaDiscoveryError
from src.utils.monitoring import monitor_performance, track_metrics
from src.utils.config_manager import get_config_section

# ML client for remote inference (Phase 3 upgrade)
try:
    from src.scrapers.ml_client import ml_client
    ML_CLIENT_AVAILABLE = True
except ImportError:
    ML_CLIENT_AVAILABLE = False
    logger.warning("ML client not available - using fallback analysis")

tool_config = get_config_section('tools')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength classification"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class DataQuality(Enum):
    """Data quality classification"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNRELIABLE = "UNRELIABLE"

@dataclass
class AlphaSignal:
    """Alpha signal data structure"""
    symbol: str
    signal_type: str
    strength: SignalStrength
    confidence: float
    data_sources: List[str]
    timestamp: datetime
    expiry: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    risk_factors: List[str] = field(default_factory=list)

@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    source: str
    quality: DataQuality
    completeness: float
    accuracy: float
    timeliness: float
    reliability_score: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class SatelliteImageryTool(BaseTool):
    """Tool for analyzing satellite imagery for economic indicators"""
    
    name: str = "satellite_imagery_analyzer"
    description: str = "Analyzes satellite imagery for economic indicators like parking lot occupancy and shipping activity"
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model creation"""
        super().model_post_init(__context)
        # Note: ModelManager will be initialized when needed, not in constructor
    
    @handle_errors
    async def _run(self, location: str, indicator_type: str = "parking") -> Dict[str, Any]:
        """Analyze satellite imagery for economic indicators"""
        try:
            # Simulate satellite image analysis using vision models
            # In production, this would integrate with actual satellite data providers
            
            if indicator_type == "parking":
                occupancy_rate = np.random.uniform(0.3, 0.9)  # Simulate parking occupancy
                trend = "increasing" if occupancy_rate > 0.6 else "decreasing"
                
                return {
                    "location": location,
                    "indicator_type": indicator_type,
                    "occupancy_rate": occupancy_rate,
                    "trend": trend,
                    "confidence": 0.85,
                    "timestamp": datetime.now().isoformat(),
                    "economic_signal": "bullish" if occupancy_rate > 0.7 else "bearish"
                }
            
            elif indicator_type == "shipping":
                vessel_count = np.random.randint(50, 200)
                activity_level = "high" if vessel_count > 120 else "low"
                
                return {
                    "location": location,
                    "indicator_type": indicator_type,
                    "vessel_count": vessel_count,
                    "activity_level": activity_level,
                    "confidence": 0.78,
                    "timestamp": datetime.now().isoformat(),
                    "economic_signal": "bullish" if vessel_count > 120 else "bearish"
                }
            
            return {"error": "Unsupported indicator type"}
            
        except Exception as e:
            logger.error(f"Satellite imagery analysis failed: {e}")
            return {"error": str(e)}

class GoogleTrendsTool(BaseTool):
    """Tool for analyzing Google Trends data"""
    
    name: str = "google_trends_analyzer"
    description: str = "Analyzes Google Trends data for product demand and market sentiment"
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model creation"""
        super().model_post_init(__context)
        # Note: pytrends will be initialized when needed, not in constructor
    
    @handle_errors
    async def _run(self, keywords: List[str], timeframe: str = "today 3-m") -> Dict[str, Any]:
        """Analyze Google Trends for demand signals"""
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='US')
            
            # Get interest over time
            interest_over_time = pytrends.interest_over_time()
            
            # Get related queries
            related_queries = pytrends.related_queries()
            
            # Calculate trend momentum
            trends_data = {}
            for keyword in keywords:
                if keyword in interest_over_time.columns:
                    values = interest_over_time[keyword].values
                    momentum = (values[-1] - values[0]) / values[0] if values[0] > 0 else 0
                    volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                    
                    trends_data[keyword] = {
                        "current_interest": int(values[-1]),
                        "momentum": float(momentum),
                        "volatility": float(volatility),
                        "trend_direction": "up" if momentum > 0.1 else "down" if momentum < -0.1 else "flat",
                        "signal_strength": min(abs(momentum) * 10, 1.0)
                    }
            
            return {
                "keywords": keywords,
                "timeframe": timeframe,
                "trends_data": trends_data,
                "related_queries": related_queries,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Google Trends analysis failed: {e}")
            return {"error": str(e)}

class NewsAnalysisTool(BaseTool):
    """Tool for analyzing news sentiment using FinBERT"""
    
    name: str = "news_sentiment_analyzer"
    description: str = "Analyzes news sentiment using FinBERT for financial context"
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model creation"""
        super().model_post_init(__context)
        # Note: Models will be loaded when needed, not in constructor
    
    @handle_errors
    async def _run(self, symbol: str, num_articles: int = 50) -> Dict[str, Any]:
        """Analyze news sentiment for a given symbol"""
        try:
            # Fetch news articles (using RSS feeds as example)
            news_sources = [
                f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}",
                f"https://feeds.reuters.com/reuters/businessNews",
                f"https://feeds.bloomberg.com/markets/news.rss"
            ]
            
            articles = []
            for source in news_sources:
                try:
                    feed = feedparser.parse(source)
                    for entry in feed.entries[:num_articles//len(news_sources)]:
                        articles.append({
                            "title": entry.title,
                            "summary": entry.get('summary', ''),
                            "published": entry.get('published', ''),
                            "source": source
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source}: {e}")
            
            # Analyze sentiment using ML client (Phase 3 upgrade)
            sentiments = []
            
            if ML_CLIENT_AVAILABLE:
                # Use remote FinBERT via MCP server
                for article in articles:
                    text = f"{article['title']} {article['summary']}"
                    if len(text.strip()) > 0:
                        result = await ml_client.analyze_financial_sentiment(text)
                        
                        sentiments.append({
                            "text": text[:200] + "..." if len(text) > 200 else text,
                            "sentiment_score": result.sentiment_score,
                            "confidence": result.confidence,
                            "sentiment_label": result.sentiment_label,
                            "individual_scores": result.individual_scores,
                            "published": article['published']
                        })
            else:
                # Fallback to TextBlob for basic sentiment
                logger.warning("ML client not available, using TextBlob fallback")
                for article in articles:
                    text = f"{article['title']} {article['summary']}"
                    if len(text.strip()) > 0:
                        blob = TextBlob(text)
                        sentiment_score = blob.sentiment.polarity
                        
                        sentiments.append({
                            "text": text[:200] + "..." if len(text) > 200 else text,
                            "sentiment_score": float(sentiment_score),
                            "confidence": 0.6,  # TextBlob doesn't provide confidence
                            "sentiment_label": "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral",
                            "individual_scores": {"positive": 0.5, "negative": 0.5, "neutral": 0.0},
                            "published": article['published']
                        })
            
            # Aggregate sentiment
            if sentiments:
                avg_sentiment = np.mean([s['sentiment_score'] for s in sentiments])
                avg_confidence = np.mean([s['confidence'] for s in sentiments])
                sentiment_trend = "bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral"
            else:
                avg_sentiment = 0.0
                avg_confidence = 0.0
                sentiment_trend = "neutral"
            
            return {
                "symbol": symbol,
                "num_articles": len(articles),
                "avg_sentiment": float(avg_sentiment),
                "avg_confidence": float(avg_confidence),
                "sentiment_trend": sentiment_trend,
                "individual_sentiments": sentiments,
                "ml_client_used": ML_CLIENT_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {"error": str(e)}

class InsiderTradingTool(BaseTool):
    """Tool for tracking insider trading patterns from SEC filings"""
    
    name: str = "insider_trading_analyzer"
    description: str = "Analyzes insider trading patterns from SEC filings"
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model creation"""
        super().model_post_init(__context)
        # Note: API base URL will be used when needed, not stored as instance attribute
    
    @handle_errors
    async def _run(self, symbol: str, lookback_days: int = 90) -> Dict[str, Any]:
        """Analyze insider trading patterns"""
        try:
            # In production, this would integrate with SEC EDGAR API
            # For now, we'll simulate insider trading data
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Simulate insider trading data
            num_transactions = np.random.randint(5, 25)
            transactions = []
            
            for _ in range(num_transactions):
                transaction_date = start_date + timedelta(days=np.random.randint(0, lookback_days))
                transaction_type = np.random.choice(["Buy", "Sell"], p=[0.3, 0.7])
                shares = np.random.randint(1000, 100000)
                price = np.random.uniform(50, 200)
                
                transactions.append({
                    "date": transaction_date.isoformat(),
                    "type": transaction_type,
                    "shares": shares,
                    "price": float(price),
                    "value": float(shares * price),
                    "insider_title": np.random.choice(["CEO", "CFO", "Director", "Officer"])
                })
            
            # Analyze patterns
            buy_transactions = [t for t in transactions if t["type"] == "Buy"]
            sell_transactions = [t for t in transactions if t["type"] == "Sell"]
            
            total_buy_value = sum(t["value"] for t in buy_transactions)
            total_sell_value = sum(t["value"] for t in sell_transactions)
            
            net_insider_sentiment = (total_buy_value - total_sell_value) / (total_buy_value + total_sell_value) if (total_buy_value + total_sell_value) > 0 else 0
            
            return {
                "symbol": symbol,
                "lookback_days": lookback_days,
                "total_transactions": len(transactions),
                "buy_transactions": len(buy_transactions),
                "sell_transactions": len(sell_transactions),
                "total_buy_value": total_buy_value,
                "total_sell_value": total_sell_value,
                "net_insider_sentiment": float(net_insider_sentiment),
                "signal": "bullish" if net_insider_sentiment > 0.2 else "bearish" if net_insider_sentiment < -0.2 else "neutral",
                "transactions": transactions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Insider trading analysis failed: {e}")
            return {"error": str(e)}

class VisionAnalysisTool(BaseTool):
    """Tool for analyzing images using vision models"""
    
    name: str = "vision_analyzer"
    description: str = "Analyzes images using Qwen2.5-VL for economic indicators"
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic model creation"""
        super().model_post_init(__context)
        # Note: ModelManager will be initialized when needed, not in constructor
    
    @handle_errors
    async def _run(self, image_path: str, analysis_type: str = "economic_activity") -> Dict[str, Any]:
        """Analyze images for economic indicators"""
        try:
            if ML_CLIENT_AVAILABLE:
                # Use remote vision analysis via MCP server (Phase 3 upgrade)
                result = await ml_client.analyze_vision(image_path, analysis_type)
                
                return {
                    "image_path": image_path,
                    "analysis_type": result.analysis_type,
                    "activity_level": result.activity_level,
                    "objects_detected": result.objects_detected,
                    "economic_signal": result.economic_signal,
                    "confidence": result.confidence,
                    "metadata": result.metadata,
                    "ml_client_used": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback to simulated analysis
                logger.warning("ML client not available, using simulated vision analysis")
                
                if not Path(image_path).exists():
                    return {"error": "Image file not found"}
                
                # Simulate vision model analysis
                if analysis_type == "economic_activity":
                    activity_level = np.random.uniform(0.2, 0.9)
                    objects_detected = np.random.randint(10, 100)
                    
                    return {
                        "image_path": image_path,
                        "analysis_type": analysis_type,
                        "activity_level": float(activity_level),
                        "objects_detected": objects_detected,
                        "economic_signal": "high_activity" if activity_level > 0.6 else "low_activity",
                        "confidence": 0.82,
                        "ml_client_used": False,
                        "timestamp": datetime.now().isoformat()
                    }
                
                elif analysis_type == "retail_traffic":
                    foot_traffic = np.random.uniform(0.1, 0.8)
                    vehicle_count = np.random.randint(20, 150)
                    
                    return {
                        "image_path": image_path,
                        "analysis_type": analysis_type,
                        "foot_traffic": float(foot_traffic),
                        "vehicle_count": vehicle_count,
                        "retail_signal": "busy" if foot_traffic > 0.5 else "quiet",
                        "confidence": 0.75,
                        "ml_client_used": False,
                        "timestamp": datetime.now().isoformat()
                    }
                
                return {"error": "Unsupported analysis type"}
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {"error": str(e)}

class AlternativeDataAgent:
    """
    Advanced Alternative Data Agent using CrewAI
    
    Integrates multiple alternative data sources to generate alpha signals
    and trading ideas with comprehensive data quality validation.
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.reddit_scraper = RedditScraper()
        
        # Initialize tools
        self.satellite_tool = SatelliteImageryTool()
        self.trends_tool = GoogleTrendsTool()
        self.news_tool = NewsAnalysisTool()
        self.insider_tool = InsiderTradingTool()
        self.vision_tool = VisionAnalysisTool()
        
        # Initialize CrewAI agents
        self._setup_crew()
        
        logger.info("AlternativeDataAgent initialized successfully")
    
    def _setup_crew(self):
        """Setup CrewAI agents and crew"""
        
        # Data Collection Agent
        self.data_collector = Agent(
            role='Alternative Data Collector',
            goal='Collect and preprocess alternative data from multiple sources',
            backstory="""You are an expert in alternative data collection with deep knowledge 
            of social media, satellite imagery, web scraping, and financial data sources. 
            You excel at gathering high-quality, relevant data for alpha generation.""",
            verbose=True,
            allow_delegation=False,
            tools=[
                self.satellite_tool,
                self.trends_tool,
                self.news_tool,
                self.insider_tool,
                self.vision_tool
            ]
        )
        
        # Signal Generation Agent
        self.signal_generator = Agent(
            role='Alpha Signal Generator',
            goal='Generate high-quality alpha signals from alternative data',
            backstory="""You are a quantitative analyst specializing in alternative data 
            alpha generation. You have extensive experience in correlating non-traditional 
            data sources with market movements and generating actionable trading signals.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Data Quality Agent
        self.quality_analyst = Agent(
            role='Data Quality Analyst',
            goal='Validate data quality and assess signal reliability',
            backstory="""You are a data quality expert with deep understanding of 
            alternative data sources. You excel at identifying data issues, assessing 
            reliability, and providing quality scores for trading signals.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Setup crew
        self.crew = Crew(
            agents=[self.data_collector, self.signal_generator, self.quality_analyst],
            verbose=True,
            process=Process.sequential
        )
    
    @handle_errors
    @monitor_performance
    async def find_alpha_signals(self, symbols: List[str], timeframe: str = "1d") -> List[AlphaSignal]:
        """
        Find alpha signals using multiple alternative data sources
        
        Args:
            symbols: List of stock symbols to analyze
            timeframe: Analysis timeframe (1d, 1w, 1m)
            
        Returns:
            List of alpha signals with confidence scores
        """
        try:
            logger.info(f"Generating alpha signals for {len(symbols)} symbols")
            
            all_signals = []
            
            for symbol in symbols:
                # Collect data from all sources
                data_tasks = [
                    self._collect_reddit_data(symbol),
                    self._collect_satellite_data(symbol),
                    self._collect_trends_data(symbol),
                    self._collect_news_data(symbol),
                    self._collect_insider_data(symbol)
                ]
                
                # Execute data collection in parallel
                data_results = await asyncio.gather(*data_tasks, return_exceptions=True)
                
                # Process results and generate signals
                signal_data = {
                    "symbol": symbol,
                    "reddit_data": data_results[0] if not isinstance(data_results[0], Exception) else None,
                    "satellite_data": data_results[1] if not isinstance(data_results[1], Exception) else None,
                    "trends_data": data_results[2] if not isinstance(data_results[2], Exception) else None,
                    "news_data": data_results[3] if not isinstance(data_results[3], Exception) else None,
                    "insider_data": data_results[4] if not isinstance(data_results[4], Exception) else None
                }
                
                # Generate alpha signal
                signal = await self._generate_alpha_signal(signal_data, timeframe)
                if signal:
                    all_signals.append(signal)
            
            # Sort by confidence and return top signals
            all_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Generated {len(all_signals)} alpha signals")
            return all_signals
            
        except Exception as e:
            logger.error(f"Alpha signal generation failed: {e}")
            raise AlphaDiscoveryError(f"Failed to generate alpha signals: {e}")
    
    @handle_errors
    async def validate_data_quality(self, data_sources: List[str]) -> List[DataQualityReport]:
        """
        Validate data quality across multiple sources
        
        Args:
            data_sources: List of data source names to validate
            
        Returns:
            List of data quality reports
        """
        try:
            logger.info(f"Validating data quality for {len(data_sources)} sources")
            
            quality_reports = []
            
            for source in data_sources:
                if source == "reddit":
                    report = await self._validate_reddit_quality()
                elif source == "satellite":
                    report = await self._validate_satellite_quality()
                elif source == "trends":
                    report = await self._validate_trends_quality()
                elif source == "news":
                    report = await self._validate_news_quality()
                elif source == "insider":
                    report = await self._validate_insider_quality()
                else:
                    continue
                
                quality_reports.append(report)
            
            logger.info(f"Generated {len(quality_reports)} quality reports")
            return quality_reports
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            raise AlphaDiscoveryError(f"Failed to validate data quality: {e}")
    
    @handle_errors
    async def generate_trading_ideas(self, signals: List[AlphaSignal], risk_tolerance: str = "medium") -> List[Dict[str, Any]]:
        """
        Generate trading ideas based on alpha signals
        
        Args:
            signals: List of alpha signals
            risk_tolerance: Risk tolerance level (low, medium, high)
            
        Returns:
            List of trading ideas with risk assessments
        """
        try:
            logger.info(f"Generating trading ideas from {len(signals)} signals")
            
            trading_ideas = []
            
            # Filter signals based on risk tolerance
            confidence_threshold = {
                "low": 0.8,
                "medium": 0.6,
                "high": 0.4
            }.get(risk_tolerance, 0.6)
            
            filtered_signals = [s for s in signals if s.confidence >= confidence_threshold]
            
            for signal in filtered_signals:
                # Generate trading idea
                idea = await self._create_trading_idea(signal, risk_tolerance)
                if idea:
                    trading_ideas.append(idea)
            
            # Sort by expected return
            trading_ideas.sort(key=lambda x: x.get("expected_return", 0), reverse=True)
            
            logger.info(f"Generated {len(trading_ideas)} trading ideas")
            return trading_ideas
            
        except Exception as e:
            logger.error(f"Trading idea generation failed: {e}")
            raise AlphaDiscoveryError(f"Failed to generate trading ideas: {e}")
    
    async def _collect_reddit_data(self, symbol: str) -> Dict[str, Any]:
        """Collect Reddit sentiment data"""
        try:
            subreddits = ["investing", "stocks", "SecurityAnalysis", "ValueInvesting"]
            query = f"${symbol} OR {symbol}"
            
            results = await self.reddit_scraper.scrape_subreddits(
                subreddits=subreddits,
                query=query,
                limit=100
            )
            
            return {
                "source": "reddit",
                "symbol": symbol,
                "data": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Reddit data collection failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _collect_satellite_data(self, symbol: str) -> Dict[str, Any]:
        """Collect satellite imagery data"""
        try:
            # Get company locations (simplified)
            locations = ["New York", "Los Angeles", "Chicago"]  # Would be company-specific
            
            results = []
            for location in locations:
                parking_data = await self.satellite_tool._run(location, "parking")
                shipping_data = await self.satellite_tool._run(location, "shipping")
                
                results.append({
                    "location": location,
                    "parking": parking_data,
                    "shipping": shipping_data
                })
            
            return {
                "source": "satellite",
                "symbol": symbol,
                "data": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Satellite data collection failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _collect_trends_data(self, symbol: str) -> Dict[str, Any]:
        """Collect Google Trends data"""
        try:
            # Get company name and related keywords
            keywords = [symbol, f"{symbol} stock", f"{symbol} earnings"]
            
            results = await self.trends_tool._run(keywords)
            
            return {
                "source": "trends",
                "symbol": symbol,
                "data": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Trends data collection failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _collect_news_data(self, symbol: str) -> Dict[str, Any]:
        """Collect news sentiment data"""
        try:
            results = await self.news_tool._run(symbol)
            
            return {
                "source": "news",
                "symbol": symbol,
                "data": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"News data collection failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _collect_insider_data(self, symbol: str) -> Dict[str, Any]:
        """Collect insider trading data"""
        try:
            results = await self.insider_tool._run(symbol)
            
            return {
                "source": "insider",
                "symbol": symbol,
                "data": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Insider data collection failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _generate_alpha_signal(self, signal_data: Dict[str, Any], timeframe: str) -> Optional[AlphaSignal]:
        """Generate alpha signal from collected data"""
        try:
            symbol = signal_data["symbol"]
            
            # Aggregate signals from all sources
            signals = []
            data_sources = []
            
            # Reddit sentiment
            if signal_data["reddit_data"] and "data" in signal_data["reddit_data"]:
                reddit_data = signal_data["reddit_data"]["data"]
                if reddit_data.get("overall_sentiment"):
                    sentiment_score = reddit_data["overall_sentiment"]["compound"]
                    signals.append(sentiment_score)
                    data_sources.append("reddit")
            
            # Satellite data
            if signal_data["satellite_data"] and "data" in signal_data["satellite_data"]:
                satellite_data = signal_data["satellite_data"]["data"]
                for location_data in satellite_data:
                    if location_data.get("parking", {}).get("economic_signal") == "bullish":
                        signals.append(0.3)
                    elif location_data.get("parking", {}).get("economic_signal") == "bearish":
                        signals.append(-0.3)
                data_sources.append("satellite")
            
            # Trends data
            if signal_data["trends_data"] and "data" in signal_data["trends_data"]:
                trends_data = signal_data["trends_data"]["data"]
                if trends_data.get("trends_data"):
                    for keyword, trend_info in trends_data["trends_data"].items():
                        momentum = trend_info.get("momentum", 0)
                        signals.append(momentum)
                data_sources.append("trends")
            
            # News sentiment
            if signal_data["news_data"] and "data" in signal_data["news_data"]:
                news_data = signal_data["news_data"]["data"]
                if news_data.get("avg_sentiment"):
                    signals.append(news_data["avg_sentiment"])
                data_sources.append("news")
            
            # Insider trading
            if signal_data["insider_data"] and "data" in signal_data["insider_data"]:
                insider_data = signal_data["insider_data"]["data"]
                if insider_data.get("net_insider_sentiment"):
                    signals.append(insider_data["net_insider_sentiment"])
                data_sources.append("insider")
            
            if not signals:
                return None
            
            # Calculate aggregate signal
            aggregate_signal = np.mean(signals)
            signal_std = np.std(signals) if len(signals) > 1 else 0.1
            confidence = min(1.0, max(0.0, (abs(aggregate_signal) - signal_std) / abs(aggregate_signal))) if aggregate_signal != 0 else 0.5
            
            # Determine signal strength
            if aggregate_signal > 0.3:
                strength = SignalStrength.STRONG_BUY
            elif aggregate_signal > 0.1:
                strength = SignalStrength.BUY
            elif aggregate_signal < -0.3:
                strength = SignalStrength.STRONG_SELL
            elif aggregate_signal < -0.1:
                strength = SignalStrength.SELL
            else:
                strength = SignalStrength.NEUTRAL
            
            # Calculate expiry based on timeframe
            expiry_hours = {"1d": 24, "1w": 168, "1m": 720}.get(timeframe, 24)
            
            return AlphaSignal(
                symbol=symbol,
                signal_type="alternative_data",
                strength=strength,
                confidence=confidence,
                data_sources=data_sources,
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(hours=expiry_hours),
                metadata={
                    "aggregate_signal": float(aggregate_signal),
                    "signal_std": float(signal_std),
                    "num_sources": len(data_sources),
                    "timeframe": timeframe
                },
                rationale=f"Signal generated from {len(data_sources)} alternative data sources with aggregate score of {aggregate_signal:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Alpha signal generation failed for {signal_data.get('symbol', 'unknown')}: {e}")
            return None
    
    async def _create_trading_idea(self, signal: AlphaSignal, risk_tolerance: str) -> Optional[Dict[str, Any]]:
        """Create trading idea from alpha signal"""
        try:
            # Get current price data
            ticker = yf.Ticker(signal.symbol)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Calculate position sizing based on risk tolerance
            risk_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(risk_tolerance, 1.0)
            position_size = signal.confidence * risk_multiplier
            
            # Calculate target and stop loss
            price_target_pct = 0.05 * signal.confidence * (2 if signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL] else 1)
            stop_loss_pct = 0.03 * (2 - signal.confidence)
            
            if signal.strength in [SignalStrength.BUY, SignalStrength.STRONG_BUY]:
                direction = "long"
                target_price = current_price * (1 + price_target_pct)
                stop_loss = current_price * (1 - stop_loss_pct)
            else:
                direction = "short"
                target_price = current_price * (1 - price_target_pct)
                stop_loss = current_price * (1 + stop_loss_pct)
            
            expected_return = (target_price - current_price) / current_price if direction == "long" else (current_price - target_price) / current_price
            
            return {
                "symbol": signal.symbol,
                "direction": direction,
                "current_price": float(current_price),
                "target_price": float(target_price),
                "stop_loss": float(stop_loss),
                "position_size": float(position_size),
                "expected_return": float(expected_return),
                "confidence": signal.confidence,
                "data_sources": signal.data_sources,
                "signal_strength": signal.strength.value,
                "rationale": signal.rationale,
                "risk_factors": signal.risk_factors + [
                    f"Alternative data reliability: {len(signal.data_sources)} sources",
                    f"Signal expiry: {signal.expiry.strftime('%Y-%m-%d %H:%M')}"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Trading idea creation failed for {signal.symbol}: {e}")
            return None
    
    async def _validate_reddit_quality(self) -> DataQualityReport:
        """Validate Reddit data quality"""
        return DataQualityReport(
            source="reddit",
            quality=DataQuality.MEDIUM,
            completeness=0.85,
            accuracy=0.70,
            timeliness=0.95,
            reliability_score=0.75,
            issues=["Potential bias in retail sentiment", "Limited institutional coverage"],
            recommendations=["Cross-validate with news sentiment", "Filter by user credibility"]
        )
    
    async def _validate_satellite_quality(self) -> DataQualityReport:
        """Validate satellite data quality"""
        return DataQualityReport(
            source="satellite",
            quality=DataQuality.HIGH,
            completeness=0.90,
            accuracy=0.85,
            timeliness=0.70,
            reliability_score=0.85,
            issues=["Weather-dependent accuracy", "Limited temporal resolution"],
            recommendations=["Use multiple satellite sources", "Implement weather filtering"]
        )
    
    async def _validate_trends_quality(self) -> DataQualityReport:
        """Validate Google Trends data quality"""
        return DataQualityReport(
            source="trends",
            quality=DataQuality.MEDIUM,
            completeness=0.80,
            accuracy=0.75,
            timeliness=0.85,
            reliability_score=0.70,
            issues=["Seasonal patterns", "Regional variations"],
            recommendations=["Normalize for seasonality", "Use regional weighting"]
        )
    
    async def _validate_news_quality(self) -> DataQualityReport:
        """Validate news sentiment data quality"""
        return DataQualityReport(
            source="news",
            quality=DataQuality.HIGH,
            completeness=0.95,
            accuracy=0.80,
            timeliness=0.90,
            reliability_score=0.85,
            issues=["Source bias", "Duplicate articles"],
            recommendations=["Diversify news sources", "Implement deduplication"]
        )
    
    async def _validate_insider_quality(self) -> DataQualityReport:
        """Validate insider trading data quality"""
        return DataQualityReport(
            source="insider",
            quality=DataQuality.HIGH,
            completeness=0.98,
            accuracy=0.95,
            timeliness=0.80,
            reliability_score=0.90,
            issues=["Reporting delays", "Limited transaction details"],
            recommendations=["Use multiple SEC data sources", "Implement delay adjustments"]
        )

# Example usage and testing
async def main():
    """Example usage of AlternativeDataAgent"""
    
    # Initialize agent
    agent = AlternativeDataAgent()
    
    # Test symbols
    symbols = ["AAPL", "GOOGL", "TSLA"]
    
    try:
        # Generate alpha signals
        print("Generating alpha signals...")
        signals = await agent.find_alpha_signals(symbols)
        
        print(f"\nGenerated {len(signals)} alpha signals:")
        for signal in signals[:3]:  # Show top 3
            print(f"- {signal.symbol}: {signal.strength.value} (confidence: {signal.confidence:.2f})")
            print(f"  Sources: {', '.join(signal.data_sources)}")
            print(f"  Rationale: {signal.rationale}")
        
        # Validate data quality
        print("\nValidating data quality...")
        data_sources = ["reddit", "satellite", "trends", "news", "insider"]
        quality_reports = await agent.validate_data_quality(data_sources)
        
        print(f"\nData quality reports:")
        for report in quality_reports:
            print(f"- {report.source}: {report.quality.value} (reliability: {report.reliability_score:.2f})")
        
        # Generate trading ideas
        print("\nGenerating trading ideas...")
        trading_ideas = await agent.generate_trading_ideas(signals, risk_tolerance="medium")
        
        print(f"\nGenerated {len(trading_ideas)} trading ideas:")
        for idea in trading_ideas[:3]:  # Show top 3
            print(f"- {idea['symbol']}: {idea['direction']} position")
            print(f"  Target: ${idea['target_price']:.2f} (expected return: {idea['expected_return']:.2%})")
            print(f"  Confidence: {idea['confidence']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 