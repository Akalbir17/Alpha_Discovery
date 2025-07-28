"""
Alpha Discovery MCP Server using official MCP SDK
Provides tools for market data, sentiment analysis, risk assessment, and trading operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import time
import hashlib

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent, ImageContent
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    change: float
    change_percent: float

@dataclass
class SentimentData:
    """Sentiment analysis data structure"""
    symbol: str
    sentiment_score: float
    confidence: float
    sources: List[str]
    timestamp: datetime

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    symbol: str
    volatility: float
    beta: float
    var_95: float
    sharpe_ratio: float
    max_drawdown: float

@dataclass
class AlphaDiscoveryData:
    """Alpha discovery data structure"""
    symbol: str
    alpha_score: float
    confidence: float
    factors: Dict[str, float]
    timestamp: datetime

class AlphaDiscoveryContext:
    """Application context for Alpha Discovery MCP server"""
    
    def __init__(self):
        self.market_data_cache: Dict[str, MarketData] = {}
        self.sentiment_cache: Dict[str, SentimentData] = {}
        self.risk_cache: Dict[str, RiskMetrics] = {}
        self.alpha_discoveries: List[AlphaDiscoveryData] = []
        self.active_positions: Dict[str, Dict] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.cache_timestamps: Dict[str, float] = {}
        self.start_time = datetime.now()
        
    async def initialize(self):
        """Initialize the context"""
        logger.info("Initializing Alpha Discovery MCP context")
        # Initialize any required resources
        
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Alpha Discovery MCP context")
        
    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate a cache key"""
        key_parts = [prefix] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()
        
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        return time.time() - self.cache_timestamps[cache_key] < self.cache_ttl
        
    def _update_cache_timestamp(self, cache_key: str):
        """Update cache timestamp"""
        self.cache_timestamps[cache_key] = time.time()

# Global context instance
app_context = AlphaDiscoveryContext()

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AlphaDiscoveryContext]:
    """Manage application lifecycle"""
    await app_context.initialize()
    try:
        yield app_context
    finally:
        await app_context.cleanup()

# Create FastMCP server with lifespan management
mcp = FastMCP("AlphaDiscoveryServer", lifespan=lifespan)

# Market Data Tools
@mcp.tool()
async def get_market_data(symbol: str, ctx: Context, period: str = "1d") -> Dict[str, Any]:
    """
    Get market data for a symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL, GOOGL)
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    """
    try:
        # Check cache first
        cache_key = app_context._get_cache_key("market_data", symbol=symbol, period=period)
        if app_context._is_cache_valid(cache_key) and symbol in app_context.market_data_cache:
            await ctx.info(f"Returning cached market data for {symbol}")
            cached_data = app_context.market_data_cache[symbol]
            return {
                "symbol": symbol,
                "price": cached_data.price,
                "volume": cached_data.volume,
                "change": cached_data.change,
                "change_percent": cached_data.change_percent,
                "timestamp": cached_data.timestamp.isoformat(),
                "cached": True
            }
        
        await ctx.info(f"Fetching market data for {symbol}")
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for symbol {symbol}"}
            
        latest = hist.iloc[-1]
        previous = hist.iloc[-2] if len(hist) > 1 else latest
        
        change = latest['Close'] - previous['Close']
        change_percent = (change / previous['Close']) * 100
        
        market_data = MarketData(
            symbol=symbol,
            price=float(latest['Close']),
            volume=int(latest['Volume']),
            timestamp=datetime.now(),
            change=float(change),
            change_percent=float(change_percent)
        )
        
        # Cache the data
        app_context.market_data_cache[symbol] = market_data
        app_context._update_cache_timestamp(cache_key)
        
        return {
            "symbol": symbol,
            "price": market_data.price,
            "volume": market_data.volume,
            "change": market_data.change,
            "change_percent": market_data.change_percent,
            "timestamp": market_data.timestamp.isoformat(),
            "cached": False
        }
        
    except Exception as e:
        await ctx.error(f"Error fetching market data: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_technical_indicators(symbol: str, ctx: Context, period: str = "1mo") -> Dict[str, Any]:
    """
    Calculate technical indicators for a symbol
    
    Args:
        symbol: Stock symbol
        period: Time period for analysis
    """
    try:
        # Check cache first
        cache_key = app_context._get_cache_key("technical_indicators", symbol=symbol, period=period)
        if app_context._is_cache_valid(cache_key):
            await ctx.info(f"Returning cached technical indicators for {symbol}")
            return {"cached": True, "message": "Cached data available"}
        
        await ctx.info(f"Calculating technical indicators for {symbol}")
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for symbol {symbol}"}
        
        # Calculate technical indicators
        close_prices = hist['Close']
        
        # Simple Moving Averages
        sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Bollinger Bands
        sma_20_series = close_prices.rolling(window=20).mean()
        std_20 = close_prices.rolling(window=20).std()
        upper_band = (sma_20_series + (std_20 * 2)).iloc[-1]
        lower_band = (sma_20_series - (std_20 * 2)).iloc[-1]
        
        # MACD
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        
        result = {
            "symbol": symbol,
            "sma_20": float(sma_20) if not pd.isna(sma_20) else None,
            "sma_50": float(sma_50) if not pd.isna(sma_50) else None,
            "rsi": float(rsi) if not pd.isna(rsi) else None,
            "bollinger_upper": float(upper_band) if not pd.isna(upper_band) else None,
            "bollinger_lower": float(lower_band) if not pd.isna(lower_band) else None,
            "macd": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
            "macd_signal": float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else None,
            "current_price": float(close_prices.iloc[-1]),
            "cached": False
        }
        
        app_context._update_cache_timestamp(cache_key)
        return result
        
    except Exception as e:
        await ctx.error(f"Error calculating technical indicators: {str(e)}")
        return {"error": str(e)}

# Sentiment Analysis Tools
@mcp.tool()
async def analyze_sentiment(symbol: str, ctx: Context, sources: List[str] = None) -> Dict[str, Any]:
    """
    Analyze sentiment for a symbol from various sources
    
    Args:
        symbol: Stock symbol
        sources: List of sources to analyze (news, social, analyst)
    """
    try:
        await ctx.info(f"Analyzing sentiment for {symbol}")
        
        if sources is None:
            sources = ["news", "social", "analyst"]
        
        # Simulate sentiment analysis (in real implementation, integrate with news APIs, social media, etc.)
        sentiment_scores = []
        
        for source in sources:
            # Simulate sentiment score between -1 and 1
            score = np.random.uniform(-0.5, 0.5)  # Placeholder
            sentiment_scores.append(score)
        
        overall_sentiment = np.mean(sentiment_scores)
        confidence = min(1.0, abs(overall_sentiment) + 0.3)
        
        sentiment_data = SentimentData(
            symbol=symbol,
            sentiment_score=overall_sentiment,
            confidence=confidence,
            sources=sources,
            timestamp=datetime.now()
        )
        
        # Cache the sentiment data
        app_context.sentiment_cache[symbol] = sentiment_data
        
        return {
            "symbol": symbol,
            "sentiment_score": sentiment_data.sentiment_score,
            "confidence": sentiment_data.confidence,
            "sources": sentiment_data.sources,
            "interpretation": "bullish" if overall_sentiment > 0.1 else "bearish" if overall_sentiment < -0.1 else "neutral",
            "timestamp": sentiment_data.timestamp.isoformat()
        }
        
    except Exception as e:
        await ctx.error(f"Error analyzing sentiment: {str(e)}")
        return {"error": str(e)}

# Risk Assessment Tools
@mcp.tool()
async def calculate_risk_metrics(symbol: str, ctx: Context, period: str = "1y") -> Dict[str, Any]:
    """
    Calculate risk metrics for a symbol
    
    Args:
        symbol: Stock symbol
        period: Time period for risk calculation
    """
    try:
        await ctx.info(f"Calculating risk metrics for {symbol}")
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for symbol {symbol}"}
        
        # Calculate returns
        returns = hist['Close'].pct_change().dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Beta (simplified - would need market data for proper calculation)
        beta = 1.0  # Placeholder
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Sharpe Ratio (simplified - would need risk-free rate)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        risk_metrics = RiskMetrics(
            symbol=symbol,
            volatility=volatility,
            beta=beta,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )
        
        # Cache the risk metrics
        app_context.risk_cache[symbol] = risk_metrics
        
        return {
            "symbol": symbol,
            "volatility": float(risk_metrics.volatility),
            "beta": float(risk_metrics.beta),
            "var_95": float(risk_metrics.var_95),
            "sharpe_ratio": float(risk_metrics.sharpe_ratio),
            "max_drawdown": float(risk_metrics.max_drawdown),
            "risk_level": "high" if volatility > 0.3 else "medium" if volatility > 0.15 else "low"
        }
        
    except Exception as e:
        await ctx.error(f"Error calculating risk metrics: {str(e)}")
        return {"error": str(e)}

# Alpha Discovery Tools
@mcp.tool()
async def discover_alpha_opportunities(
    ctx: Context,
    universe: List[str] = None,
    min_score: float = 0.6
) -> Dict[str, Any]:
    """
    Discover alpha opportunities across a universe of stocks
    
    Args:
        universe: List of symbols to analyze (default: popular stocks)
        min_score: Minimum alpha score threshold
    """
    try:
        await ctx.info("Discovering alpha opportunities")
        
        if universe is None:
            universe = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
        
        opportunities = []
        
        for symbol in universe:
            await ctx.info(f"Analyzing {symbol} for alpha opportunities")
            
            # Get market data
            market_result = await get_market_data(symbol, ctx, "1mo")
            if "error" in market_result:
                continue
                
            # Get technical indicators
            tech_result = await get_technical_indicators(symbol, ctx, "1mo")
            if "error" in tech_result:
                continue
                
            # Get sentiment
            sentiment_result = await analyze_sentiment(symbol, ctx)
            if "error" in sentiment_result:
                continue
                
            # Calculate alpha score (simplified algorithm)
            factors = {}
            
            # Technical factor
            if tech_result.get("rsi"):
                rsi = tech_result["rsi"]
                factors["technical"] = 1.0 if rsi < 30 else -1.0 if rsi > 70 else 0.0
            
            # Sentiment factor
            factors["sentiment"] = sentiment_result.get("sentiment_score", 0.0)
            
            # Momentum factor
            factors["momentum"] = market_result.get("change_percent", 0.0) / 100.0
            
            # Volume factor (simplified)
            factors["volume"] = 0.1  # Placeholder
            
            # Calculate overall alpha score
            alpha_score = (
                factors.get("technical", 0.0) * 0.3 +
                factors.get("sentiment", 0.0) * 0.3 +
                factors.get("momentum", 0.0) * 0.2 +
                factors.get("volume", 0.0) * 0.2
            )
            
            confidence = min(1.0, abs(alpha_score) + 0.2)
            
            if abs(alpha_score) >= min_score:
                alpha_data = AlphaDiscoveryData(
                    symbol=symbol,
                    alpha_score=alpha_score,
                    confidence=confidence,
                    factors=factors,
                    timestamp=datetime.now()
                )
                
                app_context.alpha_discoveries.append(alpha_data)
                
                opportunities.append({
                    "symbol": symbol,
                    "alpha_score": alpha_score,
                    "confidence": confidence,
                    "factors": factors,
                    "recommendation": "buy" if alpha_score > 0 else "sell",
                    "current_price": market_result.get("price"),
                    "timestamp": alpha_data.timestamp.isoformat()
                })
        
        # Sort by alpha score
        opportunities.sort(key=lambda x: abs(x["alpha_score"]), reverse=True)
        
        return {
            "opportunities": opportunities,
            "total_analyzed": len(universe),
            "opportunities_found": len(opportunities),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        await ctx.error(f"Error discovering alpha opportunities: {str(e)}")
        return {"error": str(e)}

# Trading Tools
@mcp.tool()
async def simulate_trade(
    symbol: str,
    action: str,
    quantity: int,
    ctx: Context,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Simulate a trade execution
    
    Args:
        symbol: Stock symbol
        action: "buy" or "sell"
        quantity: Number of shares
        price: Execution price (current market price if not specified)
    """
    try:
        await ctx.info(f"Simulating {action} trade for {symbol}")
        
        if action not in ["buy", "sell"]:
            return {"error": "Action must be 'buy' or 'sell'"}
        
        # Get current market price if not specified
        if price is None:
            market_result = await get_market_data(symbol, ctx, "1d")
            if "error" in market_result:
                return market_result
            price = market_result["price"]
        
        # Calculate trade value
        trade_value = price * quantity
        
        # Simulate execution
        execution_time = datetime.now()
        
        # Update positions
        if symbol not in app_context.active_positions:
            app_context.active_positions[symbol] = {
                "quantity": 0,
                "avg_price": 0.0,
                "total_value": 0.0
            }
        
        position = app_context.active_positions[symbol]
        
        if action == "buy":
            new_quantity = position["quantity"] + quantity
            new_total_value = position["total_value"] + trade_value
            new_avg_price = new_total_value / new_quantity if new_quantity > 0 else 0.0
            
            position["quantity"] = new_quantity
            position["avg_price"] = new_avg_price
            position["total_value"] = new_total_value
            
        else:  # sell
            if position["quantity"] < quantity:
                return {"error": f"Insufficient shares to sell. Available: {position['quantity']}"}
            
            new_quantity = position["quantity"] - quantity
            position["quantity"] = new_quantity
            
            if new_quantity == 0:
                position["avg_price"] = 0.0
                position["total_value"] = 0.0
            else:
                position["total_value"] = position["avg_price"] * new_quantity
        
        return {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "trade_value": trade_value,
            "execution_time": execution_time.isoformat(),
            "position_after": {
                "quantity": position["quantity"],
                "avg_price": position["avg_price"],
                "total_value": position["total_value"]
            },
            "status": "executed"
        }
        
    except Exception as e:
        await ctx.error(f"Error simulating trade: {str(e)}")
        return {"error": str(e)}

# Portfolio Management Tools
@mcp.tool()
async def get_portfolio_summary(ctx: Context) -> Dict[str, Any]:
    """Get current portfolio summary"""
    try:
        await ctx.info("Generating portfolio summary")
        
        portfolio_value = 0.0
        positions = []
        
        for symbol, position in app_context.active_positions.items():
            if position["quantity"] > 0:
                # Get current market price
                market_result = await get_market_data(symbol, ctx, "1d")
                if "error" not in market_result:
                    current_price = market_result["price"]
                    current_value = current_price * position["quantity"]
                    unrealized_pnl = current_value - position["total_value"]
                    unrealized_pnl_percent = (unrealized_pnl / position["total_value"]) * 100
                    
                    positions.append({
                        "symbol": symbol,
                        "quantity": position["quantity"],
                        "avg_price": position["avg_price"],
                        "current_price": current_price,
                        "current_value": current_value,
                        "cost_basis": position["total_value"],
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pnl_percent": unrealized_pnl_percent
                    })
                    
                    portfolio_value += current_value
        
        return {
            "total_value": portfolio_value,
            "positions": positions,
            "position_count": len(positions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        await ctx.error(f"Error generating portfolio summary: {str(e)}")
        return {"error": str(e)}

# Logging and Discovery Tools
@mcp.tool()
async def log_discovery(
    discovery_type: str,
    symbol: str,
    details: Dict[str, Any],
    ctx: Context
) -> Dict[str, Any]:
    """
    Log an alpha discovery for tracking and analysis
    
    Args:
        discovery_type: Type of discovery (alpha, risk, sentiment, etc.)
        symbol: Stock symbol
        details: Discovery details
    """
    try:
        await ctx.info(f"Logging {discovery_type} discovery for {symbol}")
        
        log_entry = {
            "type": discovery_type,
            "symbol": symbol,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "id": f"{discovery_type}_{symbol}_{int(datetime.now().timestamp())}"
        }
        
        # In a real implementation, this would be stored in a database
        # For now, we'll just return the log entry
        
        return {
            "status": "logged",
            "log_entry": log_entry
        }
        
    except Exception as e:
        await ctx.error(f"Error logging discovery: {str(e)}")
        return {"error": str(e)}

# Resources
@mcp.resource("market-data://{symbol}")
async def get_market_data_resource(symbol: str) -> str:
    """Get market data as a resource"""
    if symbol in app_context.market_data_cache:
        data = app_context.market_data_cache[symbol]
        return f"Market Data for {symbol}:\nPrice: ${data.price:.2f}\nVolume: {data.volume:,}\nChange: {data.change:+.2f} ({data.change_percent:+.2f}%)"
    return f"No cached market data for {symbol}"

@mcp.resource("portfolio://summary")
async def get_portfolio_resource() -> str:
    """Get portfolio summary as a resource"""
    if not app_context.active_positions:
        return "No active positions in portfolio"
    
    summary = "Portfolio Summary:\n"
    for symbol, position in app_context.active_positions.items():
        if position["quantity"] > 0:
            summary += f"{symbol}: {position['quantity']} shares @ ${position['avg_price']:.2f}\n"
    
    return summary

@mcp.resource("discoveries://recent")
async def get_recent_discoveries() -> str:
    """Get recent alpha discoveries as a resource"""
    if not app_context.alpha_discoveries:
        return "No recent alpha discoveries"
    
    # Get last 5 discoveries
    recent = app_context.alpha_discoveries[-5:]
    summary = "Recent Alpha Discoveries:\n"
    
    for discovery in recent:
        summary += f"{discovery.symbol}: Alpha Score {discovery.alpha_score:.3f} (Confidence: {discovery.confidence:.3f})\n"
    
    return summary

# Prompts
@mcp.prompt()
async def analyze_stock_prompt(symbol: str, analysis_type: str = "comprehensive") -> str:
    """Generate a prompt for stock analysis"""
    return f"""
    Please analyze {symbol} stock with a {analysis_type} approach.
    
    Consider the following aspects:
    1. Technical indicators and chart patterns
    2. Fundamental analysis (if applicable)
    3. Market sentiment and news
    4. Risk factors and volatility
    5. Alpha opportunities and trading signals
    
    Provide actionable insights and recommendations.
    """

@mcp.prompt()
async def portfolio_review_prompt() -> List[base.Message]:
    """Generate a prompt for portfolio review"""
    return [
        base.UserMessage("I need to review my current portfolio performance and positions."),
        base.AssistantMessage("I'll help you analyze your portfolio. Let me gather the current data and provide insights on performance, risk, and potential optimizations."),
        base.UserMessage("Please focus on risk-adjusted returns and suggest any rebalancing opportunities.")
    ]

@mcp.prompt()
async def alpha_discovery_prompt(market_condition: str = "normal") -> str:
    """Generate a prompt for alpha discovery"""
    return f"""
    Discover alpha opportunities in the current {market_condition} market conditions.
    
    Please:
    1. Scan for undervalued or overvalued securities
    2. Identify momentum and mean reversion opportunities
    3. Analyze sector rotation patterns
    4. Consider risk-adjusted returns
    5. Provide specific trading recommendations with entry/exit points
    
    Focus on high-probability setups with favorable risk-reward ratios.
    """

# Additional Enhanced Tools
@mcp.tool()
async def get_market_overview(ctx: Context, symbols: List[str] = None) -> Dict[str, Any]:
    """
    Get an overview of multiple symbols or market indices
    
    Args:
        symbols: List of symbols to analyze (default: major indices)
    """
    try:
        if symbols is None:
            symbols = ["^GSPC", "^DJI", "^IXIC", "^VIX"]  # S&P 500, Dow, NASDAQ, VIX
        
        await ctx.info(f"Getting market overview for {len(symbols)} symbols")
        
        overview = {
            "timestamp": datetime.now().isoformat(),
            "symbols": {},
            "summary": {
                "advancing": 0,
                "declining": 0,
                "unchanged": 0,
                "total_volume": 0
            }
        }
        
        for symbol in symbols:
            try:
                data = await get_market_data(symbol, ctx, "1d")
                if "error" not in data:
                    overview["symbols"][symbol] = data
                    
                    # Update summary
                    if data["change"] > 0:
                        overview["summary"]["advancing"] += 1
                    elif data["change"] < 0:
                        overview["summary"]["declining"] += 1
                    else:
                        overview["summary"]["unchanged"] += 1
                        
                    overview["summary"]["total_volume"] += data["volume"]
                    
            except Exception as e:
                await ctx.warning(f"Error fetching data for {symbol}: {str(e)}")
                overview["symbols"][symbol] = {"error": str(e)}
        
        return overview
        
    except Exception as e:
        await ctx.error(f"Error getting market overview: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def validate_symbol(symbol: str, ctx: Context) -> Dict[str, Any]:
    """
    Validate if a symbol exists and is tradeable
    
    Args:
        symbol: Stock symbol to validate
    """
    try:
        await ctx.info(f"Validating symbol: {symbol}")
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'regularMarketPrice' not in info:
            return {
                "symbol": symbol,
                "valid": False,
                "error": "Symbol not found or not tradeable"
            }
        
        return {
            "symbol": symbol,
            "valid": True,
            "name": info.get('longName', 'Unknown'),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": info.get('marketCap', 0),
            "volume": info.get('volume', 0),
            "price": info.get('regularMarketPrice', 0)
        }
        
    except Exception as e:
        await ctx.error(f"Error validating symbol: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_system_health(ctx: Context) -> Dict[str, Any]:
    """Get system health and performance metrics"""
    try:
        await ctx.info("Checking system health")
        
        # Check cache health
        cache_stats = {
            "market_data_cache_size": len(app_context.market_data_cache),
            "sentiment_cache_size": len(app_context.sentiment_cache),
            "risk_cache_size": len(app_context.risk_cache),
            "cache_entries": len(app_context.cache_timestamps)
        }
        
        # Check portfolio health
        portfolio_stats = {
            "active_positions": len([p for p in app_context.active_positions.values() if p["quantity"] > 0]),
            "total_positions": len(app_context.active_positions)
        }
        
        # Check discovery health
        discovery_stats = {
            "total_discoveries": len(app_context.alpha_discoveries),
            "recent_discoveries": len([d for d in app_context.alpha_discoveries 
                                     if (datetime.now() - d.timestamp).days <= 7])
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "cache": cache_stats,
            "portfolio": portfolio_stats,
            "discoveries": discovery_stats,
            "uptime": (datetime.now() - app_context.start_time).total_seconds() if hasattr(app_context, 'start_time') else 0
        }
        
    except Exception as e:
        await ctx.error(f"Error checking system health: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def clear_cache(ctx: Context, cache_type: str = "all") -> Dict[str, Any]:
    """
    Clear cache to free up memory
    
    Args:
        cache_type: Type of cache to clear (market_data, sentiment, risk, all)
    """
    try:
        await ctx.info(f"Clearing cache: {cache_type}")
        
        cleared = []
        
        if cache_type in ["market_data", "all"]:
            app_context.market_data_cache.clear()
            cleared.append("market_data")
            
        if cache_type in ["sentiment", "all"]:
            app_context.sentiment_cache.clear()
            cleared.append("sentiment")
            
        if cache_type in ["risk", "all"]:
            app_context.risk_cache.clear()
            cleared.append("risk")
            
        if cache_type == "all":
            app_context.cache_timestamps.clear()
            cleared.append("timestamps")
        
        return {
            "status": "success",
            "cleared_caches": cleared,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        await ctx.error(f"Error clearing cache: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_trading_signals(ctx: Context, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
    """
    Generate trading signals based on technical analysis
    
    Args:
        symbol: Stock symbol
        timeframe: Analysis timeframe
    """
    try:
        await ctx.info(f"Generating trading signals for {symbol}")
        
        # Get technical indicators
        indicators = await get_technical_indicators(symbol, ctx, timeframe)
        if "error" in indicators:
            return indicators
            
        # Get market data
        market_data = await get_market_data(symbol, ctx, timeframe)
        if "error" in market_data:
            return market_data
            
        signals = []
        current_price = market_data["price"]
        
        # RSI signals
        if indicators["rsi"] is not None:
            if indicators["rsi"] < 30:
                signals.append({
                    "type": "RSI_OVERSOLD",
                    "strength": "strong",
                    "action": "BUY",
                    "reasoning": f"RSI at {indicators['rsi']:.2f} indicates oversold conditions"
                })
            elif indicators["rsi"] > 70:
                signals.append({
                    "type": "RSI_OVERBOUGHT",
                    "strength": "strong",
                    "action": "SELL",
                    "reasoning": f"RSI at {indicators['rsi']:.2f} indicates overbought conditions"
                })
        
        # Moving average signals
        if indicators["sma_20"] is not None and indicators["sma_50"] is not None:
            if current_price > indicators["sma_20"] > indicators["sma_50"]:
                signals.append({
                    "type": "GOLDEN_CROSS",
                    "strength": "medium",
                    "action": "BUY",
                    "reasoning": "Price above 20-day SMA above 50-day SMA (bullish alignment)"
                })
            elif current_price < indicators["sma_20"] < indicators["sma_50"]:
                signals.append({
                    "type": "DEATH_CROSS",
                    "strength": "medium",
                    "action": "SELL",
                    "reasoning": "Price below 20-day SMA below 50-day SMA (bearish alignment)"
                })
        
        # Bollinger Bands signals
        if indicators["bollinger_upper"] is not None and indicators["bollinger_lower"] is not None:
            if current_price <= indicators["bollinger_lower"]:
                signals.append({
                    "type": "BOLLINGER_OVERSOLD",
                    "strength": "medium",
                    "action": "BUY",
                    "reasoning": "Price at or below lower Bollinger Band"
                })
            elif current_price >= indicators["bollinger_upper"]:
                signals.append({
                    "type": "BOLLINGER_OVERBOUGHT",
                    "strength": "medium",
                    "action": "SELL",
                    "reasoning": "Price at or above upper Bollinger Band"
                })
        
        # MACD signals
        if indicators["macd"] is not None and indicators["macd_signal"] is not None:
            if indicators["macd"] > indicators["macd_signal"]:
                signals.append({
                    "type": "MACD_BULLISH",
                    "strength": "weak",
                    "action": "BUY",
                    "reasoning": "MACD above signal line (bullish momentum)"
                })
            else:
                signals.append({
                    "type": "MACD_BEARISH",
                    "strength": "weak",
                    "action": "SELL",
                    "reasoning": "MACD below signal line (bearish momentum)"
                })
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "signals": signals,
            "signal_count": len(signals),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        await ctx.error(f"Error generating trading signals: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Run the MCP server
    mcp.run() 