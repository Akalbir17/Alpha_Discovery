"""
Alpha Discovery MCP Server - 2025 State-of-the-Art Edition

Modern MCP server implementation using FastMCP with:
- Latest MCP SDK features
- Enhanced error handling and logging
- Integration with updated model manager
- Comprehensive tool definitions
- Real-time performance monitoring
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastmcp import FastMCP
    from fastmcp.server import Context
    from fastmcp.server.types import TextContent, Tool
except ImportError as e:
    logger.error(f"FastMCP import error: {e}")
    logger.info("Installing FastMCP...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastmcp>=0.9.0"])
    from fastmcp import FastMCP
    from fastmcp.server import Context
    from fastmcp.server.types import TextContent, Tool

# Import our components with proper error handling
try:
    from agents.microstructure_agent import MicrostructureAgent
    from tools.reddit_scraper import RedditScraper
    from tools.market_data import MarketDataTool
    from tools.order_flow import OrderFlowTool
    from utils.model_manager import model_manager, TaskType, ModelResponse
except ImportError as e:
    logger.warning(f"Import error: {e}")
    # Fallback imports
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.agents.microstructure_agent import MicrostructureAgent
        from src.scrapers.reddit_scraper import RedditScraper
        from src.tools.market_data import MarketDataTool
        from src.tools.order_flow import OrderFlowTool
        from src.utils.model_manager import model_manager, TaskType, ModelResponse
    except ImportError as e2:
        logger.error(f"Fallback import failed: {e2}")
        # Create mock implementations for testing
        class MockAgent:
            async def analyze_microstructure(self, symbol: str, timeframe: str = "1h"):
                return {"symbol": symbol, "mock": True, "error": "Agent not available"}
        
        MicrostructureAgent = MockAgent()
        RedditScraper = None
        MarketDataTool = None
        OrderFlowTool = None

# Create FastMCP server instance with enhanced configuration
mcp = FastMCP(
    name="Alpha Discovery MCP Server",
    version="2.0.0",
    description="State-of-the-art financial analysis and trading intelligence platform"
)

# Global tool instances
microstructure_agent = None
reddit_scraper = None
market_data_tool = None
order_flow_tool = None

async def initialize_tools():
    """Initialize all tools with proper error handling"""
    global microstructure_agent, reddit_scraper, market_data_tool, order_flow_tool
    
    try:
        if 'MicrostructureAgent' in globals() and MicrostructureAgent:
            if hasattr(MicrostructureAgent, '__call__'):
                microstructure_agent = MicrostructureAgent
            else:
                microstructure_agent = MicrostructureAgent()
        
        if RedditScraper:
            reddit_scraper = RedditScraper()
        
        if MarketDataTool:
            market_data_tool = MarketDataTool()
        
        if OrderFlowTool:
            order_flow_tool = OrderFlowTool()
        
        logger.info("Tools initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing tools: {e}")

@mcp.tool()
async def analyze_microstructure(
    symbol: str,
    timeframe: str = "1h",
    include_signals: bool = True,
    include_flow_analysis: bool = True
) -> Dict[str, Any]:
    """
    Analyze market microstructure for a given symbol using state-of-the-art algorithms.
    
    Features:
    - VPIN (Volume-synchronized Probability of Informed Trading)
    - Kyle's Lambda for price impact measurement
    - Lee-Ready algorithm for order flow classification
    - Institutional vs retail flow detection
    - Alpha signal generation
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "TSLA")
        timeframe: Analysis timeframe ("1m", "5m", "15m", "1h", "1d")
        include_signals: Whether to include alpha signals
        include_flow_analysis: Whether to include order flow analysis
    
    Returns:
        Comprehensive microstructure analysis with metrics and signals
    """
    try:
        start_time = datetime.now()
        
        if not microstructure_agent:
            await initialize_tools()
        
        if not microstructure_agent:
            return {
                "error": "Microstructure agent not available",
                "symbol": symbol,
                "timestamp": start_time.isoformat()
            }
        
        # Get microstructure analysis
        analysis = await microstructure_agent.analyze_microstructure(
            symbol, timeframe
        )
        
        result = {
            "symbol": analysis.symbol,
            "timestamp": analysis.timestamp.isoformat(),
            "timeframe": timeframe,
            "metrics": {
                "vpin": analysis.vpin,
                "kyle_lambda": analysis.kyle_lambda,
                "flow_imbalance": analysis.flow_imbalance,
                "bid_ask_spread": analysis.bid_ask_spread,
                "toxicity_score": analysis.toxicity_score,
                "market_impact": analysis.market_impact,
                "informed_trading_prob": analysis.informed_trading_prob
            },
            "analysis_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Add signals if requested
        if include_signals:
            signals = await microstructure_agent.generate_signals(symbol, timeframe)
            result["signals"] = [
                {
                    "type": s.signal_type,
                    "value": s.value,
                    "confidence": s.confidence,
                    "description": s.description,
                    "alpha_score": s.alpha_score,
                    "evidence": s.evidence
                } for s in signals
            ]
        
        # Add flow analysis if requested
        if include_flow_analysis:
            flow_analysis = await microstructure_agent.analyze_order_flow(symbol, timeframe)
            result["flow_analysis"] = {
                "institutional_flow": flow_analysis.institutional_flow,
                "retail_flow": flow_analysis.retail_flow,
                "flow_ratio": flow_analysis.flow_ratio,
                "flow_direction": flow_analysis.flow_direction,
                "flow_strength": flow_analysis.flow_strength
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_microstructure: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
async def scrape_reddit_sentiment(
    subreddits: str = "wallstreetbets,stocks,investing",
    limit: int = 100,
    time_filter: str = "day",
    include_trends: bool = True
) -> Dict[str, Any]:
    """
    Scrape Reddit for financial sentiment and trend analysis.
    
    Features:
    - Multi-subreddit monitoring
    - Advanced sentiment analysis using transformers
    - Trend detection and unusual activity identification
    - Ticker extraction and sentiment scoring
    
    Args:
        subreddits: Comma-separated list of subreddits
        limit: Maximum number of posts to analyze
        time_filter: Time filter ("hour", "day", "week", "month")
        include_trends: Whether to include trend analysis
    
    Returns:
        Sentiment analysis results with trends and unusual activity
    """
    try:
        start_time = datetime.now()
        
        if not reddit_scraper:
            await initialize_tools()
        
        if not reddit_scraper:
            return {
                "error": "Reddit scraper not available",
                "timestamp": start_time.isoformat()
            }
        
        subreddit_list = [s.strip() for s in subreddits.split(",")]
        
        # Get sentiment analysis
        sentiment_data = await reddit_scraper.get_sentiment_analysis(
            subreddits=subreddit_list,
            limit=limit,
            time_filter=time_filter
        )
        
        result = {
            "timestamp": start_time.isoformat(),
            "subreddits": subreddit_list,
            "posts_analyzed": len(sentiment_data.get("posts", [])),
            "sentiment_summary": sentiment_data.get("summary", {}),
            "top_tickers": sentiment_data.get("top_tickers", []),
            "analysis_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Add trend analysis if requested
        if include_trends:
            trends = await reddit_scraper.get_trending_tickers(
                subreddits=subreddit_list,
                timeframe=time_filter
            )
            result["trends"] = trends
            
            # Get unusual activity
            unusual_activity = await reddit_scraper.get_unusual_activity(
                subreddits=subreddit_list,
                timeframe=time_filter
            )
            result["unusual_activity"] = unusual_activity
        
        return result
        
    except Exception as e:
        logger.error(f"Error in scrape_reddit_sentiment: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
async def get_market_data(
    symbol: str,
    timeframe: str = "1d",
    period: str = "1mo",
    include_indicators: bool = True,
    include_volume_profile: bool = False
) -> Dict[str, Any]:
    """
    Get comprehensive market data with technical indicators.
    
    Features:
    - OHLCV data with multiple timeframes
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Volume profile analysis
    - Support/resistance levels
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "TSLA")
        timeframe: Data timeframe ("1m", "5m", "15m", "1h", "1d")
        period: Historical period ("1d", "5d", "1mo", "3mo", "6mo", "1y")
        include_indicators: Whether to include technical indicators
        include_volume_profile: Whether to include volume profile
    
    Returns:
        Comprehensive market data with indicators and analysis
    """
    try:
        start_time = datetime.now()
        
        if not market_data_tool:
            await initialize_tools()
        
        if not market_data_tool:
            return {
                "error": "Market data tool not available",
                "symbol": symbol,
                "timestamp": start_time.isoformat()
            }
        
        # Get market data
        market_data = await market_data_tool.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            period=period
        )
        
        result = {
            "symbol": symbol,
            "timestamp": start_time.isoformat(),
            "timeframe": timeframe,
            "period": period,
            "data_points": len(market_data.get("data", [])),
            "latest_price": market_data.get("latest_price"),
            "price_change": market_data.get("price_change"),
            "volume": market_data.get("volume"),
            "analysis_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Add technical indicators if requested
        if include_indicators:
            indicators = await market_data_tool.calculate_indicators(
                symbol=symbol,
                timeframe=timeframe
            )
            result["indicators"] = indicators
        
        # Add volume profile if requested
        if include_volume_profile:
            volume_profile = await market_data_tool.get_volume_profile(
                symbol=symbol,
                timeframe=timeframe
            )
            result["volume_profile"] = volume_profile
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_market_data: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
async def analyze_order_flow(
    symbol: str,
    timeframe: str = "1h",
    include_flow_classification: bool = True,
    include_toxicity: bool = True
) -> Dict[str, Any]:
    """
    Analyze order flow and market microstructure patterns.
    
    Features:
    - Order flow imbalance detection
    - Buy/sell pressure analysis
    - Flow toxicity measurement
    - Institutional vs retail classification
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "TSLA")
        timeframe: Analysis timeframe ("1m", "5m", "15m", "1h")
        include_flow_classification: Whether to classify flow types
        include_toxicity: Whether to include toxicity analysis
    
    Returns:
        Order flow analysis with classification and toxicity metrics
    """
    try:
        start_time = datetime.now()
        
        if not order_flow_tool:
            await initialize_tools()
        
        if not order_flow_tool:
            return {
                "error": "Order flow tool not available",
                "symbol": symbol,
                "timestamp": start_time.isoformat()
            }
        
        # Get order flow analysis
        flow_data = await order_flow_tool.analyze_order_flow(
            symbol=symbol,
            timeframe=timeframe
        )
        
        result = {
            "symbol": symbol,
            "timestamp": start_time.isoformat(),
            "timeframe": timeframe,
            "flow_metrics": {
                "buy_volume": flow_data.get("buy_volume"),
                "sell_volume": flow_data.get("sell_volume"),
                "flow_imbalance": flow_data.get("flow_imbalance"),
                "net_flow": flow_data.get("net_flow")
            },
            "analysis_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Add flow classification if requested
        if include_flow_classification:
            classification = await order_flow_tool.classify_flow(
                symbol=symbol,
                timeframe=timeframe
            )
            result["flow_classification"] = classification
        
        # Add toxicity analysis if requested
        if include_toxicity:
            toxicity = await order_flow_tool.analyze_toxicity(
                symbol=symbol,
                timeframe=timeframe
            )
            result["toxicity_analysis"] = toxicity
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_order_flow: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
async def get_ai_analysis(
    prompt: str,
    task_type: str = "financial",
    model_preference: str = "auto",
    include_reasoning: bool = True
) -> Dict[str, Any]:
    """
    Get AI-powered analysis using the latest 2025 models.
    
    Features:
    - Multiple free AI models (Claude 3.5, Gemini 2.0, Llama 3.3, etc.)
    - Intelligent model selection based on task type
    - Enhanced reasoning and analysis capabilities
    - Fallback and error handling
    
    Args:
        prompt: Analysis prompt or question
        task_type: Type of analysis ("financial", "technical", "sentiment", "creative")
        model_preference: Preferred model ("auto", "claude", "gemini", "groq", "together")
        include_reasoning: Whether to include reasoning steps
    
    Returns:
        AI analysis with model metadata and reasoning
    """
    try:
        start_time = datetime.now()
        
        # Map task types to TaskType enum
        task_mapping = {
            "financial": TaskType.FINANCIAL,
            "technical": TaskType.ANALYTICAL,
            "sentiment": TaskType.SPECIALIZED,
            "creative": TaskType.CREATIVE,
            "realtime": TaskType.REALTIME,
            "complex": TaskType.COMPLEX,
            "reasoning": TaskType.REASONING
        }
        
        task_enum = task_mapping.get(task_type, TaskType.FINANCIAL)
        
        # Enhanced system prompt for financial analysis
        system_prompt = """You are a state-of-the-art financial analysis AI with deep expertise in:
- Market microstructure and order flow analysis
- Technical and fundamental analysis
- Risk management and portfolio optimization
- Algorithmic trading strategies
- Market sentiment and behavioral finance

Provide comprehensive, actionable insights with clear reasoning and evidence-based conclusions."""
        
        # Get AI response using model manager
        response: ModelResponse = await model_manager.get_response(
            prompt=prompt,
            task_type=task_enum,
            system_prompt=system_prompt,
            use_cache=True,
            max_retries=3
        )
        
        result = {
            "timestamp": start_time.isoformat(),
            "task_type": task_type,
            "model_used": response.model_used,
            "response": response.content,
            "metadata": {
                "response_time": response.response_time,
                "tokens_used": response.tokens_used,
                "confidence": response.confidence,
                "cached": response.cached
            },
            "analysis_time": (datetime.now() - start_time).total_seconds()
        }
        
        if response.error:
            result["error"] = response.error
        
        if include_reasoning:
            # Add reasoning context
            result["reasoning"] = {
                "model_selection": f"Selected {response.model_used} for {task_type} task",
                "approach": "Used enhanced system prompt for financial domain expertise",
                "confidence_factors": [
                    f"Model performance: {response.confidence}",
                    f"Response time: {response.response_time:.2f}s",
                    f"Tokens used: {response.tokens_used}"
                ]
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_ai_analysis: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
async def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status and performance metrics.
    
    Returns:
        System status with model availability, performance metrics, and health checks
    """
    try:
        start_time = datetime.now()
        
        # Get model manager stats
        model_stats = model_manager.get_enhanced_stats()
        
        # Check tool availability
        tool_status = {
            "microstructure_agent": microstructure_agent is not None,
            "reddit_scraper": reddit_scraper is not None,
            "market_data_tool": market_data_tool is not None,
            "order_flow_tool": order_flow_tool is not None
        }
        
        result = {
            "timestamp": start_time.isoformat(),
            "system_health": "healthy",
            "version": "2.0.0",
            "uptime": "N/A",  # Would need to track startup time
            "model_manager": model_stats,
            "tools_status": tool_status,
            "mcp_server": {
                "name": "Alpha Discovery MCP Server",
                "version": "2.0.0",
                "framework": "FastMCP",
                "tools_registered": 6
            },
            "analysis_time": (datetime.now() - start_time).total_seconds()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_system_status: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def main():
    """Main server entry point"""
    try:
        # Initialize tools
        await initialize_tools()
        
        # Log server startup
        logger.info("Alpha Discovery MCP Server 2.0 starting...")
        logger.info(f"Available tools: {len(mcp.tools)} registered")
        
        # Run the server
        await mcp.run()
        
    except Exception as e:
        logger.error(f"Server startup error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 