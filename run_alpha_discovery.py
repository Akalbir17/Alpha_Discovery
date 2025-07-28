#!/usr/bin/env python3
"""
Alpha Discovery Main Execution Script

This is the main orchestration script for the Alpha Discovery platform.
It initializes all agents using CrewAI, starts the MCP server for tool access,
begins market data streaming, starts Reddit monitoring, and runs the continuous
alpha discovery loop with comprehensive logging and performance reporting.
"""

import os
import sys
import asyncio
import signal
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Configuration and utilities
from src.utils.config_manager import get_config_manager, get_config_value, get_config_section
from src.utils.sentry_config import init_sentry, set_tag, set_context, capture_exception, capture_message

# Import our existing specialized agents
from src.agents.microstructure_agent import MicrostructureAgent
from src.agents.altdata_agent import AlternativeDataAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.regime_agent import RegimeDetectionAgent

from src.data.market_data_manager import MarketDataManager
from src.data.reddit_monitor import RedditMonitor
from src.execution.trading_engine import TradingEngine
from src.risk.risk_manager import RiskManager
from src.portfolio.portfolio_manager import PortfolioManager
from src.utils.database import DatabaseManager
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsCollector
from src.utils.performance import PerformanceTracker
from src.mcp.alpha_discovery_server import mcp as alpha_mcp_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alpha-discovery.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AlphaDiscoveryState:
    """State management for Alpha Discovery platform."""
    is_running: bool = False
    start_time: datetime = field(default_factory=datetime.now)
    agents_initialized: bool = False
    mcp_server_running: bool = False
    market_data_streaming: bool = False
    reddit_monitoring: bool = False
    discovery_loop_running: bool = False
    shutdown_requested: bool = False
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)

class AlphaDiscoveryMCPServer:
    """MCP Server for Alpha Discovery tools and resources"""
    
    def __init__(self):
        self.server = alpha_mcp_server  # Use the FastMCP server directly
        self.is_running = False
        
    async def start(self):
        """Start the MCP server"""
        try:
            logger.info("Starting Alpha Discovery MCP server...")
            self.is_running = True
            
            # The FastMCP server will handle its own lifecycle
            # We just need to ensure it's available for the orchestrator
            logger.info("Alpha Discovery MCP server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {str(e)}")
            raise
            
    async def stop(self):
        """Stop the MCP server"""
        try:
            logger.info("Stopping Alpha Discovery MCP server...")
            self.is_running = False
            logger.info("Alpha Discovery MCP server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MCP server: {str(e)}")
            
    def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools"""
        return [
            "get_market_data",
            "get_technical_indicators", 
            "analyze_sentiment",
            "calculate_risk_metrics",
            "discover_alpha_opportunities",
            "simulate_trade",
            "get_portfolio_summary",
            "log_discovery"
        ]
        
    def get_available_resources(self) -> List[str]:
        """Get list of available MCP resources"""
        return [
            "market-data://{symbol}",
            "portfolio://summary", 
            "discoveries://recent"
        ]
        
    def get_available_prompts(self) -> List[str]:
        """Get list of available MCP prompts"""
        return [
            "analyze_stock_prompt",
            "portfolio_review_prompt",
            "alpha_discovery_prompt"
        ]

class AlphaDiscoveryOrchestrator:
    """Main orchestrator for the Alpha Discovery platform."""
    
    def __init__(self):
        self.state = AlphaDiscoveryState()
        self.config_loader = get_config_manager()
        self.config = get_config_section('database')
        
        # Initialize core components
        self.db_manager = DatabaseManager(self.config)
        self.market_data_manager = MarketDataManager(self.config)
        self.reddit_monitor = RedditMonitor(self.config)
        self.trading_engine = TradingEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        self.metrics_collector = MetricsCollector(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        
        # Initialize MCP server
        self.mcp_server = AlphaDiscoveryMCPServer()
        
        # Initialize our existing specialized agents
        self.microstructure_agent = MicrostructureAgent()
        self.altdata_agent = AlternativeDataAgent()
        self.strategy_agent = StrategyAgent()
        self.regime_agent = RegimeDetectionAgent()
        
        # Initialize CrewAI agents (wrappers)
        self.agents = {}
        self.crew = None
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.event_loop = None
        
        # Shutdown handling
        self.shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Performance monitoring
        self.last_performance_report = datetime.now()
        self.performance_report_interval = timedelta(minutes=15)
        
        logger.info("Alpha Discovery Orchestrator initialized")
    
    def _initialize_llm(self) -> Any:
        """Initialize the language model for agents."""
        try:
            # Try to use Claude first (as per user preference for latest models)
            if os.getenv('ANTHROPIC_API_KEY'):
                return ChatAnthropic(
                    model_name="claude-3-sonnet-20240229",
                    temperature=0.1,
                    max_tokens=4000
                )
            elif os.getenv('OPENAI_API_KEY'):
                return ChatOpenAI(
                    model_name="gpt-4-turbo-preview",
                    temperature=0.1,
                    max_tokens=4000
                )
            else:
                logger.warning("No API keys found, using mock LLM")
                return None
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return None
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.state.shutdown_requested = True
        self.shutdown_event.set()
    
    async def initialize_agents(self):
        """Initialize all trading agents with CrewAI."""
        logger.info("Initializing trading agents...")
        
        try:
            # Initialize individual agents
            self.agents['microstructure'] = Agent(
                role='Microstructure Analyst',
                goal='Analyze market microstructure and order flow to identify trading opportunities',
                backstory='''You are an expert in market microstructure and order flow analysis. 
                           You can identify patterns in order book data, identify liquidity 
                           bottlenecks, and predict price movements based on order flow.''',
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                tools=[]  # Tools will be provided via MCP
            )
            
            self.agents['altdata'] = Agent(
                role='Alternative Data Analyst',
                goal='Analyze alternative data sources (news, social media, etc.) to identify trading opportunities',
                backstory='''You are an expert in alternative data analysis, including news, 
                           social media sentiment, and market sentiment. You can identify 
                           patterns in non-traditional data sources that might not be 
                           immediately apparent in traditional market data.''',
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                tools=[]
            )
            
            self.agents['strategy'] = Agent(
                role='Strategy Analyst',
                goal='Develop and refine trading strategies based on market conditions and historical performance',
                backstory='''You are a seasoned strategy analyst with expertise in developing 
                           and optimizing trading strategies. You can analyze market 
                           conditions, identify potential opportunities, and refine 
                           strategies based on performance.''',
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                tools=[]
            )
            
            self.agents['regime'] = Agent(
                role='Regime Detection Analyst',
                goal='Identify and analyze market regimes (bull, bear, sideways) to inform strategy selection',
                backstory='''You are an expert in regime detection and market analysis. 
                           You can identify market regimes, understand their characteristics, 
                           and predict future market movements based on regime.''',
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                tools=[]
            )
            
            # Create crew with agents
            self.crew = Crew(
                agents=list(self.agents.values()),
                tasks=[],  # Tasks will be created dynamically
                verbose=2,
                process=Process.hierarchical,
                manager_llm=self.llm
            )
            
            self.state.agents_initialized = True
            logger.info("All trading agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    async def start_mcp_server(self):
        """Start the MCP server for tool access."""
        logger.info("Starting MCP server...")
        
        try:
            await self.mcp_server.start()
            self.state.mcp_server_running = True
            logger.info("MCP server started successfully")
            
        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            raise
    
    async def start_market_data_streaming(self):
        """Start market data streaming."""
        logger.info("Starting market data streaming...")
        
        try:
            # Get symbols to monitor from configuration
            symbols = self._get_monitored_symbols()
            
            # Start market data streaming
            await self.market_data_manager.start_streaming(symbols)
            self.state.market_data_streaming = True
            
            logger.info(f"Market data streaming started for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error starting market data streaming: {e}")
            raise
    
    async def start_reddit_monitoring(self):
        """Start Reddit monitoring."""
        logger.info("Starting Reddit monitoring...")
        
        try:
            # Get Reddit configuration
            reddit_config = get_config_section('monitoring')
            symbols = self._get_monitored_symbols()
            
            # Start Reddit monitoring
            await self.reddit_monitor.start_monitoring(symbols)
            self.state.reddit_monitoring = True
            
            logger.info("Reddit monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Reddit monitoring: {e}")
            raise
    
    async def run_discovery_loop(self):
        """Run the continuous alpha discovery loop."""
        logger.info("Starting alpha discovery loop...")
        
        self.state.discovery_loop_running = True
        discovery_count = 0
        
        try:
            while not self.state.shutdown_requested:
                loop_start = time.time()
                
                try:
                    # Update heartbeat
                    self.state.last_heartbeat = datetime.now()
                    
                    # Run discovery cycle
                    discoveries = await self._run_discovery_cycle()
                    
                    # Process discoveries
                    for discovery in discoveries:
                        await self._process_discovery(discovery)
                        discovery_count += 1
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    # Generate performance report if needed
                    if datetime.now() - self.last_performance_report > self.performance_report_interval:
                        await self._generate_performance_report()
                        self.last_performance_report = datetime.now()
                    
                    # Calculate loop duration
                    loop_duration = time.time() - loop_start
                    
                    # Log loop statistics
                    if discovery_count % 10 == 0:  # Log every 10 cycles
                        logger.info(f"Discovery loop cycle {discovery_count} completed in {loop_duration:.2f}s, "
                                  f"found {len(discoveries)} discoveries")
                    
                    # Sleep before next cycle (configurable)
                    await asyncio.sleep(self.config.get('discovery_loop_interval', 60))
                    
                except Exception as e:
                    self.state.error_count += 1
                    logger.error(f"Error in discovery loop cycle: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Exponential backoff on errors
                    error_sleep = min(60, 2 ** min(self.state.error_count, 6))
                    await asyncio.sleep(error_sleep)
        
        except Exception as e:
            logger.error(f"Fatal error in discovery loop: {e}")
            raise
        
        finally:
            self.state.discovery_loop_running = False
            logger.info("Alpha discovery loop stopped")
    
    async def _run_discovery_cycle(self) -> List[Dict[str, Any]]:
        """Run a single discovery cycle."""
        discoveries = []
        
        # Get symbols to analyze
        symbols = self._get_active_symbols()
        
        # Create tasks for each agent
        tasks = []
        
        for symbol in symbols:
            # Microstructure analysis task
            micro_task = Task(
                description=f"Analyze {symbol} using market microstructure and order flow. "
                          f"Identify potential trading opportunities with entry/exit levels.",
                agent=self.agents['microstructure'],
                expected_output="Microstructure analysis report with trading signals"
            )
            tasks.append(micro_task)
            
            # Alternative data analysis task
            altdata_task = Task(
                description=f"Analyze {symbol} using alternative data sources (news, social media, etc.) "
                          f"to identify sentiment-driven opportunities.",
                agent=self.agents['altdata'],
                expected_output="Alternative data analysis report with market psychology insights"
            )
            tasks.append(altdata_task)
            
            # Strategy analysis task
            strategy_task = Task(
                description=f"Analyze {symbol} using trading strategies to identify potential opportunities "
                          f"and refine strategy performance.",
                agent=self.agents['strategy'],
                expected_output="Strategy analysis report with trading opportunities"
            )
            tasks.append(strategy_task)
            
            # Regime detection task
            regime_task = Task(
                description=f"Analyze {symbol} to identify market regime (bull, bear, sideways) and "
                          f"predict future market movements.",
                agent=self.agents['regime'],
                expected_output="Regime detection report with market prediction"
            )
            tasks.append(regime_task)
        
        # Execute tasks using CrewAI
        try:
            # Update crew with new tasks
            self.crew.tasks = tasks
            
            # Execute crew tasks
            results = self.crew.kickoff()
            
            # Process results into discoveries
            for result in results:
                discovery = self._parse_agent_result(result)
                if discovery:
                    discoveries.append(discovery)
            
        except Exception as e:
            logger.error(f"Error executing crew tasks: {e}")
            capture_exception(e, extra={
                "component": "discovery_cycle",
                "error_type": "crew_execution_error",
                "symbols_count": len(symbols),
                "tasks_count": len(tasks)
            })
        
        return discoveries
    
    def _parse_agent_result(self, result: Any) -> Optional[Dict[str, Any]]:
        """Parse agent result into discovery format."""
        try:
            # Extract relevant information from agent result
            # This would be customized based on the actual CrewAI result format
            
            discovery = {
                'timestamp': datetime.now().isoformat(),
                'agent_type': result.get('agent_role', 'unknown'),
                'symbol': result.get('symbol', 'unknown'),
                'discovery_type': result.get('discovery_type', 'signal'),
                'confidence': result.get('confidence', 0.5),
                'reasoning': result.get('reasoning', ''),
                'metadata': result.get('metadata', {}),
                'raw_result': str(result)
            }
            
            return discovery
            
        except Exception as e:
            logger.error(f"Error parsing agent result: {e}")
            capture_exception(e, extra={
                "component": "agent_parsing",
                "error_type": "result_parsing_error",
                "raw_result": str(result)[:500]  # Truncate to avoid large payloads
            })
            return None
    
    async def _process_discovery(self, discovery: Dict[str, Any]):
        """Process a single discovery."""
        try:
            # Log the discovery
            await self.log_discovery(
                discovery['symbol'],
                discovery['discovery_type'],
                discovery['confidence'],
                discovery['reasoning'],
                discovery['metadata']
            )
            
            # Check if discovery meets trading criteria
            if self._should_trade_discovery(discovery):
                # Generate trading signal
                trade_signal = await self._generate_trade_signal(discovery)
                
                if trade_signal:
                    # Execute trade
                    await self._execute_discovery_trade(trade_signal)
            
        except Exception as e:
            logger.error(f"Error processing discovery: {e}")
    
    async def _should_trade_discovery(self, discovery: Dict[str, Any], position_size: float, price: float) -> bool:
        """Determine if a discovery should result in a trade."""
        # Check confidence threshold
        min_confidence = self.config.get('min_trading_confidence', 0.7)
        if discovery['confidence'] < min_confidence:
            return False

        # Check risk limits with the calculated position size
        if not await self.risk_manager.check_position_limits(discovery['symbol'], position_size, price):
            return False

        return True

    async def _generate_trade_signal(self, discovery: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a trade signal from a discovery."""
        try:
            # Fetch current price
            market_data = await self.market_data_manager.get_market_data(discovery['symbol'])
            if not market_data:
                logger.warning(f"No market data for {discovery['symbol']}")
                return None

            price = market_data.get('price', 0)

            # Calculate position size
            position_size = self.portfolio_manager.calculate_position_size(
                discovery['symbol'],
                discovery['confidence']
            )

            if position_size <= 0:
                return None

            # Check if we should trade this discovery
            if not await self._should_trade_discovery(discovery, position_size, price):
                return None

            trade_side = discovery['metadata'].get('direction', 'buy')

            trade_signal = {
                'symbol': discovery['symbol'],
                'side': trade_side,
                'quantity': position_size,
                'order_type': 'market',
                'price': price,
                'discovery_id': discovery.get('id'),
                'confidence': discovery['confidence'],
                'reasoning': discovery['reasoning']
            }
            return trade_signal

        except Exception as e:
            logger.error(f"Error generating trade signal: {e}")
            return None
    
    async def _execute_discovery_trade(self, trade_signal: Dict[str, Any]):
        """Execute a trade based on a discovery."""
        try:
            # Execute trade through trading engine
            executed_order = await self.trading_engine.submit_order(
                trade_signal['symbol'],
                trade_signal['side'],
                trade_signal['order_type'],
                trade_signal['quantity'],
                trade_signal.get('price')
            )
            
            # Log trade execution
            logger.info(f"Executed discovery trade: {executed_order.symbol} "
                       f"{executed_order.side} {executed_order.quantity} shares, Order ID: {executed_order.id}")
            
            # Update portfolio
            await self.portfolio_manager.update_position(
                executed_order.symbol,
                executed_order.quantity,
                executed_order.filled_price
            )
            
            # Update metrics
            self.metrics_collector.record_trade({
                "side": executed_order.side.value,
                "quantity": executed_order.quantity,
                "price": executed_order.filled_price,
            })
            
        except Exception as e:
            logger.error(f"Error executing discovery trade: {e}")
    
    def _get_monitored_symbols(self) -> List[str]:
        """Get list of symbols to monitor."""
        market_data_config = get_config_section('market_data')
        symbols = []
        
        # Get symbols from configuration
        for category in market_data_config['symbols'].values():
            if isinstance(category, dict):
                for subcategory in category.values():
                    if isinstance(subcategory, list):
                        for symbol_info in subcategory:
                            if isinstance(symbol_info, dict) and symbol_info.get('enabled', True):
                                symbols.append(symbol_info['symbol'])
        
        return symbols
    
    def _get_active_symbols(self) -> List[str]:
        """Get list of symbols to actively analyze."""
        # For now, return a subset of monitored symbols
        monitored = self._get_monitored_symbols()
        return monitored[:10]  # Limit to top 10 for performance
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Update system metrics
            self.state.performance_metrics.update({
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'uptime_seconds': (datetime.now() - self.state.start_time).total_seconds(),
                'error_count': self.state.error_count,
                'last_heartbeat': self.state.last_heartbeat.isoformat()
            })
            
            # Update trading metrics
            portfolio_value = self.portfolio_manager.get_portfolio_value()
            daily_pnl = self.portfolio_manager.get_daily_pnl()
            
            self.state.performance_metrics.update({
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'active_positions': len(self.portfolio_manager.get_positions()),
                'cash_balance': self.portfolio_manager.get_cash_balance()
            })
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _generate_performance_report(self):
        """Generate and log performance report."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'uptime_hours': (datetime.now() - self.state.start_time).total_seconds() / 3600,
                'system_metrics': {
                    'cpu_percent': self.state.performance_metrics.get('cpu_percent', 0),
                    'memory_percent': self.state.performance_metrics.get('memory_percent', 0),
                    'disk_usage': self.state.performance_metrics.get('disk_usage', 0),
                    'error_count': self.state.error_count
                },
                'trading_metrics': {
                    'portfolio_value': self.state.performance_metrics.get('portfolio_value', 0),
                    'daily_pnl': self.state.performance_metrics.get('daily_pnl', 0),
                    'active_positions': self.state.performance_metrics.get('active_positions', 0),
                    'cash_balance': self.state.performance_metrics.get('cash_balance', 0)
                },
                'service_status': {
                    'agents_initialized': self.state.agents_initialized,
                    'mcp_server_running': self.state.mcp_server_running,
                    'market_data_streaming': self.state.market_data_streaming,
                    'reddit_monitoring': self.state.reddit_monitoring,
                    'discovery_loop_running': self.state.discovery_loop_running
                }
            }
            
            logger.info(f"Performance Report: {json.dumps(report, indent=2)}")
            
            # Save report to database
            for key, value in report['system_metrics'].items():
                await self.db_manager.insert_performance_metric(f"system_{key}", value, datetime.now())
            for key, value in report['trading_metrics'].items():
                await self.db_manager.insert_performance_metric(f"trading_{key}", value, datetime.now())

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
    
    async def graceful_shutdown(self):
        """Perform graceful shutdown."""
        logger.info("Starting graceful shutdown...")
        
        try:
            # Stop discovery loop
            self.state.discovery_loop_running = False
            
            # Stop market data streaming
            if self.state.market_data_streaming:
                await self.market_data_manager.stop_streaming()
                self.state.market_data_streaming = False
            
            # Stop Reddit monitoring
            if self.state.reddit_monitoring:
                await self.reddit_monitor.stop_monitoring()
                self.state.reddit_monitoring = False
            
            # Close all positions if configured
            if self.config.get('close_positions_on_shutdown', False):
                await self.portfolio_manager.close_all_positions()
            
            # Generate final performance report
            await self._generate_performance_report()
            
            # Close database connections
            await self.db_manager.close()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
    
    # MCP Tool implementations
    async def get_market_data(self, symbol: str, data_type: str = "price") -> Dict[str, Any]:
        """Get market data for MCP tool."""
        return await self.market_data_manager.get_current_data(symbol, data_type)
    
    async def get_reddit_sentiment(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get Reddit sentiment for MCP tool."""
        return await self.reddit_monitor.get_sentiment(symbol, timeframe)
    
    async def calculate_risk_metrics(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Calculate risk metrics for MCP tool."""
        return await self.risk_manager.calculate_position_risk(symbol, quantity, price)
    
    async def execute_trade(self, symbol: str, side: str, quantity: float, 
                           order_type: str, price: Optional[float] = None) -> Dict[str, Any]:
        """Execute trade for MCP tool."""
        return await self.trading_engine.execute_trade(symbol, side, quantity, order_type, price)
    
    async def get_portfolio_status(self, include_positions: bool = True, 
                                  include_metrics: bool = True) -> Dict[str, Any]:
        """Get portfolio status for MCP tool."""
        return await self.portfolio_manager.get_status(include_positions, include_metrics)
    
    async def log_discovery(self, symbol: str, discovery_type: str, confidence: float,
                           reasoning: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log discovery for MCP tool."""
        discovery = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'discovery_type': discovery_type,
            'confidence': confidence,
            'reasoning': reasoning,
            'metadata': metadata or {}
        }
        
        # Save to database
        await self.db_manager.insert_alpha_discovery(
            symbol=discovery['symbol'],
            alpha_score=0,  # Placeholder for alpha score
            confidence=discovery['confidence'],
            factors=discovery['metadata'],
            timestamp=discovery['timestamp'],
            agent_id=discovery.get('agent_type')
        )
        
        # Log to file
        logger.info(f"Discovery logged: {symbol} - {discovery_type} (confidence: {confidence})")
        
        return {'status': 'success', 'discovery_id': discovery.get('id')}
    
    async def run(self):
        """Main execution method."""
        logger.info("Starting Alpha Discovery Platform...")
        
        try:
            self.state.is_running = True
            
            # Initialize all components
            await self.initialize_agents()
            await self.start_mcp_server()
            await self.start_market_data_streaming()
            await self.start_reddit_monitoring()
            
            # Start discovery loop
            await self.run_discovery_loop()
            
        except Exception as e:
            logger.error(f"Fatal error in main execution: {e}")
            logger.error(traceback.format_exc())
            capture_exception(e, extra={
                "component": "orchestrator",
                "error_type": "main_execution_error",
                "state": {
                    "is_running": self.state.is_running,
                    "agents_initialized": self.state.agents_initialized,
                    "mcp_server_running": self.state.mcp_server_running,
                    "market_data_streaming": self.state.market_data_streaming,
                    "reddit_monitoring": self.state.reddit_monitoring,
                    "discovery_loop_running": self.state.discovery_loop_running
                }
            })
            raise
        
        finally:
            self.state.is_running = False
            await self.graceful_shutdown()

def main():
    """Main entry point."""
    logger.info("Alpha Discovery Platform starting...")
    
    # Initialize Sentry for error monitoring
    try:
        init_sentry()
        set_tag("component", "alpha_discovery_main")
        set_context("application", {
            "name": "Alpha Discovery Platform",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        })
        logger.info("Sentry initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Sentry: {e}")
    
    # Set up environment
    os.makedirs('logs', exist_ok=True)
    
    # Initialize orchestrator
    orchestrator = AlphaDiscoveryOrchestrator()
    
    try:
        # Run the platform
        asyncio.run(orchestrator.run())
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        capture_message("Application shutdown requested by user", level="info")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        capture_exception(e, extra={
            "component": "main",
            "error_type": "fatal_error"
        })
        sys.exit(1)
    
    logger.info("Alpha Discovery Platform stopped")

if __name__ == "__main__":
    main() 