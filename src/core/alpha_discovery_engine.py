#!/usr/bin/env python3
"""
Alpha Discovery Engine

Core engine that coordinates all agents and discovery processes for the Alpha Discovery platform.
Manages the continuous discovery loop, agent coordination, and discovery evaluation.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

from configs.config_loader import get_config
from src.data.market_data_manager import MarketDataManager
from src.data.reddit_sentiment_monitor import RedditSentimentMonitor

# Import our existing specialized agents
from src.agents.microstructure_agent import MicrostructureAgent
from src.agents.altdata_agent import AlternativeDataAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.regime_agent import RegimeDetectionAgent

from src.execution.trading_engine import TradingEngine
from src.risk.risk_manager import RiskManager
from src.portfolio.portfolio_manager import PortfolioManager
from src.utils.database import DatabaseManager
from src.utils.metrics import MetricsCollector
from src.utils.performance import PerformanceTracker

logger = logging.getLogger(__name__)

class DiscoveryType(Enum):
    """Types of alpha discoveries."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    OPTIONS = "options"
    PAIRS_TRADING = "pairs_trading"
    EVENT_DRIVEN = "event_driven"

class DiscoveryStatus(Enum):
    """Status of discoveries."""
    PENDING = "pending"
    VALIDATED = "validated"
    EXECUTED = "executed"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class AlphaDiscovery:
    """Alpha discovery data structure."""
    id: str
    symbol: str
    discovery_type: DiscoveryType
    agent_source: str
    timestamp: datetime
    confidence: float
    signal_strength: float
    direction: str  # bullish, bearish, neutral
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Trading parameters
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    
    # Validation
    validation_score: float = 0.0
    risk_score: float = 0.0
    status: DiscoveryStatus = DiscoveryStatus.PENDING
    
    # Execution
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    order_id: Optional[str] = None
    
    # Performance tracking
    max_profit: float = 0.0
    max_loss: float = 0.0
    current_pnl: float = 0.0
    closed_pnl: Optional[float] = None
    
    # Expiration
    expires_at: Optional[datetime] = None

@dataclass
class DiscoveryStats:
    """Discovery statistics."""
    total_discoveries: int = 0
    validated_discoveries: int = 0
    executed_discoveries: int = 0
    successful_discoveries: int = 0
    failed_discoveries: int = 0
    average_confidence: float = 0.0
    average_return: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

class AlphaDiscoveryEngine:
    """Core alpha discovery engine."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.config = get_config('strategies')
        
        # Core components
        self.market_data_manager = MarketDataManager()
        self.reddit_monitor = RedditSentimentMonitor()
        self.trading_engine = TradingEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        self.db_manager = DatabaseManager(self.config)
        self.metrics_collector = MetricsCollector(self.config)
        self.performance_tracker = PerformanceTracker()
        
        # Agent instances
        self.microstructure_agent = MicrostructureAgent()
        self.altdata_agent = AlternativeDataAgent()
        self.strategy_agent = StrategyAgent()
        self.regime_agent = RegimeDetectionAgent()
        
        # Discovery management
        self.active_discoveries = {}
        self.discovery_history = []
        self.discovery_stats = DiscoveryStats()
        
        # Processing
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.discovery_queue = asyncio.Queue(maxsize=1000)
        self.validation_queue = asyncio.Queue(maxsize=500)
        
        # State
        self.is_running = False
        self.discovery_count = 0
        self.last_discovery_time = datetime.now()
        
        # Configuration
        self.discovery_interval = self.config.get('discovery_interval', 60)
        self.max_concurrent_discoveries = self.config.get('max_concurrent_discoveries', 50)
        self.discovery_timeout = self.config.get('discovery_timeout', 3600)  # 1 hour
        
        logger.info("Alpha Discovery Engine initialized")
    
    async def start_discovery_loop(self) -> None:
        """Start the continuous alpha discovery loop."""
        logger.info("Starting alpha discovery loop...")
        
        self.is_running = True
        
        # Start discovery processing tasks
        discovery_task = asyncio.create_task(self._discovery_loop())
        validation_task = asyncio.create_task(self._validation_loop())
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Wait for all tasks
        await asyncio.gather(
            discovery_task,
            validation_task,
            monitoring_task,
            cleanup_task
        )
    
    async def stop_discovery_loop(self) -> None:
        """Stop the discovery loop."""
        logger.info("Stopping alpha discovery loop...")
        self.is_running = False
        
        # Close all active discoveries
        for discovery in self.active_discoveries.values():
            if discovery.status == DiscoveryStatus.PENDING:
                discovery.status = DiscoveryStatus.EXPIRED
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Alpha discovery loop stopped")
    
    async def _discovery_loop(self) -> None:
        """Main discovery loop."""
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Get symbols to analyze
                symbols = await self._get_analysis_symbols()
                
                # Run discovery for each symbol
                discoveries = await self._run_discovery_cycle(symbols)
                
                # Queue discoveries for validation
                for discovery in discoveries:
                    if not self.validation_queue.full():
                        await self.validation_queue.put(discovery)
                    else:
                        logger.warning("Validation queue is full, dropping discovery")
                
                # Update statistics
                self.discovery_count += len(discoveries)
                self.last_discovery_time = datetime.now()
                
                # Log cycle statistics
                cycle_time = time.time() - loop_start
                logger.info(f"Discovery cycle completed in {cycle_time:.2f}s, "
                           f"found {len(discoveries)} discoveries")
                
                # Wait before next cycle
                await asyncio.sleep(self.discovery_interval)
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(self.discovery_interval * 2)
    
    async def _validation_loop(self) -> None:
        """Validation loop for discoveries."""
        while self.is_running:
            try:
                # Get discovery from queue
                discovery = await asyncio.wait_for(
                    self.validation_queue.get(),
                    timeout=1.0
                )
                
                # Validate discovery
                await self._validate_discovery(discovery)
                
                # Add to active discoveries if validated
                if discovery.status == DiscoveryStatus.VALIDATED:
                    self.active_discoveries[discovery.id] = discovery
                    
                    # Check if we should execute
                    if await self._should_execute_discovery(discovery):
                        await self._execute_discovery(discovery)
                
                # Mark task as done
                self.validation_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in validation loop: {e}")
                await asyncio.sleep(1)
    
    async def _monitoring_loop(self) -> None:
        """Monitor active discoveries."""
        while self.is_running:
            try:
                # Monitor each active discovery
                for discovery_id, discovery in list(self.active_discoveries.items()):
                    await self._monitor_discovery(discovery)
                    
                    # Remove completed or expired discoveries
                    if discovery.status in [DiscoveryStatus.COMPLETED, DiscoveryStatus.EXPIRED]:
                        self.discovery_history.append(discovery)
                        del self.active_discoveries[discovery_id]
                
                # Update statistics
                await self._update_discovery_stats()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup expired discoveries."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check for expired discoveries
                for discovery_id, discovery in list(self.active_discoveries.items()):
                    if (discovery.expires_at and 
                        current_time > discovery.expires_at and
                        discovery.status == DiscoveryStatus.PENDING):
                        
                        discovery.status = DiscoveryStatus.EXPIRED
                        logger.info(f"Discovery {discovery_id} expired")
                
                # Clean old history
                if len(self.discovery_history) > 10000:
                    self.discovery_history = self.discovery_history[-5000:]
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _get_analysis_symbols(self) -> List[str]:
        """Get symbols to analyze."""
        try:
            # Get symbols from market data manager
            all_symbols = await self.market_data_manager.get_available_symbols()
            
            # Filter based on volume and activity
            active_symbols = []
            for symbol in all_symbols:
                current_data = await self.market_data_manager.get_current_data(symbol)
                if current_data and current_data.get('volume', 0) > 100000:  # Min volume
                    active_symbols.append(symbol)
            
            # Limit to top symbols by volume
            return active_symbols[:50]
            
        except Exception as e:
            logger.error(f"Error getting analysis symbols: {e}")
            return []
    
    async def _run_discovery_cycle(self, symbols: List[str]) -> List[AlphaDiscovery]:
        """Run discovery cycle for symbols."""
        discoveries = []
        
        try:
            # Create tasks for each agent and symbol combination
            tasks = []
            
            for symbol in symbols:
                # Microstructure analysis
                micro_task = asyncio.create_task(
                    self._run_microstructure_analysis(symbol)
                )
                tasks.append(micro_task)
                
                # Alternative data analysis
                altdata_task = asyncio.create_task(
                    self._run_altdata_analysis(symbol)
                )
                tasks.append(altdata_task)
                
                # Strategy analysis
                strategy_task = asyncio.create_task(
                    self._run_strategy_analysis(symbol)
                )
                tasks.append(strategy_task)
                
                # Regime detection
                regime_task = asyncio.create_task(
                    self._run_regime_detection(symbol)
                )
                tasks.append(regime_task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in discovery task: {result}")
                    continue
                
                if result:
                    discoveries.extend(result)
            
            return discoveries
            
        except Exception as e:
            logger.error(f"Error in discovery cycle: {e}")
            return []
    
    async def _run_microstructure_analysis(self, symbol: str) -> List[AlphaDiscovery]:
        """Run microstructure analysis for a symbol."""
        try:
            # Get market data
            market_data = await self.market_data_manager.get_current_data(symbol, "all")
            historical_data = await self.market_data_manager.get_historical_data(symbol, "1mo")
            
            # Run microstructure analysis
            analysis_result = await self.microstructure_agent.analyze(symbol, market_data, historical_data)
            
            # Convert to discoveries
            discoveries = []
            for signal in analysis_result.get('signals', []):
                discovery = AlphaDiscovery(
                    id=f"micro_{symbol}_{int(time.time())}_{len(discoveries)}",
                    symbol=symbol,
                    discovery_type=DiscoveryType.TECHNICAL, # Microstructure is technical
                    agent_source="microstructure_agent",
                    timestamp=datetime.now(),
                    confidence=signal.get('confidence', 0.5),
                    signal_strength=signal.get('strength', 0.5),
                    direction=signal.get('direction', 'neutral'),
                    reasoning=signal.get('reasoning', ''),
                    metadata=signal.get('metadata', {}),
                    target_price=signal.get('target_price'),
                    stop_loss=signal.get('stop_loss'),
                    expires_at=datetime.now() + timedelta(hours=self.discovery_timeout)
                )
                discoveries.append(discovery)
            
            return discoveries
            
        except Exception as e:
            logger.error(f"Error in microstructure analysis for {symbol}: {e}")
            return []
    
    async def _run_altdata_analysis(self, symbol: str) -> List[AlphaDiscovery]:
        """Run alternative data analysis for a symbol."""
        try:
            # Get alternative data
            altdata_data = await self.market_data_manager.get_alternative_data(symbol)
            
            if not altdata_data:
                return []
            
            # Run alternative data analysis
            analysis_result = await self.altdata_agent.analyze(symbol, altdata_data)
            
            # Convert to discoveries
            discoveries = []
            for signal in analysis_result.get('signals', []):
                discovery = AlphaDiscovery(
                    id=f"altdata_{symbol}_{int(time.time())}_{len(discoveries)}",
                    symbol=symbol,
                    discovery_type=DiscoveryType.FUNDAMENTAL, # Alternative data is fundamental
                    agent_source="altdata_agent",
                    timestamp=datetime.now(),
                    confidence=signal.get('confidence', 0.5),
                    signal_strength=signal.get('strength', 0.5),
                    direction=signal.get('direction', 'neutral'),
                    reasoning=signal.get('reasoning', ''),
                    metadata=signal.get('metadata', {}),
                    target_price=signal.get('target_price'),
                    expires_at=datetime.now() + timedelta(days=7) # Shorter for alternative data
                )
                discoveries.append(discovery)
            
            return discoveries
            
        except Exception as e:
            logger.error(f"Error in alternative data analysis for {symbol}: {e}")
            return []
    
    async def _run_strategy_analysis(self, symbol: str) -> List[AlphaDiscovery]:
        """Run strategy analysis for a symbol."""
        try:
            # Get market data
            market_data = await self.market_data_manager.get_current_data(symbol, "all")
            historical_data = await self.market_data_manager.get_historical_data(symbol, "1mo")
            
            # Run strategy analysis
            analysis_result = await self.strategy_agent.analyze(symbol, market_data, historical_data)
            
            # Convert to discoveries
            discoveries = []
            for signal in analysis_result.get('signals', []):
                discovery = AlphaDiscovery(
                    id=f"strategy_{symbol}_{int(time.time())}_{len(discoveries)}",
                    symbol=symbol,
                    discovery_type=DiscoveryType.SENTIMENT, # Strategy is sentiment
                    agent_source="strategy_agent",
                    timestamp=datetime.now(),
                    confidence=signal.get('confidence', 0.5),
                    signal_strength=signal.get('strength', 0.5),
                    direction=signal.get('direction', 'neutral'),
                    reasoning=signal.get('reasoning', ''),
                    metadata=signal.get('metadata', {}),
                    expires_at=datetime.now() + timedelta(hours=6) # Shorter for strategy
                )
                discoveries.append(discovery)
            
            return discoveries
            
        except Exception as e:
            logger.error(f"Error in strategy analysis for {symbol}: {e}")
            return []
    
    async def _run_regime_detection(self, symbol: str) -> List[AlphaDiscovery]:
        """Run regime detection for a symbol."""
        try:
            # Get market data
            market_data = await self.market_data_manager.get_current_data(symbol, "all")
            historical_data = await self.market_data_manager.get_historical_data(symbol, "1mo")
            
            # Run regime detection
            analysis_result = await self.regime_agent.analyze(symbol, market_data, historical_data)
            
            # Convert to discoveries
            discoveries = []
            for signal in analysis_result.get('signals', []):
                discovery = AlphaDiscovery(
                    id=f"regime_{symbol}_{int(time.time())}_{len(discoveries)}",
                    symbol=symbol,
                    discovery_type=DiscoveryType.VOLATILITY, # Regime is volatility
                    agent_source="regime_agent",
                    timestamp=datetime.now(),
                    confidence=signal.get('confidence', 0.5),
                    signal_strength=signal.get('strength', 0.5),
                    direction=signal.get('direction', 'neutral'),
                    reasoning=signal.get('reasoning', ''),
                    metadata=signal.get('metadata', {}),
                    expires_at=datetime.now() + timedelta(hours=12) # Shorter for regime
                )
                discoveries.append(discovery)
            
            return discoveries
            
        except Exception as e:
            logger.error(f"Error in regime detection for {symbol}: {e}")
            return []
    
    async def _validate_discovery(self, discovery: AlphaDiscovery) -> None:
        """Validate a discovery."""
        try:
            # Risk validation
            risk_assessment = await self.risk_manager.assess_discovery_risk(discovery)
            discovery.risk_score = risk_assessment.get('risk_score', 0.5)
            
            # Market validation
            market_validation = await self._validate_market_conditions(discovery)
            
            # Cross-validation with other agents
            cross_validation = await self._cross_validate_discovery(discovery)
            
            # Calculate validation score
            validation_score = (
                (1.0 - discovery.risk_score) * 0.4 +
                market_validation * 0.3 +
                cross_validation * 0.3
            )
            
            discovery.validation_score = validation_score
            
            # Determine if validated
            min_validation_score = self.config.get('min_validation_score', 0.6)
            if validation_score >= min_validation_score:
                discovery.status = DiscoveryStatus.VALIDATED
                logger.info(f"Discovery {discovery.id} validated with score {validation_score:.2f}")
            else:
                discovery.status = DiscoveryStatus.REJECTED
                logger.info(f"Discovery {discovery.id} rejected with score {validation_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error validating discovery {discovery.id}: {e}")
            discovery.status = DiscoveryStatus.REJECTED
    
    async def _validate_market_conditions(self, discovery: AlphaDiscovery) -> float:
        """Validate market conditions for discovery."""
        try:
            # Get current market data
            market_data = await self.market_data_manager.get_current_data(discovery.symbol)
            
            # Check liquidity
            volume = market_data.get('volume', 0)
            avg_volume = market_data.get('avg_volume', 0)
            liquidity_score = min(1.0, volume / max(avg_volume, 1))
            
            # Check volatility
            volatility = market_data.get('volatility', 0)
            volatility_score = 1.0 - min(1.0, volatility / 0.5)  # Penalize high volatility
            
            # Check spread
            spread = market_data.get('spread', 0)
            spread_score = 1.0 - min(1.0, spread / 0.05)  # Penalize wide spreads
            
            # Overall market validation score
            market_score = (liquidity_score * 0.4 + volatility_score * 0.3 + spread_score * 0.3)
            
            return market_score
            
        except Exception as e:
            logger.error(f"Error validating market conditions: {e}")
            return 0.5
    
    async def _cross_validate_discovery(self, discovery: AlphaDiscovery) -> float:
        """Cross-validate discovery with other agents."""
        try:
            # Get opinions from other agents
            opinions = []
            
            # Microstructure validation
            if discovery.discovery_type != DiscoveryType.TECHNICAL:
                micro_opinion = await self.microstructure_agent.validate_discovery(discovery)
                opinions.append(micro_opinion)
            
            # Alternative data validation
            if discovery.discovery_type != DiscoveryType.FUNDAMENTAL:
                altdata_opinion = await self.altdata_agent.validate_discovery(discovery)
                opinions.append(altdata_opinion)
            
            # Strategy validation
            if discovery.discovery_type != DiscoveryType.SENTIMENT:
                strategy_opinion = await self.strategy_agent.validate_discovery(discovery)
                opinions.append(strategy_opinion)
            
            # Regime validation
            if discovery.discovery_type != DiscoveryType.VOLATILITY:
                regime_opinion = await self.regime_agent.validate_discovery(discovery)
                opinions.append(regime_opinion)
            
            # Calculate cross-validation score
            if opinions:
                cross_validation_score = sum(opinions) / len(opinions)
            else:
                cross_validation_score = 0.5
            
            return cross_validation_score
            
        except Exception as e:
            logger.error(f"Error cross-validating discovery: {e}")
            return 0.5
    
    async def _should_execute_discovery(self, discovery: AlphaDiscovery) -> bool:
        """Determine if discovery should be executed."""
        try:
            # Check confidence threshold
            min_confidence = self.config.get('min_execution_confidence', 0.7)
            if discovery.confidence < min_confidence:
                return False
            
            # Check validation score
            min_validation = self.config.get('min_execution_validation', 0.7)
            if discovery.validation_score < min_validation:
                return False
            
            # Check risk limits
            if not await self.risk_manager.can_execute_discovery(discovery):
                return False
            
            # Check portfolio limits
            if not await self.portfolio_manager.can_add_discovery(discovery):
                return False
            
            # Check for conflicting discoveries
            if await self._has_conflicting_discoveries(discovery):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking execution criteria: {e}")
            return False
    
    async def _has_conflicting_discoveries(self, discovery: AlphaDiscovery) -> bool:
        """Check for conflicting discoveries."""
        try:
            for active_discovery in self.active_discoveries.values():
                if (active_discovery.symbol == discovery.symbol and
                    active_discovery.direction != discovery.direction and
                    active_discovery.status == DiscoveryStatus.EXECUTED):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for conflicts: {e}")
            return False
    
    async def _execute_discovery(self, discovery: AlphaDiscovery) -> None:
        """Execute a discovery."""
        try:
            # Calculate position size
            position_size = await self.portfolio_manager.calculate_discovery_position_size(discovery)
            discovery.position_size = position_size
            
            # Execute trade
            execution_result = await self.trading_engine.execute_discovery_trade(discovery)
            
            if execution_result.get('success'):
                discovery.status = DiscoveryStatus.EXECUTED
                discovery.execution_price = execution_result.get('price')
                discovery.execution_time = datetime.now()
                discovery.order_id = execution_result.get('order_id')
                
                logger.info(f"Discovery {discovery.id} executed at {discovery.execution_price}")
                
                # Update portfolio
                await self.portfolio_manager.add_discovery_position(discovery)
                
                # Record metrics
                self.metrics_collector.record_discovery_execution(discovery)
                
            else:
                discovery.status = DiscoveryStatus.REJECTED
                logger.error(f"Failed to execute discovery {discovery.id}: {execution_result.get('error')}")
            
        except Exception as e:
            logger.error(f"Error executing discovery {discovery.id}: {e}")
            discovery.status = DiscoveryStatus.REJECTED
    
    async def _monitor_discovery(self, discovery: AlphaDiscovery) -> None:
        """Monitor an active discovery."""
        try:
            if discovery.status != DiscoveryStatus.EXECUTED:
                return
            
            # Get current market data
            market_data = await self.market_data_manager.get_current_data(discovery.symbol)
            current_price = market_data.get('price')
            
            if not current_price or not discovery.execution_price:
                return
            
            # Calculate current P&L
            if discovery.direction == 'bullish':
                discovery.current_pnl = (current_price - discovery.execution_price) * discovery.position_size
            else:
                discovery.current_pnl = (discovery.execution_price - current_price) * discovery.position_size
            
            # Update max profit/loss
            discovery.max_profit = max(discovery.max_profit, discovery.current_pnl)
            discovery.max_loss = min(discovery.max_loss, discovery.current_pnl)
            
            # Check exit conditions
            await self._check_exit_conditions(discovery, current_price)
            
        except Exception as e:
            logger.error(f"Error monitoring discovery {discovery.id}: {e}")
    
    async def _check_exit_conditions(self, discovery: AlphaDiscovery, current_price: float) -> None:
        """Check exit conditions for a discovery."""
        try:
            should_exit = False
            exit_reason = ""
            
            # Check stop loss
            if discovery.stop_loss:
                if ((discovery.direction == 'bullish' and current_price <= discovery.stop_loss) or
                    (discovery.direction == 'bearish' and current_price >= discovery.stop_loss)):
                    should_exit = True
                    exit_reason = "stop_loss"
            
            # Check take profit
            if discovery.take_profit:
                if ((discovery.direction == 'bullish' and current_price >= discovery.take_profit) or
                    (discovery.direction == 'bearish' and current_price <= discovery.take_profit)):
                    should_exit = True
                    exit_reason = "take_profit"
            
            # Check time-based exit
            if discovery.expires_at and datetime.now() > discovery.expires_at:
                should_exit = True
                exit_reason = "expired"
            
            # Check risk-based exit
            risk_exit = await self.risk_manager.should_exit_discovery(discovery)
            if risk_exit:
                should_exit = True
                exit_reason = "risk_management"
            
            # Execute exit if needed
            if should_exit:
                await self._exit_discovery(discovery, exit_reason)
                
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
    
    async def _exit_discovery(self, discovery: AlphaDiscovery, reason: str) -> None:
        """Exit a discovery position."""
        try:
            # Execute exit trade
            exit_result = await self.trading_engine.exit_discovery_trade(discovery)
            
            if exit_result.get('success'):
                discovery.status = DiscoveryStatus.COMPLETED
                discovery.closed_pnl = discovery.current_pnl
                
                logger.info(f"Discovery {discovery.id} exited: {reason}, P&L: {discovery.closed_pnl:.2f}")
                
                # Update portfolio
                await self.portfolio_manager.remove_discovery_position(discovery)
                
                # Record metrics
                self.metrics_collector.record_discovery_exit(discovery, reason)
                
            else:
                logger.error(f"Failed to exit discovery {discovery.id}: {exit_result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error exiting discovery {discovery.id}: {e}")
    
    async def _update_discovery_stats(self) -> None:
        """Update discovery statistics."""
        try:
            # Count discoveries by status
            total = len(self.discovery_history) + len(self.active_discoveries)
            validated = sum(1 for d in self.discovery_history if d.status != DiscoveryStatus.REJECTED)
            executed = sum(1 for d in self.discovery_history if d.status == DiscoveryStatus.EXECUTED)
            successful = sum(1 for d in self.discovery_history if d.closed_pnl and d.closed_pnl > 0)
            
            # Calculate averages
            if self.discovery_history:
                avg_confidence = sum(d.confidence for d in self.discovery_history) / len(self.discovery_history)
                
                completed_discoveries = [d for d in self.discovery_history if d.closed_pnl is not None]
                if completed_discoveries:
                    avg_return = sum(d.closed_pnl for d in completed_discoveries) / len(completed_discoveries)
                    win_rate = successful / len(completed_discoveries)
                else:
                    avg_return = 0.0
                    win_rate = 0.0
            else:
                avg_confidence = 0.0
                avg_return = 0.0
                win_rate = 0.0
            
            # Update stats
            self.discovery_stats = DiscoveryStats(
                total_discoveries=total,
                validated_discoveries=validated,
                executed_discoveries=executed,
                successful_discoveries=successful,
                failed_discoveries=executed - successful,
                average_confidence=avg_confidence,
                average_return=avg_return,
                win_rate=win_rate
            )
            
        except Exception as e:
            logger.error(f"Error updating discovery stats: {e}")
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return {
            'total_discoveries': self.discovery_stats.total_discoveries,
            'validated_discoveries': self.discovery_stats.validated_discoveries,
            'executed_discoveries': self.discovery_stats.executed_discoveries,
            'successful_discoveries': self.discovery_stats.successful_discoveries,
            'failed_discoveries': self.discovery_stats.failed_discoveries,
            'average_confidence': self.discovery_stats.average_confidence,
            'average_return': self.discovery_stats.average_return,
            'win_rate': self.discovery_stats.win_rate,
            'active_discoveries': len(self.active_discoveries),
            'discovery_count': self.discovery_count,
            'last_discovery_time': self.last_discovery_time.isoformat()
        }
    
    def get_active_discoveries(self) -> List[Dict[str, Any]]:
        """Get active discoveries."""
        return [
            {
                'id': d.id,
                'symbol': d.symbol,
                'type': d.discovery_type.value,
                'agent': d.agent_source,
                'confidence': d.confidence,
                'direction': d.direction,
                'status': d.status.value,
                'current_pnl': d.current_pnl,
                'timestamp': d.timestamp.isoformat()
            }
            for d in self.active_discoveries.values()
        ] 