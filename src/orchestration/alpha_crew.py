"""
Alpha Discovery Crew - Multi-Agent Orchestrator

This orchestrator manages all agents in a hierarchical structure with debate protocols,
consensus mechanisms, performance tracking, and human-in-the-loop capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict, deque
import uuid
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import redis
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import warnings
warnings.filterwarnings('ignore')

# Import our agents
from src.agents.microstructure_agent import MicrostructureAgent
from src.agents.altdata_agent import AlternativeDataAgent
from src.agents.regime_agent import RegimeDetectionAgent
from src.agents.strategy_agent import StrategyAgent
from src.models.model_manager import ModelManager
from src.utils.error_handling import handle_errors, AlphaDiscoveryError
from src.utils.monitoring import monitor_performance, track_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Agent role classification"""
    MICROSTRUCTURE = "MICROSTRUCTURE"
    ALTERNATIVE_DATA = "ALTERNATIVE_DATA"
    REGIME_DETECTION = "REGIME_DETECTION"
    STRATEGY_SYNTHESIS = "STRATEGY_SYNTHESIS"
    ORCHESTRATOR = "ORCHESTRATOR"

class DecisionType(Enum):
    """Decision type classification"""
    ROUTINE = "ROUTINE"
    HIGH_CONVICTION = "HIGH_CONVICTION"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class VoteType(Enum):
    """Vote type classification"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    ABSTAIN = "ABSTAIN"

@dataclass
class AgentVote:
    """Agent vote in consensus mechanism"""
    agent_id: str
    agent_role: AgentRole
    vote: VoteType
    confidence: float
    reasoning: str
    supporting_data: Dict[str, Any]
    timestamp: datetime
    weight: float = 1.0

@dataclass
class ConsensusResult:
    """Result of consensus mechanism"""
    decision: VoteType
    confidence: float
    participating_agents: List[str]
    vote_distribution: Dict[VoteType, int]
    weighted_score: float
    dissenting_opinions: List[AgentVote]
    consensus_strength: float
    requires_human_review: bool
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentPerformance:
    """Agent performance tracking"""
    agent_id: str
    agent_role: AgentRole
    total_predictions: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float
    sharpe_ratio: float
    max_drawdown: float
    last_updated: datetime
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    weight_adjustment: float = 1.0

@dataclass
class DebateSession:
    """Debate session between agents"""
    session_id: str
    topic: str
    participants: List[str]
    rounds: List[Dict[str, Any]]
    moderator: str
    start_time: datetime
    end_time: Optional[datetime]
    outcome: Optional[Dict[str, Any]]
    human_intervention: bool = False

class HumanInTheLoopInterface:
    """Interface for human-in-the-loop decisions"""
    
    def __init__(self):
        self.pending_decisions = {}
        self.decision_callbacks = {}
        self.approval_threshold = 0.7
        
    async def request_human_approval(self, decision_id: str, decision_data: Dict[str, Any], 
                                   timeout: int = 300) -> Dict[str, Any]:
        """Request human approval for critical decisions"""
        try:
            logger.info(f"Requesting human approval for decision: {decision_id}")
            
            # Store decision for human review
            self.pending_decisions[decision_id] = {
                "data": decision_data,
                "timestamp": datetime.now(),
                "timeout": timeout,
                "status": "pending"
            }
            
            # In production, this would integrate with a web interface or notification system
            # For now, we'll simulate human approval based on decision confidence
            confidence = decision_data.get("confidence", 0.5)
            
            # Simulate human review time
            await asyncio.sleep(2)
            
            # Simulate human decision based on confidence and risk
            if confidence > 0.8 and decision_data.get("risk_level", "medium") != "high":
                approval = {
                    "approved": True,
                    "feedback": "High confidence trade approved",
                    "modifications": {},
                    "human_id": "system_admin",
                    "timestamp": datetime.now()
                }
            elif confidence > 0.6:
                approval = {
                    "approved": True,
                    "feedback": "Approved with position size reduction",
                    "modifications": {"position_size_multiplier": 0.5},
                    "human_id": "system_admin",
                    "timestamp": datetime.now()
                }
            else:
                approval = {
                    "approved": False,
                    "feedback": "Insufficient confidence for execution",
                    "modifications": {},
                    "human_id": "system_admin",
                    "timestamp": datetime.now()
                }
            
            # Update decision status
            self.pending_decisions[decision_id]["status"] = "reviewed"
            self.pending_decisions[decision_id]["approval"] = approval
            
            logger.info(f"Human approval {'granted' if approval['approved'] else 'denied'} for {decision_id}")
            return approval
            
        except Exception as e:
            logger.error(f"Human approval request failed: {e}")
            return {
                "approved": False,
                "feedback": f"Approval system error: {e}",
                "modifications": {},
                "human_id": "system",
                "timestamp": datetime.now()
            }
    
    def get_pending_decisions(self) -> List[Dict[str, Any]]:
        """Get list of pending human decisions"""
        return [
            {"id": decision_id, **data} 
            for decision_id, data in self.pending_decisions.items() 
            if data["status"] == "pending"
        ]

class AgentCommunicationHub:
    """Manages communication and memory sharing between agents"""
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.shared_memory = {}
        self.communication_log = []
        self.subscriptions = defaultdict(list)
        
    async def broadcast_message(self, sender: str, message_type: str, content: Dict[str, Any]):
        """Broadcast message to all subscribed agents"""
        try:
            message = {
                "id": str(uuid.uuid4()),
                "sender": sender,
                "type": message_type,
                "content": content,
                "timestamp": datetime.now()
            }
            
            # Add to communication log
            self.communication_log.append(message)
            
            # Notify subscribers
            for subscriber in self.subscriptions.get(message_type, []):
                await self.message_queue.put({
                    "recipient": subscriber,
                    "message": message
                })
                
            logger.debug(f"Broadcast message from {sender}: {message_type}")
            
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
    
    def subscribe(self, agent_id: str, message_types: List[str]):
        """Subscribe agent to specific message types"""
        for msg_type in message_types:
            if agent_id not in self.subscriptions[msg_type]:
                self.subscriptions[msg_type].append(agent_id)
    
    def update_shared_memory(self, key: str, value: Any, agent_id: str):
        """Update shared memory with agent contribution"""
        self.shared_memory[key] = {
            "value": value,
            "updated_by": agent_id,
            "timestamp": datetime.now()
        }
    
    def get_shared_memory(self, key: str) -> Optional[Any]:
        """Get value from shared memory"""
        return self.shared_memory.get(key, {}).get("value")

class PerformanceTracker:
    """Tracks agent performance and adjusts weights"""
    
    def __init__(self):
        self.agent_performances = {}
        self.performance_history = defaultdict(list)
        self.weight_adjustments = defaultdict(lambda: 1.0)
        
    def update_performance(self, agent_id: str, prediction: Dict[str, Any], actual: Dict[str, Any]):
        """Update agent performance based on prediction accuracy"""
        try:
            # Calculate accuracy metrics
            predicted_direction = prediction.get("direction", "neutral")
            actual_direction = actual.get("direction", "neutral")
            
            is_correct = predicted_direction == actual_direction
            confidence = prediction.get("confidence", 0.5)
            
            # Update performance record
            if agent_id not in self.agent_performances:
                self.agent_performances[agent_id] = AgentPerformance(
                    agent_id=agent_id,
                    agent_role=AgentRole(prediction.get("agent_role", "ORCHESTRATOR")),
                    total_predictions=0,
                    correct_predictions=0,
                    accuracy=0.0,
                    avg_confidence=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    last_updated=datetime.now()
                )
            
            perf = self.agent_performances[agent_id]
            perf.total_predictions += 1
            if is_correct:
                perf.correct_predictions += 1
            
            perf.accuracy = perf.correct_predictions / perf.total_predictions
            perf.avg_confidence = (perf.avg_confidence * (perf.total_predictions - 1) + confidence) / perf.total_predictions
            perf.last_updated = datetime.now()
            
            # Add to history
            perf.performance_history.append({
                "prediction": prediction,
                "actual": actual,
                "correct": is_correct,
                "timestamp": datetime.now()
            })
            
            # Adjust weights based on performance
            self._adjust_weights(agent_id, perf)
            
            logger.debug(f"Updated performance for {agent_id}: {perf.accuracy:.2f} accuracy")
            
        except Exception as e:
            logger.error(f"Performance update failed for {agent_id}: {e}")
    
    def _adjust_weights(self, agent_id: str, performance: AgentPerformance):
        """Adjust agent weights based on performance"""
        try:
            # Base weight adjustment on accuracy and confidence calibration
            accuracy_factor = max(0.5, min(2.0, performance.accuracy / 0.6))  # Scale around 60% baseline
            confidence_factor = max(0.8, min(1.2, 1.0 - abs(performance.avg_confidence - performance.accuracy)))
            
            # Combine factors
            new_weight = accuracy_factor * confidence_factor
            
            # Smooth weight changes
            current_weight = self.weight_adjustments[agent_id]
            self.weight_adjustments[agent_id] = 0.8 * current_weight + 0.2 * new_weight
            
            performance.weight_adjustment = self.weight_adjustments[agent_id]
            
        except Exception as e:
            logger.error(f"Weight adjustment failed for {agent_id}: {e}")
    
    def get_agent_weight(self, agent_id: str) -> float:
        """Get current weight for agent"""
        return self.weight_adjustments.get(agent_id, 1.0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents"""
        return {
            agent_id: {
                "accuracy": perf.accuracy,
                "total_predictions": perf.total_predictions,
                "avg_confidence": perf.avg_confidence,
                "weight": perf.weight_adjustment,
                "last_updated": perf.last_updated.isoformat()
            }
            for agent_id, perf in self.agent_performances.items()
        }

class AlphaDiscoveryCrew:
    """
    Multi-Agent Orchestrator for Alpha Discovery
    
    Manages all agents in a hierarchical structure with debate protocols,
    consensus mechanisms, performance tracking, and human-in-the-loop capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_manager = ModelManager()
        
        # Initialize components
        self.communication_hub = AgentCommunicationHub()
        self.performance_tracker = PerformanceTracker()
        self.human_interface = HumanInTheLoopInterface()
        
        # Initialize agents
        self._initialize_agents()
        
        # Setup CrewAI orchestrator
        self._setup_orchestrator_crew()
        
        # Performance tracking
        self.session_history = []
        self.active_debates = {}
        self.consensus_cache = {}
        
        # Threading for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        logger.info("AlphaDiscoveryCrew initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all specialized agents"""
        try:
            self.microstructure_agent = MicrostructureAgent()
            self.altdata_agent = AlternativeDataAgent()
            self.regime_agent = RegimeDetectionAgent()
            self.strategy_agent = StrategyAgent()
            
            # Register agents with communication hub
            self.communication_hub.subscribe("microstructure", ["market_data", "regime_change"])
            self.communication_hub.subscribe("altdata", ["sentiment_update", "news_alert"])
            self.communication_hub.subscribe("regime", ["market_data", "volatility_spike"])
            self.communication_hub.subscribe("strategy", ["all"])
            
            logger.info("All agents initialized and registered")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise AlphaDiscoveryError(f"Failed to initialize agents: {e}")
    
    def _setup_orchestrator_crew(self):
        """Setup CrewAI orchestrator crew"""
        
        # Chief Investment Officer Agent
        self.cio_agent = Agent(
            role='Chief Investment Officer',
            goal='Orchestrate alpha discovery process and make final investment decisions',
            backstory="""You are the Chief Investment Officer responsible for overseeing 
            the entire alpha discovery process. You coordinate between specialized agents, 
            moderate debates, and make final investment decisions. You have ultimate 
            responsibility for portfolio performance and risk management.""",
            verbose=True,
            allow_delegation=True
        )
        
        # Risk Officer Agent
        self.risk_officer = Agent(
            role='Chief Risk Officer',
            goal='Monitor and control portfolio risk across all strategies',
            backstory="""You are the Chief Risk Officer responsible for monitoring 
            portfolio risk, setting position limits, and ensuring compliance with 
            risk management policies. You have veto power over high-risk strategies.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Debate Moderator Agent
        self.debate_moderator = Agent(
            role='Debate Moderator',
            goal='Facilitate productive debates between agents and ensure fair discussion',
            backstory="""You are an experienced debate moderator who ensures fair 
            and productive discussions between agents. You maintain order, ensure 
            all viewpoints are heard, and guide debates toward actionable conclusions.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Setup hierarchical crew
        self.orchestrator_crew = Crew(
            agents=[self.cio_agent, self.risk_officer, self.debate_moderator],
            verbose=True,
            process=Process.hierarchical,
            manager_llm=self.model_manager.get_model("claude")
        )
    
    @handle_errors
    @monitor_performance
    async def discover_alpha(self, symbols: List[str], timeframe: str = "1d", 
                           parallel: bool = True) -> Dict[str, Any]:
        """
        Main alpha discovery process
        
        Args:
            symbols: List of symbols to analyze
            timeframe: Analysis timeframe
            parallel: Whether to run agents in parallel
            
        Returns:
            Alpha discovery results with consensus
        """
        try:
            logger.info(f"Starting alpha discovery for {len(symbols)} symbols")
            
            # Phase 1: Collect signals from all agents
            if parallel:
                agent_signals = await self._collect_signals_parallel(symbols, timeframe)
            else:
                agent_signals = await self._collect_signals_sequential(symbols, timeframe)
            
            # Phase 2: Execute debate protocol
            debate_results = await self.execute_debate(agent_signals, symbols)
            
            # Phase 3: Get consensus
            consensus = await self.get_consensus(agent_signals, debate_results)
            
            # Phase 4: Check if human approval needed
            if consensus.requires_human_review:
                human_approval = await self._request_human_approval(consensus, symbols)
                consensus.metadata["human_approval"] = human_approval
            
            # Phase 5: Generate final recommendations
            recommendations = await self._generate_recommendations(consensus, symbols)
            
            # Store session
            session = {
                "session_id": str(uuid.uuid4()),
                "symbols": symbols,
                "timeframe": timeframe,
                "agent_signals": agent_signals,
                "debate_results": debate_results,
                "consensus": consensus,
                "recommendations": recommendations,
                "timestamp": datetime.now()
            }
            
            self.session_history.append(session)
            
            logger.info(f"Alpha discovery completed - Consensus: {consensus.decision.value}")
            
            return {
                "session_id": session["session_id"],
                "consensus": consensus,
                "recommendations": recommendations,
                "agent_signals": agent_signals,
                "debate_summary": debate_results,
                "performance_metrics": self.performance_tracker.get_performance_summary()
            }
            
        except Exception as e:
            logger.error(f"Alpha discovery failed: {e}")
            raise AlphaDiscoveryError(f"Failed to discover alpha: {e}")
    
    @handle_errors
    async def execute_debate(self, agent_signals: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """
        Execute debate protocol between agents
        
        Args:
            agent_signals: Signals from all agents
            symbols: Symbols being analyzed
            
        Returns:
            Debate results and outcomes
        """
        try:
            logger.info("Executing inter-agent debate protocol")
            
            # Create debate session
            debate_id = str(uuid.uuid4())
            debate_session = DebateSession(
                session_id=debate_id,
                topic=f"Alpha opportunities in {', '.join(symbols)}",
                participants=list(agent_signals.keys()),
                rounds=[],
                moderator="debate_moderator",
                start_time=datetime.now(),
                end_time=None
            )
            
            # Round 1: Initial positions
            round1 = await self._debate_round_initial_positions(agent_signals, symbols)
            debate_session.rounds.append(round1)
            
            # Round 2: Cross-examination
            round2 = await self._debate_round_cross_examination(agent_signals, round1)
            debate_session.rounds.append(round2)
            
            # Round 3: Consensus building
            round3 = await self._debate_round_consensus_building(agent_signals, round1, round2)
            debate_session.rounds.append(round3)
            
            # Finalize debate
            debate_session.end_time = datetime.now()
            debate_session.outcome = await self._finalize_debate(debate_session)
            
            # Store debate
            self.active_debates[debate_id] = debate_session
            
            logger.info(f"Debate completed - Duration: {debate_session.end_time - debate_session.start_time}")
            
            return {
                "debate_id": debate_id,
                "session": debate_session,
                "outcome": debate_session.outcome,
                "rounds": len(debate_session.rounds),
                "duration": (debate_session.end_time - debate_session.start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Debate execution failed: {e}")
            return {"error": str(e)}
    
    @handle_errors
    async def get_consensus(self, agent_signals: Dict[str, Any], debate_results: Dict[str, Any]) -> ConsensusResult:
        """
        Get consensus from all agents using weighted voting
        
        Args:
            agent_signals: Signals from all agents
            debate_results: Results from debate process
            
        Returns:
            Consensus result with voting details
        """
        try:
            logger.info("Calculating agent consensus")
            
            # Collect votes from all agents
            votes = []
            
            for agent_id, signals in agent_signals.items():
                if signals and not isinstance(signals, Exception):
                    vote = await self._extract_agent_vote(agent_id, signals, debate_results)
                    if vote:
                        votes.append(vote)
            
            if not votes:
                return ConsensusResult(
                    decision=VoteType.NEUTRAL,
                    confidence=0.0,
                    participating_agents=[],
                    vote_distribution={},
                    weighted_score=0.0,
                    dissenting_opinions=[],
                    consensus_strength=0.0,
                    requires_human_review=True,
                    timestamp=datetime.now(),
                    metadata={"error": "No valid votes collected"}
                )
            
            # Calculate weighted consensus
            consensus = self._calculate_weighted_consensus(votes)
            
            # Determine if human review is required
            consensus.requires_human_review = self._requires_human_review(consensus)
            
            logger.info(f"Consensus reached: {consensus.decision.value} (confidence: {consensus.confidence:.2f})")
            
            return consensus
            
        except Exception as e:
            logger.error(f"Consensus calculation failed: {e}")
            return ConsensusResult(
                decision=VoteType.NEUTRAL,
                confidence=0.0,
                participating_agents=[],
                vote_distribution={},
                weighted_score=0.0,
                dissenting_opinions=[],
                consensus_strength=0.0,
                requires_human_review=True,
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
    
    async def _collect_signals_parallel(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Collect signals from all agents in parallel"""
        try:
            # Create tasks for parallel execution
            tasks = {
                "microstructure": self._run_microstructure_analysis(symbols, timeframe),
                "altdata": self._run_altdata_analysis(symbols, timeframe),
                "regime": self._run_regime_analysis(symbols, timeframe),
                "strategy": self._run_strategy_analysis(symbols, timeframe)
            }
            
            # Execute in parallel
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Map results back to agent names
            agent_signals = {}
            for i, (agent_name, task) in enumerate(tasks.items()):
                agent_signals[agent_name] = results[i]
            
            return agent_signals
            
        except Exception as e:
            logger.error(f"Parallel signal collection failed: {e}")
            return {}
    
    async def _collect_signals_sequential(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Collect signals from all agents sequentially"""
        try:
            agent_signals = {}
            
            # Microstructure analysis
            agent_signals["microstructure"] = await self._run_microstructure_analysis(symbols, timeframe)
            
            # Alternative data analysis
            agent_signals["altdata"] = await self._run_altdata_analysis(symbols, timeframe)
            
            # Regime analysis
            agent_signals["regime"] = await self._run_regime_analysis(symbols, timeframe)
            
            # Strategy synthesis
            agent_signals["strategy"] = await self._run_strategy_analysis(symbols, timeframe)
            
            return agent_signals
            
        except Exception as e:
            logger.error(f"Sequential signal collection failed: {e}")
            return {}
    
    async def _run_microstructure_analysis(self, symbols: List[str], timeframe: str) -> Any:
        """Run microstructure analysis"""
        try:
            signals = []
            for symbol in symbols:
                signal = await self.microstructure_agent.generate_signals(symbol, timeframe)
                signals.extend(signal)
            return signals
        except Exception as e:
            logger.error(f"Microstructure analysis failed: {e}")
            return e
    
    async def _run_altdata_analysis(self, symbols: List[str], timeframe: str) -> Any:
        """Run alternative data analysis"""
        try:
            return await self.altdata_agent.find_alpha_signals(symbols, timeframe)
        except Exception as e:
            logger.error(f"Alternative data analysis failed: {e}")
            return e
    
    async def _run_regime_analysis(self, symbols: List[str], timeframe: str) -> Any:
        """Run regime analysis"""
        try:
            return await self.regime_agent.detect_current_regime(symbols)
        except Exception as e:
            logger.error(f"Regime analysis failed: {e}")
            return e
    
    async def _run_strategy_analysis(self, symbols: List[str], timeframe: str) -> Any:
        """Run strategy analysis"""
        try:
            return await self.strategy_agent.synthesize_strategy(symbols, timeframe)
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            return e
    
    async def _debate_round_initial_positions(self, agent_signals: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """Round 1: Initial positions from each agent"""
        try:
            positions = {}
            
            for agent_id, signals in agent_signals.items():
                if signals and not isinstance(signals, Exception):
                    position = await self._get_agent_position(agent_id, signals, symbols)
                    positions[agent_id] = position
            
            return {
                "round": 1,
                "type": "initial_positions",
                "positions": positions,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Initial positions round failed: {e}")
            return {"error": str(e)}
    
    async def _debate_round_cross_examination(self, agent_signals: Dict[str, Any], round1: Dict[str, Any]) -> Dict[str, Any]:
        """Round 2: Cross-examination between agents"""
        try:
            examinations = {}
            positions = round1.get("positions", {})
            
            # Each agent examines others' positions
            for examiner_id in positions.keys():
                examinations[examiner_id] = {}
                
                for target_id, target_position in positions.items():
                    if examiner_id != target_id:
                        examination = await self._cross_examine_position(examiner_id, target_id, target_position)
                        examinations[examiner_id][target_id] = examination
            
            return {
                "round": 2,
                "type": "cross_examination",
                "examinations": examinations,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Cross-examination round failed: {e}")
            return {"error": str(e)}
    
    async def _debate_round_consensus_building(self, agent_signals: Dict[str, Any], round1: Dict[str, Any], round2: Dict[str, Any]) -> Dict[str, Any]:
        """Round 3: Consensus building"""
        try:
            consensus_attempts = {}
            
            for agent_id in agent_signals.keys():
                if agent_signals[agent_id] and not isinstance(agent_signals[agent_id], Exception):
                    consensus_attempt = await self._build_consensus_position(agent_id, round1, round2)
                    consensus_attempts[agent_id] = consensus_attempt
            
            return {
                "round": 3,
                "type": "consensus_building",
                "consensus_attempts": consensus_attempts,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Consensus building round failed: {e}")
            return {"error": str(e)}
    
    async def _get_agent_position(self, agent_id: str, signals: Any, symbols: List[str]) -> Dict[str, Any]:
        """Get agent's position on the symbols"""
        try:
            # Create position prompt based on agent type
            position_prompt = f"""
            As the {agent_id} agent, provide your position on trading opportunities in {', '.join(symbols)}.
            
            Based on your analysis, provide:
            1. Overall market view (bullish/bearish/neutral)
            2. Top 3 trading opportunities
            3. Key risks to monitor
            4. Confidence level (0-1)
            5. Recommended position sizing
            
            Be specific and analytical. Support your position with your specialized expertise.
            """
            
            position = await self.model_manager.get_completion(
                prompt=position_prompt,
                model_type="claude",
                max_tokens=1500
            )
            
            return {
                "agent_id": agent_id,
                "position": position,
                "signals": str(signals)[:500],  # Truncated for brevity
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get position for {agent_id}: {e}")
            return {"error": str(e)}
    
    async def _cross_examine_position(self, examiner_id: str, target_id: str, target_position: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-examine another agent's position"""
        try:
            examination_prompt = f"""
            As the {examiner_id} agent, critically examine the {target_id} agent's position:
            
            {target_position.get('position', 'No position available')}
            
            Provide:
            1. What aspects do you agree with?
            2. What concerns do you have?
            3. What additional risks should be considered?
            4. How would you modify their recommendations?
            
            Be constructive but thorough in your analysis.
            """
            
            examination = await self.model_manager.get_completion(
                prompt=examination_prompt,
                model_type="claude",
                max_tokens=1000
            )
            
            return {
                "examiner": examiner_id,
                "target": target_id,
                "examination": examination,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Cross-examination failed: {e}")
            return {"error": str(e)}
    
    async def _build_consensus_position(self, agent_id: str, round1: Dict[str, Any], round2: Dict[str, Any]) -> Dict[str, Any]:
        """Build consensus position after debate"""
        try:
            consensus_prompt = f"""
            As the {agent_id} agent, after hearing all positions and cross-examinations, 
            provide your final consensus position.
            
            Consider:
            - Initial positions from all agents
            - Cross-examination feedback
            - Areas of agreement and disagreement
            
            Provide your final recommendation with:
            1. Modified position (if any)
            2. Areas where you've changed your view
            3. Remaining concerns
            4. Final confidence level
            """
            
            consensus = await self.model_manager.get_completion(
                prompt=consensus_prompt,
                model_type="claude",
                max_tokens=1000
            )
            
            return {
                "agent_id": agent_id,
                "consensus_position": consensus,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Consensus building failed for {agent_id}: {e}")
            return {"error": str(e)}
    
    async def _finalize_debate(self, debate_session: DebateSession) -> Dict[str, Any]:
        """Finalize debate and extract key outcomes"""
        try:
            # Analyze debate for key themes and conclusions
            outcome = {
                "key_agreements": [],
                "key_disagreements": [],
                "risk_concerns": [],
                "opportunity_consensus": [],
                "recommended_actions": []
            }
            
            # In production, this would use more sophisticated NLP analysis
            # For now, we'll provide a structured summary
            outcome["summary"] = "Multi-agent debate completed with structured analysis of positions"
            outcome["quality_score"] = 0.85
            outcome["consensus_strength"] = 0.75
            
            return outcome
            
        except Exception as e:
            logger.error(f"Debate finalization failed: {e}")
            return {"error": str(e)}
    
    async def _extract_agent_vote(self, agent_id: str, signals: Any, debate_results: Dict[str, Any]) -> Optional[AgentVote]:
        """Extract vote from agent signals and debate participation"""
        try:
            # Determine vote based on signals
            if isinstance(signals, list) and signals:
                # For list of signals, aggregate sentiment
                total_confidence = sum(getattr(s, 'confidence', 0.5) for s in signals)
                avg_confidence = total_confidence / len(signals)
                
                total_expected_return = sum(getattr(s, 'expected_return', 0) for s in signals)
                avg_expected_return = total_expected_return / len(signals)
                
                if avg_expected_return > 0.02:
                    vote = VoteType.BUY if avg_expected_return > 0.05 else VoteType.BUY
                elif avg_expected_return < -0.02:
                    vote = VoteType.SELL if avg_expected_return < -0.05 else VoteType.SELL
                else:
                    vote = VoteType.NEUTRAL
                    
            else:
                # Single signal or regime state
                vote = VoteType.NEUTRAL
                avg_confidence = 0.5
            
            # Get agent weight
            weight = self.performance_tracker.get_agent_weight(agent_id)
            
            return AgentVote(
                agent_id=agent_id,
                agent_role=AgentRole(agent_id.upper()),
                vote=vote,
                confidence=avg_confidence,
                reasoning=f"Based on {agent_id} analysis",
                supporting_data={"signals_count": len(signals) if isinstance(signals, list) else 1},
                timestamp=datetime.now(),
                weight=weight
            )
            
        except Exception as e:
            logger.error(f"Vote extraction failed for {agent_id}: {e}")
            return None
    
    def _calculate_weighted_consensus(self, votes: List[AgentVote]) -> ConsensusResult:
        """Calculate weighted consensus from agent votes"""
        try:
            # Count votes by type
            vote_counts = defaultdict(int)
            weighted_scores = defaultdict(float)
            total_weight = 0
            
            for vote in votes:
                vote_counts[vote.vote] += 1
                weighted_scores[vote.vote] += vote.confidence * vote.weight
                total_weight += vote.weight
            
            # Find winning vote
            if not votes:
                decision = VoteType.NEUTRAL
                confidence = 0.0
                weighted_score = 0.0
            else:
                # Weight-adjusted decision
                if total_weight > 0:
                    normalized_scores = {vote_type: score/total_weight for vote_type, score in weighted_scores.items()}
                    decision = max(normalized_scores.keys(), key=lambda x: normalized_scores[x])
                    weighted_score = normalized_scores[decision]
                else:
                    decision = max(vote_counts.keys(), key=lambda x: vote_counts[x])
                    weighted_score = 0.5
                
                # Calculate confidence
                confidence = weighted_score if weighted_score > 0 else 0.5
            
            # Find dissenting opinions
            dissenting_opinions = [vote for vote in votes if vote.vote != decision]
            
            # Calculate consensus strength
            consensus_strength = (len(votes) - len(dissenting_opinions)) / len(votes) if votes else 0
            
            return ConsensusResult(
                decision=decision,
                confidence=confidence,
                participating_agents=[vote.agent_id for vote in votes],
                vote_distribution=dict(vote_counts),
                weighted_score=weighted_score,
                dissenting_opinions=dissenting_opinions,
                consensus_strength=consensus_strength,
                requires_human_review=False,  # Will be set later
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Consensus calculation failed: {e}")
            return ConsensusResult(
                decision=VoteType.NEUTRAL,
                confidence=0.0,
                participating_agents=[],
                vote_distribution={},
                weighted_score=0.0,
                dissenting_opinions=[],
                consensus_strength=0.0,
                requires_human_review=True,
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
    
    def _requires_human_review(self, consensus: ConsensusResult) -> bool:
        """Determine if consensus requires human review"""
        try:
            # High-conviction trades with strong consensus
            if consensus.decision in [VoteType.STRONG_BUY, VoteType.STRONG_SELL]:
                return True
            
            # Low consensus strength
            if consensus.consensus_strength < 0.6:
                return True
            
            # Low confidence
            if consensus.confidence < 0.7:
                return True
            
            # Significant dissenting opinions
            if len(consensus.dissenting_opinions) > len(consensus.participating_agents) * 0.4:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Human review determination failed: {e}")
            return True
    
    async def _request_human_approval(self, consensus: ConsensusResult, symbols: List[str]) -> Dict[str, Any]:
        """Request human approval for critical decisions"""
        try:
            decision_data = {
                "symbols": symbols,
                "decision": consensus.decision.value,
                "confidence": consensus.confidence,
                "consensus_strength": consensus.consensus_strength,
                "participating_agents": consensus.participating_agents,
                "dissenting_opinions": len(consensus.dissenting_opinions),
                "risk_level": "high" if consensus.decision in [VoteType.STRONG_BUY, VoteType.STRONG_SELL] else "medium"
            }
            
            decision_id = f"alpha_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return await self.human_interface.request_human_approval(decision_id, decision_data)
            
        except Exception as e:
            logger.error(f"Human approval request failed: {e}")
            return {"approved": False, "feedback": f"Approval system error: {e}"}
    
    async def _generate_recommendations(self, consensus: ConsensusResult, symbols: List[str]) -> Dict[str, Any]:
        """Generate final recommendations based on consensus"""
        try:
            recommendations = {
                "action": consensus.decision.value,
                "confidence": consensus.confidence,
                "symbols": symbols,
                "position_sizing": self._calculate_position_sizing(consensus),
                "risk_management": self._generate_risk_management_rules(consensus),
                "execution_strategy": self._generate_execution_strategy(consensus),
                "monitoring_plan": self._generate_monitoring_plan(consensus),
                "timestamp": datetime.now()
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_position_sizing(self, consensus: ConsensusResult) -> Dict[str, Any]:
        """Calculate position sizing based on consensus"""
        try:
            base_size = 0.05  # 5% base position
            
            # Adjust based on confidence
            confidence_multiplier = consensus.confidence
            
            # Adjust based on consensus strength
            consensus_multiplier = consensus.consensus_strength
            
            # Final position size
            position_size = base_size * confidence_multiplier * consensus_multiplier
            
            return {
                "base_size": base_size,
                "confidence_multiplier": confidence_multiplier,
                "consensus_multiplier": consensus_multiplier,
                "final_size": min(position_size, 0.15),  # Cap at 15%
                "max_portfolio_impact": 0.15
            }
            
        except Exception as e:
            logger.error(f"Position sizing calculation failed: {e}")
            return {"final_size": 0.02}  # Conservative fallback
    
    def _generate_risk_management_rules(self, consensus: ConsensusResult) -> Dict[str, Any]:
        """Generate risk management rules"""
        return {
            "stop_loss": 0.02,  # 2% stop loss
            "take_profit": 0.05,  # 5% take profit
            "position_limit": 0.15,  # 15% position limit
            "correlation_limit": 0.3,  # 30% correlation limit
            "volatility_threshold": 0.25,  # 25% volatility threshold
            "review_frequency": "daily"
        }
    
    def _generate_execution_strategy(self, consensus: ConsensusResult) -> Dict[str, Any]:
        """Generate execution strategy"""
        return {
            "execution_type": "TWAP",  # Time-weighted average price
            "execution_window": "4h",  # 4-hour execution window
            "participation_rate": 0.1,  # 10% participation rate
            "urgency": "normal" if consensus.confidence < 0.8 else "high"
        }
    
    def _generate_monitoring_plan(self, consensus: ConsensusResult) -> Dict[str, Any]:
        """Generate monitoring plan"""
        return {
            "monitoring_frequency": "5min",
            "alert_thresholds": {
                "price_movement": 0.02,
                "volume_spike": 2.0,
                "volatility_spike": 1.5
            },
            "review_triggers": [
                "regime_change",
                "earnings_announcement",
                "news_event"
            ]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            return {
                "agent_performance": self.performance_tracker.get_performance_summary(),
                "session_count": len(self.session_history),
                "active_debates": len(self.active_debates),
                "consensus_cache_size": len(self.consensus_cache),
                "last_session": self.session_history[-1]["timestamp"].isoformat() if self.session_history else None
            }
        except Exception as e:
            logger.error(f"Performance summary failed: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Example usage of AlphaDiscoveryCrew"""
    
    # Initialize crew
    crew = AlphaDiscoveryCrew()
    
    # Test symbols
    symbols = ["AAPL", "GOOGL", "TSLA", "MSFT"]
    
    try:
        # Discover alpha opportunities
        print("Discovering alpha opportunities...")
        results = await crew.discover_alpha(symbols, timeframe="1d", parallel=True)
        
        print(f"\nAlpha Discovery Results:")
        print(f"- Session ID: {results['session_id']}")
        print(f"- Consensus: {results['consensus'].decision.value}")
        print(f"- Confidence: {results['consensus'].confidence:.2f}")
        print(f"- Consensus Strength: {results['consensus'].consensus_strength:.2f}")
        print(f"- Participating Agents: {', '.join(results['consensus'].participating_agents)}")
        
        # Show recommendations
        recommendations = results['recommendations']
        print(f"\nRecommendations:")
        print(f"- Action: {recommendations['action']}")
        print(f"- Position Size: {recommendations['position_sizing']['final_size']:.2%}")
        print(f"- Risk Management: {recommendations['risk_management']}")
        
        # Show performance summary
        performance = crew.get_performance_summary()
        print(f"\nPerformance Summary:")
        print(f"- Total Sessions: {performance['session_count']}")
        print(f"- Active Debates: {performance['active_debates']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 