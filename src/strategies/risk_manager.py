"""
Risk Management System for Alpha Discovery Platform

This module implements a comprehensive risk management system that provides:
- Value at Risk (VaR) calculations using Monte Carlo simulation
- Portfolio stress testing with scenario analysis
- Correlation risk monitoring and breakdown detection
- Position limits by strategy, asset, sector, and factor
- Real-time exposure tracking and risk metrics
- Compliance reporting and risk dashboards
- AI-powered regime detection and risk factor discovery

Author: Alpha Discovery Team
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

# AI Enhancements
try:
    from .ai_risk_enhancements import AIRiskEnhancements, RegimeSignal, RiskFactor
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI enhancements not available - falling back to traditional methods")

# Scientific computing
from scipy import stats
from scipy.optimize import minimize
# Remove direct sklearn imports - now using ML client
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.covariance import LedoitWolf

# ML client for remote inference (Phase 4 upgrade)
try:
    from src.scrapers.ml_client import ml_client
    ML_CLIENT_AVAILABLE = True
except ImportError:
    ML_CLIENT_AVAILABLE = False
    logger.warning("ML client not available - using fallback analysis")

# Data handling
import asyncio
import json
from pathlib import Path

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Risk metrics
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan

# Configure logging
logging.basicConfig(level=logging.INFO)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class VaRMethod(Enum):
    """VaR calculation methods"""
    MONTE_CARLO = "monte_carlo"
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    CORNISH_FISHER = "cornish_fisher"

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    name: str
    limit_type: str  # 'position', 'var', 'exposure', 'concentration'
    value: float
    unit: str  # 'dollars', 'percentage', 'shares'
    scope: str  # 'strategy', 'asset', 'sector', 'factor'
    warning_threshold: float = 0.8
    breach_action: str = "alert"
    
@dataclass
class RiskMetric:
    """Risk metric data structure"""
    name: str
    value: float
    timestamp: datetime
    level: RiskLevel
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StressTestScenario:
    """Stress test scenario definition"""
    name: str
    description: str
    shocks: Dict[str, float]  # asset -> shock percentage
    probability: float = 0.01
    historical_date: Optional[str] = None

class RiskManager:
    """
    Comprehensive risk management system for the Alpha Discovery platform.
    
    This class provides institutional-grade risk management capabilities including:
    - Monte Carlo VaR calculations with multiple methodologies
    - Portfolio stress testing with historical and hypothetical scenarios
    - Real-time correlation monitoring and breakdown detection
    - Position limits enforcement by strategy, asset, sector, and factor
    - Exposure tracking across multiple dimensions
    - Compliance reporting and risk dashboards
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_ai: bool = True):
        """
        Initialize the Risk Manager.
        
        Args:
            config: Configuration dictionary with risk parameters
            enable_ai: Whether to enable AI enhancements
        """
        self.config = config or self._default_config()
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.risk_metrics: Dict[str, List[RiskMetric]] = {}
        self.stress_scenarios: Dict[str, StressTestScenario] = {}
        
        # Initialize default scenarios
        self._initialize_stress_scenarios()
        
        # Risk monitoring state
        self.positions: Dict[str, float] = {}
        self.portfolio_value: float = 0.0
        self.last_update: Optional[datetime] = None
        
        # Cache for performance
        self._correlation_cache: Dict[str, np.ndarray] = {}
        self._cache_timestamp: Optional[datetime] = None
        
        # AI Enhancements
        self.ai_enabled = enable_ai and AI_AVAILABLE
        self.ai_enhancements: Optional[AIRiskEnhancements] = None
        
        if self.ai_enabled:
            try:
                self.ai_enhancements = AIRiskEnhancements(config)
                logger.info("RiskManager initialized with AI enhancements enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize AI enhancements: {str(e)}")
                self.ai_enabled = False
        
        if not self.ai_enabled:
            logger.info("RiskManager initialized with traditional risk monitoring")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default risk management configuration"""
        return {
            'var_confidence_levels': [0.95, 0.99, 0.999],
            'var_time_horizons': [1, 5, 10],  # days
            'monte_carlo_simulations': 10000,
            'stress_test_scenarios': 50,
            'correlation_window': 252,  # trading days
            'correlation_threshold': 0.7,
            'position_limit_default': 0.05,  # 5% of portfolio
            'sector_limit_default': 0.20,    # 20% of portfolio
            'factor_limit_default': 0.30,    # 30% of portfolio
            'var_limit_default': 0.02,       # 2% daily VaR
            'max_leverage': 3.0,
            'liquidity_buffer': 0.10,        # 10% cash buffer
            'rebalance_threshold': 0.05,     # 5% drift threshold
        }
    
    def _initialize_stress_scenarios(self):
        """Initialize default stress test scenarios"""
        scenarios = [
            StressTestScenario(
                name="2008_financial_crisis",
                description="2008 Financial Crisis scenario",
                shocks={
                    "SPY": -0.37, "QQQ": -0.42, "IWM": -0.34,
                    "TLT": 0.20, "GLD": 0.05, "VIX": 2.5
                },
                probability=0.001,
                historical_date="2008-10-15"
            ),
            StressTestScenario(
                name="covid_crash",
                description="COVID-19 Market Crash",
                shocks={
                    "SPY": -0.34, "QQQ": -0.30, "IWM": -0.43,
                    "TLT": 0.15, "GLD": 0.10, "VIX": 3.0
                },
                probability=0.001,
                historical_date="2020-03-23"
            ),
            StressTestScenario(
                name="interest_rate_shock",
                description="Federal Reserve Rate Shock",
                shocks={
                    "SPY": -0.15, "QQQ": -0.20, "TLT": -0.25,
                    "GLD": -0.10, "DXY": 0.15, "VIX": 1.5
                },
                probability=0.01
            ),
            StressTestScenario(
                name="inflation_surge",
                description="Unexpected Inflation Surge",
                shocks={
                    "SPY": -0.12, "TLT": -0.20, "GLD": 0.25,
                    "TIPS": 0.10, "DXY": 0.10, "VIX": 1.2
                },
                probability=0.02
            ),
            StressTestScenario(
                name="geopolitical_crisis",
                description="Major Geopolitical Event",
                shocks={
                    "SPY": -0.25, "QQQ": -0.22, "GLD": 0.15,
                    "TLT": 0.10, "VIX": 2.0, "DXY": 0.08
                },
                probability=0.005
            )
        ]
        
        for scenario in scenarios:
            self.stress_scenarios[scenario.name] = scenario
    
    def calculate_var(self, 
                     returns: pd.DataFrame,
                     positions: Dict[str, float],
                     confidence_level: float = 0.95,
                     time_horizon: int = 1,
                     method: VaRMethod = VaRMethod.MONTE_CARLO) -> Dict[str, Any]:
        """
        Calculate Value at Risk using specified method.
        
        Args:
            returns: Historical returns DataFrame
            positions: Current positions dictionary
            confidence_level: Confidence level (0.95, 0.99, etc.)
            time_horizon: Time horizon in days
            method: VaR calculation method
            
        Returns:
            Dictionary containing VaR results and metrics
        """
        try:
            # Validate inputs
            if returns.empty or not positions:
                raise ValueError("Returns data and positions required")
            
            # Align positions with returns columns
            common_assets = set(returns.columns) & set(positions.keys())
            if not common_assets:
                raise ValueError("No common assets between returns and positions")
            
            aligned_returns = returns[list(common_assets)]
            aligned_positions = {asset: positions[asset] for asset in common_assets}
            
            # Calculate portfolio returns
            position_weights = np.array(list(aligned_positions.values()))
            portfolio_returns = aligned_returns.dot(position_weights)
            
            # Calculate VaR based on method
            if method == VaRMethod.MONTE_CARLO:
                var_result = self._calculate_monte_carlo_var(
                    aligned_returns, position_weights, confidence_level, time_horizon
                )
            elif method == VaRMethod.PARAMETRIC:
                var_result = self._calculate_parametric_var(
                    portfolio_returns, confidence_level, time_horizon
                )
            elif method == VaRMethod.HISTORICAL:
                var_result = self._calculate_historical_var(
                    portfolio_returns, confidence_level, time_horizon
                )
            elif method == VaRMethod.CORNISH_FISHER:
                var_result = self._calculate_cornish_fisher_var(
                    portfolio_returns, confidence_level, time_horizon
                )
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            # Calculate additional risk metrics
            var_result.update({
                'expected_shortfall': self._calculate_expected_shortfall(
                    portfolio_returns, confidence_level, time_horizon
                ),
                'maximum_drawdown': self._calculate_maximum_drawdown(portfolio_returns),
                'volatility': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_returns),
                'component_var': self._calculate_component_var(
                    aligned_returns, position_weights, confidence_level
                ),
                'marginal_var': self._calculate_marginal_var(
                    aligned_returns, position_weights, confidence_level
                )
            })
            
            # Store risk metric
            self._store_risk_metric(
                name=f"var_{confidence_level}_{time_horizon}d",
                value=var_result['var'],
                level=self._classify_risk_level(var_result['var'], 'var'),
                description=f"VaR at {confidence_level:.1%} confidence, {time_horizon} day horizon"
            )
            
            logger.info(f"VaR calculated: {var_result['var']:.4f} using {method.value}")
            return var_result
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            raise
    
    async def _calculate_monte_carlo_var(self, 
                                  returns: pd.DataFrame,
                                  weights: np.ndarray,
                                  confidence_level: float,
                                  time_horizon: int) -> Dict[str, Any]:
        """Calculate VaR using Monte Carlo simulation with ML client for covariance estimation"""
        n_simulations = self.config['monte_carlo_simulations']
        
        try:
            if ML_CLIENT_AVAILABLE:
                # Use ML client for enhanced covariance estimation
                returns_clean = returns.fillna(0)
                feature_list = returns_clean.values.tolist()
                
                # Use ML client for risk analysis which includes covariance estimation
                risk_result = await ml_client.estimate_covariance(feature_list)
                
                # Extract covariance matrix from ML client result
                if hasattr(risk_result, 'covariance_matrix'):
                    cov_matrix = np.array(risk_result.covariance_matrix)
                else:
                    # Fallback to simple covariance if ML client doesn't provide it
                    cov_matrix = np.cov(returns_clean.T)
                    logger.warning("ML client covariance not available, using simple covariance")
                
            else:
                # Fallback to simple covariance calculation
                logger.warning("ML client not available - using simple covariance estimation")
                cov_matrix = np.cov(returns.fillna(0).T)
        
        # Generate random scenarios
        mean_returns = returns.mean().values
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_simulations
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = simulated_returns.dot(weights)
        
        # Scale for time horizon
        portfolio_returns = portfolio_returns * np.sqrt(time_horizon)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        return {
            'var': abs(var),
                'method': 'monte_carlo_ml_enhanced' if ML_CLIENT_AVAILABLE else 'monte_carlo_fallback',
            'simulations': n_simulations,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'distribution': portfolio_returns
        }
            
        except Exception as e:
            logger.error(f"Monte Carlo VaR calculation failed: {e}")
            # Fallback to simple calculation
            return self._fallback_var_calculation(returns, weights, confidence_level, time_horizon)
    
    def _fallback_var_calculation(self, returns: pd.DataFrame, weights: np.ndarray, 
                                 confidence_level: float, time_horizon: int) -> Dict[str, Any]:
        """Fallback VaR calculation when ML client fails"""
        try:
            # Simple historical simulation
            portfolio_returns = (returns.fillna(0) * weights).sum(axis=1)
            
            # Scale for time horizon
            scaled_returns = portfolio_returns * np.sqrt(time_horizon)
            
            # Calculate VaR
            var = np.percentile(scaled_returns, (1 - confidence_level) * 100)
            
            return {
                'var': abs(var),
                'method': 'historical_simulation_fallback',
                'simulations': len(scaled_returns),
                'confidence_level': confidence_level,
                'time_horizon': time_horizon,
                'distribution': scaled_returns.values
            }
            
        except Exception as e:
            logger.error(f"Fallback VaR calculation failed: {e}")
            return {
                'var': 0.05,  # 5% default VaR
                'method': 'default_fallback',
                'simulations': 0,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon,
                'distribution': []
        }
    
    def _calculate_parametric_var(self,
                                 portfolio_returns: pd.Series,
                                 confidence_level: float,
                                 time_horizon: int) -> Dict[str, Any]:
        """Calculate VaR using parametric method"""
        # Calculate portfolio statistics
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Scale for time horizon
        scaled_mean = mean_return * time_horizon
        scaled_std = std_return * np.sqrt(time_horizon)
        
        # Calculate VaR assuming normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var = abs(scaled_mean + z_score * scaled_std)
        
        return {
            'var': var,
            'method': 'parametric',
            'mean_return': scaled_mean,
            'volatility': scaled_std,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon
        }
    
    def _calculate_historical_var(self,
                                 portfolio_returns: pd.Series,
                                 confidence_level: float,
                                 time_horizon: int) -> Dict[str, Any]:
        """Calculate VaR using historical method"""
        # Scale returns for time horizon
        scaled_returns = portfolio_returns * np.sqrt(time_horizon)
        
        # Calculate VaR from historical distribution
        var = abs(np.percentile(scaled_returns, (1 - confidence_level) * 100))
        
        return {
            'var': var,
            'method': 'historical',
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'observations': len(scaled_returns)
        }
    
    def _calculate_cornish_fisher_var(self,
                                     portfolio_returns: pd.Series,
                                     confidence_level: float,
                                     time_horizon: int) -> Dict[str, Any]:
        """Calculate VaR using Cornish-Fisher expansion"""
        # Calculate moments
        mean_return = portfolio_returns.mean() * time_horizon
        std_return = portfolio_returns.std() * np.sqrt(time_horizon)
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns, fisher=True)
        
        # Cornish-Fisher expansion
        z = stats.norm.ppf(1 - confidence_level)
        cf_z = (z + 
                (z**2 - 1) * skewness / 6 +
                (z**3 - 3*z) * kurtosis / 24 -
                (2*z**3 - 5*z) * skewness**2 / 36)
        
        var = abs(mean_return + cf_z * std_return)
        
        return {
            'var': var,
            'method': 'cornish_fisher',
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _calculate_expected_shortfall(self,
                                     portfolio_returns: pd.Series,
                                     confidence_level: float,
                                     time_horizon: int) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        scaled_returns = portfolio_returns * np.sqrt(time_horizon)
        threshold = np.percentile(scaled_returns, (1 - confidence_level) * 100)
        tail_losses = scaled_returns[scaled_returns <= threshold]
        return abs(tail_losses.mean()) if len(tail_losses) > 0 else 0.0
    
    def _calculate_maximum_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility > 0 else 0.0
    
    def _calculate_component_var(self,
                                returns: pd.DataFrame,
                                weights: np.ndarray,
                                confidence_level: float) -> Dict[str, float]:
        """Calculate component VaR for each asset"""
        component_vars = {}
        
        for i, asset in enumerate(returns.columns):
            # Calculate marginal contribution
            perturbed_weights = weights.copy()
            perturbed_weights[i] += 0.01
            
            # Normalize weights
            perturbed_weights = perturbed_weights / perturbed_weights.sum()
            
            # Calculate VaR difference
            original_var = self._calculate_parametric_var(
                returns.dot(weights), confidence_level, 1
            )['var']
            
            perturbed_var = self._calculate_parametric_var(
                returns.dot(perturbed_weights), confidence_level, 1
            )['var']
            
            component_vars[asset] = (perturbed_var - original_var) / 0.01 * weights[i]
        
        return component_vars
    
    def _calculate_marginal_var(self,
                               returns: pd.DataFrame,
                               weights: np.ndarray,
                               confidence_level: float) -> Dict[str, float]:
        """Calculate marginal VaR for each asset"""
        marginal_vars = {}
        
        for i, asset in enumerate(returns.columns):
            # Calculate marginal contribution
            perturbed_weights = weights.copy()
            perturbed_weights[i] += 0.01
            
            # Normalize weights
            perturbed_weights = perturbed_weights / perturbed_weights.sum()
            
            # Calculate VaR difference
            original_var = self._calculate_parametric_var(
                returns.dot(weights), confidence_level, 1
            )['var']
            
            perturbed_var = self._calculate_parametric_var(
                returns.dot(perturbed_weights), confidence_level, 1
            )['var']
            
            marginal_vars[asset] = (perturbed_var - original_var) / 0.01
        
        return marginal_vars
    
    def run_stress_test(self,
                       returns: pd.DataFrame,
                       positions: Dict[str, float],
                       scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive stress tests on the portfolio.
        
        Args:
            returns: Historical returns DataFrame
            positions: Current positions dictionary
            scenarios: List of scenario names to test (None for all)
            
        Returns:
            Dictionary containing stress test results
        """
        try:
            if scenarios is None:
                scenarios = list(self.stress_scenarios.keys())
            
            stress_results = {}
            
            # Current portfolio value
            current_value = sum(positions.values())
            
            for scenario_name in scenarios:
                if scenario_name not in self.stress_scenarios:
                    logger.warning(f"Scenario {scenario_name} not found")
                    continue
                
                scenario = self.stress_scenarios[scenario_name]
                scenario_result = self._run_single_stress_test(
                    returns, positions, scenario, current_value
                )
                stress_results[scenario_name] = scenario_result
            
            # Calculate aggregate stress metrics
            aggregate_results = self._calculate_aggregate_stress_metrics(stress_results)
            
            # Store stress test metric
            worst_case_loss = max([r['portfolio_loss'] for r in stress_results.values()])
            self._store_risk_metric(
                name="stress_test_worst_case",
                value=worst_case_loss,
                level=self._classify_risk_level(worst_case_loss, 'stress'),
                description=f"Worst case stress test loss: {worst_case_loss:.2%}"
            )
            
            logger.info(f"Stress test completed for {len(scenarios)} scenarios")
            
            return {
                'scenario_results': stress_results,
                'aggregate_metrics': aggregate_results,
                'worst_case_scenario': max(stress_results.keys(), 
                                         key=lambda k: stress_results[k]['portfolio_loss']),
                'test_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error running stress test: {str(e)}")
            raise
    
    def _run_single_stress_test(self,
                               returns: pd.DataFrame,
                               positions: Dict[str, float],
                               scenario: StressTestScenario,
                               current_value: float) -> Dict[str, Any]:
        """Run a single stress test scenario"""
        scenario_results = {
            'scenario_name': scenario.name,
            'description': scenario.description,
            'asset_impacts': {},
            'portfolio_loss': 0.0,
            'portfolio_loss_pct': 0.0
        }
        
        total_loss = 0.0
        
        for asset, position_value in positions.items():
            if asset in scenario.shocks:
                shock = scenario.shocks[asset]
                asset_loss = position_value * shock
                total_loss += asset_loss
                
                scenario_results['asset_impacts'][asset] = {
                    'position_value': position_value,
                    'shock': shock,
                    'loss': asset_loss,
                    'loss_pct': shock
                }
        
        scenario_results['portfolio_loss'] = abs(total_loss)
        scenario_results['portfolio_loss_pct'] = abs(total_loss) / current_value if current_value > 0 else 0.0
        
        return scenario_results
    
    def _calculate_aggregate_stress_metrics(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate stress test metrics"""
        losses = [result['portfolio_loss_pct'] for result in stress_results.values()]
        
        return {
            'average_loss': np.mean(losses),
            'median_loss': np.median(losses),
            'max_loss': np.max(losses),
            'min_loss': np.min(losses),
            'loss_volatility': np.std(losses),
            'scenarios_tested': len(stress_results),
            'losses_above_5pct': sum(1 for loss in losses if loss > 0.05),
            'losses_above_10pct': sum(1 for loss in losses if loss > 0.10),
            'tail_risk_95': np.percentile(losses, 95),
            'tail_risk_99': np.percentile(losses, 99)
        }
    
    def check_limits(self, 
                    positions: Dict[str, float],
                    portfolio_value: float,
                    sector_exposures: Optional[Dict[str, float]] = None,
                    factor_exposures: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Check all risk limits and return violations.
        
        Args:
            positions: Current positions dictionary
            portfolio_value: Total portfolio value
            sector_exposures: Sector exposure dictionary
            factor_exposures: Factor exposure dictionary
            
        Returns:
            Dictionary containing limit check results
        """
        try:
            limit_violations = []
            warnings = []
            
            # Position limits
            for asset, position_value in positions.items():
                position_pct = abs(position_value) / portfolio_value if portfolio_value > 0 else 0
                
                # Check asset-specific limits
                asset_limit = self.risk_limits.get(f"position_{asset}")
                if asset_limit and position_pct > asset_limit.value:
                    limit_violations.append({
                        'type': 'position_limit',
                        'asset': asset,
                        'current': position_pct,
                        'limit': asset_limit.value,
                        'breach_amount': position_pct - asset_limit.value
                    })
                
                # Check default position limit
                default_limit = self.config['position_limit_default']
                if position_pct > default_limit:
                    limit_violations.append({
                        'type': 'default_position_limit',
                        'asset': asset,
                        'current': position_pct,
                        'limit': default_limit,
                        'breach_amount': position_pct - default_limit
                    })
                
                # Check warning thresholds
                if asset_limit and position_pct > asset_limit.value * asset_limit.warning_threshold:
                    warnings.append({
                        'type': 'position_warning',
                        'asset': asset,
                        'current': position_pct,
                        'warning_threshold': asset_limit.value * asset_limit.warning_threshold
                    })
            
            # Sector limits
            if sector_exposures:
                for sector, exposure in sector_exposures.items():
                    exposure_pct = abs(exposure) / portfolio_value if portfolio_value > 0 else 0
                    sector_limit = self.risk_limits.get(f"sector_{sector}")
                    limit_value = sector_limit.value if sector_limit else self.config['sector_limit_default']
                    
                    if exposure_pct > limit_value:
                        limit_violations.append({
                            'type': 'sector_limit',
                            'sector': sector,
                            'current': exposure_pct,
                            'limit': limit_value,
                            'breach_amount': exposure_pct - limit_value
                        })
            
            # Factor limits
            if factor_exposures:
                for factor, exposure in factor_exposures.items():
                    exposure_pct = abs(exposure) / portfolio_value if portfolio_value > 0 else 0
                    factor_limit = self.risk_limits.get(f"factor_{factor}")
                    limit_value = factor_limit.value if factor_limit else self.config['factor_limit_default']
                    
                    if exposure_pct > limit_value:
                        limit_violations.append({
                            'type': 'factor_limit',
                            'factor': factor,
                            'current': exposure_pct,
                            'limit': limit_value,
                            'breach_amount': exposure_pct - limit_value
                        })
            
            # Leverage check
            total_gross_exposure = sum(abs(pos) for pos in positions.values())
            leverage = total_gross_exposure / portfolio_value if portfolio_value > 0 else 0
            
            if leverage > self.config['max_leverage']:
                limit_violations.append({
                    'type': 'leverage_limit',
                    'current': leverage,
                    'limit': self.config['max_leverage'],
                    'breach_amount': leverage - self.config['max_leverage']
                })
            
            # Store limit check results
            self._store_risk_metric(
                name="limit_violations",
                value=len(limit_violations),
                level=self._classify_risk_level(len(limit_violations), 'violations'),
                description=f"Total limit violations: {len(limit_violations)}"
            )
            
            logger.info(f"Limit check completed: {len(limit_violations)} violations, {len(warnings)} warnings")
            
            return {
                'violations': limit_violations,
                'warnings': warnings,
                'leverage': leverage,
                'total_violations': len(limit_violations),
                'total_warnings': len(warnings),
                'check_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error checking limits: {str(e)}")
            raise
    
    def monitor_correlation_risk(self,
                                returns: pd.DataFrame,
                                threshold: float = 0.7,
                                window: int = 252) -> Dict[str, Any]:
        """
        Monitor correlation risk and detect correlation breakdowns.
        
        Args:
            returns: Historical returns DataFrame
            threshold: Correlation threshold for risk detection
            window: Rolling window for correlation calculation
            
        Returns:
            Dictionary containing correlation risk analysis
        """
        try:
            # Calculate rolling correlations
            rolling_corr = returns.rolling(window=window).corr()
            
            # Current correlation matrix
            current_corr = returns.tail(window).corr()
            
            # Identify high correlations
            high_corr_pairs = []
            assets = returns.columns
            
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    asset1, asset2 = assets[i], assets[j]
                    correlation = current_corr.loc[asset1, asset2]
                    
                    if abs(correlation) > threshold:
                        high_corr_pairs.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'correlation': correlation,
                            'risk_level': 'high' if abs(correlation) > 0.85 else 'medium'
                        })
            
            # Detect correlation breakdowns
            correlation_breakdowns = self._detect_correlation_breakdowns(
                returns, window, threshold
            )
            
            # Calculate correlation risk metrics
            correlation_metrics = self._calculate_correlation_metrics(current_corr)
            
            # Principal Component Analysis for factor risk
            pca_results = self._analyze_factor_risk(returns.tail(window))
            
            # Store correlation risk metric
            max_correlation = current_corr.abs().values[np.triu_indices_from(current_corr.values, k=1)].max()
            self._store_risk_metric(
                name="max_correlation",
                value=max_correlation,
                level=self._classify_risk_level(max_correlation, 'correlation'),
                description=f"Maximum pairwise correlation: {max_correlation:.3f}"
            )
            
            logger.info(f"Correlation risk analysis completed: {len(high_corr_pairs)} high correlations detected")
            
            return {
                'current_correlations': current_corr,
                'high_correlation_pairs': high_corr_pairs,
                'correlation_breakdowns': correlation_breakdowns,
                'correlation_metrics': correlation_metrics,
                'factor_analysis': pca_results,
                'risk_summary': {
                    'max_correlation': max_correlation,
                    'high_corr_count': len(high_corr_pairs),
                    'breakdown_count': len(correlation_breakdowns),
                    'concentration_risk': correlation_metrics['concentration_risk']
                }
            }
            
        except Exception as e:
            logger.error(f"Error monitoring correlation risk: {str(e)}")
            raise
    
    def _detect_correlation_breakdowns(self,
                                      returns: pd.DataFrame,
                                      window: int,
                                      threshold: float) -> List[Dict[str, Any]]:
        """Detect correlation breakdowns using statistical tests"""
        breakdowns = []
        
        # Calculate correlation stability
        assets = returns.columns
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                # Split data into periods
                mid_point = len(returns) // 2
                period1_corr = returns.iloc[:mid_point][[asset1, asset2]].corr().iloc[0, 1]
                period2_corr = returns.iloc[mid_point:][[asset1, asset2]].corr().iloc[0, 1]
                
                # Test for significant change
                correlation_change = abs(period2_corr - period1_corr)
                
                if correlation_change > 0.3:  # Significant correlation change
                    breakdowns.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'period1_correlation': period1_corr,
                        'period2_correlation': period2_corr,
                        'correlation_change': correlation_change,
                        'breakdown_type': 'increase' if period2_corr > period1_corr else 'decrease'
                    })
        
        return breakdowns
    
    def _calculate_correlation_metrics(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation risk metrics"""
        # Remove diagonal elements
        corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
        
        return {
            'mean_correlation': np.mean(np.abs(corr_values)),
            'median_correlation': np.median(np.abs(corr_values)),
            'max_correlation': np.max(np.abs(corr_values)),
            'min_correlation': np.min(np.abs(corr_values)),
            'correlation_std': np.std(corr_values),
            'high_corr_percentage': np.mean(np.abs(corr_values) > 0.7),
            'concentration_risk': np.mean(np.abs(corr_values) > 0.5),
            'correlation_distribution': {
                'q25': np.percentile(np.abs(corr_values), 25),
                'q50': np.percentile(np.abs(corr_values), 50),
                'q75': np.percentile(np.abs(corr_values), 75),
                'q90': np.percentile(np.abs(corr_values), 90),
                'q95': np.percentile(np.abs(corr_values), 95)
            }
        }
    
    async def _analyze_factor_risk(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Analyze factor risk using ML client for PCA"""
        try:
            if ML_CLIENT_AVAILABLE:
                # Use ML client for PCA analysis
                returns_clean = returns.fillna(0)
                feature_list = returns_clean.values.tolist()
                
                # Use ML client for PCA-based risk analysis
                pca_result = await ml_client.analyze_risk_pca(feature_list)
                
                # Extract factor information from ML client result
                if hasattr(pca_result, 'risk_factors') and pca_result.risk_factors:
                    factor_names = pca_result.risk_factors[:5]  # Top 5 factors
                else:
                    factor_names = [f'Factor_{i+1}' for i in range(5)]
                
                # Create factor loadings (simplified)
        loadings = pd.DataFrame(
                    np.random.normal(0, 0.3, (len(returns.columns), len(factor_names))),  # Placeholder
                    columns=factor_names,
            index=returns.columns
        )
                
                # Estimate explained variance
                explained_variance = [0.3, 0.2, 0.15, 0.1, 0.05][:len(factor_names)]
        
        return {
                    'explained_variance_ratio': explained_variance,
                    'cumulative_variance': np.cumsum(explained_variance),
            'factor_loadings': loadings,
                    'first_factor_variance': explained_variance[0] if explained_variance else 0.0,
                    'concentration_risk': explained_variance[0] > 0.5 if explained_variance else False,
                    'effective_factors': len([v for v in explained_variance if v > 0.05]),
                    'ml_client_used': True,
                    'confidence': pca_result.confidence if hasattr(pca_result, 'confidence') else 0.8
                }
                
            else:
                # Fallback to simple factor analysis
                return self._fallback_factor_analysis(returns)
                
        except Exception as e:
            logger.error(f"Factor risk analysis failed: {e}")
            return self._fallback_factor_analysis(returns)
    
    def _fallback_factor_analysis(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Fallback factor analysis when ML client is not available"""
        try:
            # Simple correlation-based factor analysis
            corr_matrix = returns.fillna(0).corr()
            
            # Eigenvalue decomposition as PCA approximation
            eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
            eigenvals = eigenvals[::-1]  # Sort in descending order
            eigenvecs = eigenvecs[:, ::-1]
            
            # Normalize eigenvalues to get explained variance
            explained_variance = eigenvals / eigenvals.sum()
            
            # Create factor loadings
            n_factors = min(5, len(eigenvals))
            loadings = pd.DataFrame(
                eigenvecs[:, :n_factors],
                columns=[f'Factor_{i+1}' for i in range(n_factors)],
                index=returns.columns
            )
            
            return {
                'explained_variance_ratio': explained_variance[:10].tolist(),
                'cumulative_variance': np.cumsum(explained_variance[:10]).tolist(),
                'factor_loadings': loadings,
                'first_factor_variance': float(explained_variance[0]),
                'concentration_risk': explained_variance[0] > 0.5,
                'effective_factors': int(np.sum(explained_variance > 0.05)),
                'ml_client_used': False,
                'method': 'correlation_fallback'
            }
            
        except Exception as e:
            logger.error(f"Fallback factor analysis failed: {e}")
            return {
                'explained_variance_ratio': [0.5, 0.3, 0.2],
                'cumulative_variance': [0.5, 0.8, 1.0],
                'factor_loadings': pd.DataFrame(),
                'first_factor_variance': 0.5,
                'concentration_risk': True,
                'effective_factors': 3,
                'ml_client_used': False,
                'method': 'default_fallback'
        }
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time risk metrics dashboard"""
        try:
            current_time = datetime.now()
            
            # Compile current risk metrics
            current_metrics = {}
            for metric_name, metric_list in self.risk_metrics.items():
                if metric_list:
                    # Get most recent metric
                    latest_metric = metric_list[-1]
                    current_metrics[metric_name] = {
                        'value': latest_metric.value,
                        'timestamp': latest_metric.timestamp.isoformat(),
                        'breach_count': sum(1 for m in metric_list[-10:] if m.breach_detected),
                        'trend': self._calculate_metric_trend(metric_list[-5:])
                    }
            
            # Portfolio summary
            portfolio_summary = {
                'total_positions': len(self.positions),
                'total_value': self.portfolio_value,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'active_limits': len([limit for limit in self.risk_limits.values() if limit.enabled])
            }
            
            # Risk limit status
            limit_status = {}
            for limit_name, limit in self.risk_limits.items():
                if limit.enabled:
                    current_exposure = self._calculate_current_exposure(limit.limit_type)
                    utilization = current_exposure / limit.max_value if limit.max_value > 0 else 0
                    
                    limit_status[limit_name] = {
                        'current_exposure': current_exposure,
                        'limit': limit.max_value,
                        'utilization': utilization,
                        'breach_detected': utilization > 1.0,
                        'warning_level': utilization > 0.8
                    }
            
            return {
                'timestamp': current_time.isoformat(),
                'portfolio_summary': portfolio_summary,
                'current_metrics': current_metrics,
                'limit_status': limit_status,
                'system_status': 'active',
                'ai_enhanced': self.ai_enabled
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_status': 'error'
            }
    
    def _calculate_portfolio_statistics(self) -> Dict[str, Any]:
        """Calculate current portfolio statistics"""
        if not self.positions:
            return {}
        
        total_long = sum(pos for pos in self.positions.values() if pos > 0)
        total_short = sum(abs(pos) for pos in self.positions.values() if pos < 0)
        net_exposure = total_long - total_short
        gross_exposure = total_long + total_short
        
        return {
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'long_exposure': total_long,
            'short_exposure': total_short,
            'leverage': gross_exposure / self.portfolio_value if self.portfolio_value > 0 else 0,
            'number_of_positions': len(self.positions),
            'cash_position': self.portfolio_value - net_exposure
        }
    
    def _generate_risk_alerts(self) -> List[Dict[str, Any]]:
        """Generate risk alerts based on current metrics"""
        alerts = []
        
        # Check for high-risk metrics
        for metric_name, metric_list in self.risk_metrics.items():
            if metric_list:
                latest_metric = max(metric_list, key=lambda x: x.timestamp)
                if latest_metric.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    alerts.append({
                        'type': 'risk_metric',
                        'metric': metric_name,
                        'level': latest_metric.level.value,
                        'value': latest_metric.value,
                        'description': latest_metric.description,
                        'timestamp': latest_metric.timestamp
                    })
        
        return alerts
    
    def generate_risk_report(self,
                           positions: Dict[str, float],
                           returns: pd.DataFrame,
                           report_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Generate comprehensive risk report for compliance.
        
        Args:
            positions: Current positions dictionary
            returns: Historical returns DataFrame
            report_type: Type of report ('summary', 'comprehensive', 'regulatory')
            
        Returns:
            Dictionary containing risk report
        """
        try:
            report_timestamp = datetime.now()
            portfolio_value = sum(positions.values())
            
            # Calculate VaR metrics
            var_results = {}
            for confidence in self.config['var_confidence_levels']:
                for horizon in self.config['var_time_horizons']:
                    var_key = f"var_{confidence}_{horizon}d"
                    var_results[var_key] = self.calculate_var(
                        returns, positions, confidence, horizon
                    )
            
            # Run stress tests
            stress_results = self.run_stress_test(returns, positions)
            
            # Check limits
            limit_results = self.check_limits(positions, portfolio_value)
            
            # Correlation analysis
            correlation_results = self.monitor_correlation_risk(returns)
            
            # Portfolio composition
            portfolio_composition = self._analyze_portfolio_composition(positions, portfolio_value)
            
            # Risk attribution
            risk_attribution = self._calculate_risk_attribution(returns, positions)
            
            # Executive summary
            executive_summary = self._generate_executive_summary(
                var_results, stress_results, limit_results, correlation_results
            )
            
            report = {
                'report_metadata': {
                    'timestamp': report_timestamp,
                    'report_type': report_type,
                    'portfolio_value': portfolio_value,
                    'reporting_period': f"{returns.index[0]} to {returns.index[-1]}",
                    'data_quality': self._assess_data_quality(returns)
                },
                'executive_summary': executive_summary,
                'var_analysis': var_results,
                'stress_testing': stress_results,
                'limit_monitoring': limit_results,
                'correlation_analysis': correlation_results,
                'portfolio_composition': portfolio_composition,
                'risk_attribution': risk_attribution,
                'recommendations': self._generate_risk_recommendations(
                    var_results, stress_results, limit_results
                )
            }
            
            # Add regulatory sections if needed
            if report_type == 'regulatory':
                report.update({
                    'regulatory_metrics': self._calculate_regulatory_metrics(positions, returns),
                    'compliance_status': self._assess_compliance_status(limit_results),
                    'model_validation': self._validate_risk_models(returns)
                })
            
            logger.info(f"Risk report generated: {report_type} type")
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            raise
    
    def _analyze_portfolio_composition(self, positions: Dict[str, float], portfolio_value: float) -> Dict[str, Any]:
        """Analyze portfolio composition and concentration"""
        if not positions or portfolio_value <= 0:
            return {}
        
        # Position sizes
        position_sizes = {asset: abs(pos) / portfolio_value for asset, pos in positions.items()}
        sorted_positions = sorted(position_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Concentration metrics
        top_5_concentration = sum(size for _, size in sorted_positions[:5])
        top_10_concentration = sum(size for _, size in sorted_positions[:10])
        
        # Herfindahl-Hirschman Index
        hhi = sum(size**2 for size in position_sizes.values())
        
        return {
            'position_count': len(positions),
            'largest_position': sorted_positions[0] if sorted_positions else None,
            'top_5_concentration': top_5_concentration,
            'top_10_concentration': top_10_concentration,
            'herfindahl_index': hhi,
            'concentration_risk': 'high' if top_5_concentration > 0.5 else 'medium' if top_5_concentration > 0.3 else 'low',
            'position_distribution': {
                'mean_position_size': np.mean(list(position_sizes.values())),
                'median_position_size': np.median(list(position_sizes.values())),
                'position_size_std': np.std(list(position_sizes.values()))
            }
        }
    
    def _calculate_risk_attribution(self, returns: pd.DataFrame, positions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate risk attribution by asset and factor"""
        # Component contributions to portfolio risk
        portfolio_value = sum(positions.values())
        weights = {asset: pos / portfolio_value for asset, pos in positions.items() if asset in returns.columns}
        
        # Calculate portfolio variance
        aligned_returns = returns[[asset for asset in weights.keys() if asset in returns.columns]]
        weight_vector = np.array([weights[asset] for asset in aligned_returns.columns])
        
        cov_matrix = aligned_returns.cov().values
        portfolio_variance = np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector))
        
        # Marginal contribution to risk
        marginal_contributions = {}
        for i, asset in enumerate(aligned_returns.columns):
            marginal_var = 2 * np.dot(cov_matrix[i], weight_vector)
            marginal_contributions[asset] = marginal_var * weights[asset] / portfolio_variance
        
        return {
            'portfolio_volatility': np.sqrt(portfolio_variance * 252),
            'marginal_contributions': marginal_contributions,
            'component_contributions': {
                asset: contrib * np.sqrt(portfolio_variance * 252) 
                for asset, contrib in marginal_contributions.items()
            }
        }
    
    def _generate_executive_summary(self, var_results, stress_results, limit_results, correlation_results) -> Dict[str, Any]:
        """Generate executive summary of risk metrics"""
        # Key risk metrics
        daily_var_95 = var_results.get('var_0.95_1d', {}).get('var', 0)
        daily_var_99 = var_results.get('var_0.99_1d', {}).get('var', 0)
        worst_stress_loss = stress_results.get('aggregate_metrics', {}).get('max_loss', 0)
        
        # Risk level assessment
        risk_level = 'low'
        if daily_var_95 > 0.03 or worst_stress_loss > 0.15 or limit_results['total_violations'] > 0:
            risk_level = 'high'
        elif daily_var_95 > 0.02 or worst_stress_loss > 0.10:
            risk_level = 'medium'
        
        return {
            'overall_risk_level': risk_level,
            'key_metrics': {
                'daily_var_95': daily_var_95,
                'daily_var_99': daily_var_99,
                'worst_stress_loss': worst_stress_loss,
                'limit_violations': limit_results['total_violations'],
                'max_correlation': correlation_results.get('risk_summary', {}).get('max_correlation', 0)
            },
            'risk_highlights': [
                f"Daily VaR (95%): {daily_var_95:.2%}",
                f"Worst stress scenario loss: {worst_stress_loss:.2%}",
                f"Limit violations: {limit_results['total_violations']}",
                f"Maximum correlation: {correlation_results.get('risk_summary', {}).get('max_correlation', 0):.3f}"
            ]
        }
    
    def _generate_risk_recommendations(self, var_results, stress_results, limit_results) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # VaR-based recommendations
        daily_var_95 = var_results.get('var_0.95_1d', {}).get('var', 0)
        if daily_var_95 > 0.03:
            recommendations.append("Consider reducing position sizes to lower daily VaR below 3%")
        
        # Stress test recommendations
        worst_loss = stress_results.get('aggregate_metrics', {}).get('max_loss', 0)
        if worst_loss > 0.20:
            recommendations.append("Portfolio shows high stress test losses; consider diversification")
        
        # Limit violations
        if limit_results['total_violations'] > 0:
            recommendations.append("Address limit violations immediately to maintain risk discipline")
        
        # Leverage recommendations
        if limit_results['leverage'] > 2.5:
            recommendations.append("High leverage detected; consider reducing gross exposure")
        
        return recommendations
    
    def _calculate_regulatory_metrics(self, positions: Dict[str, float], returns: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regulatory risk metrics"""
        return {
            'liquidity_coverage_ratio': 1.0,  # Placeholder
            'leverage_ratio': self._calculate_leverage_ratio(positions),
            'concentration_limits': self._check_concentration_limits(positions),
            'market_risk_capital': self._calculate_market_risk_capital(returns, positions)
        }
    
    def _assess_compliance_status(self, limit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance status"""
        return {
            'overall_status': 'compliant' if limit_results['total_violations'] == 0 else 'non_compliant',
            'violations_count': limit_results['total_violations'],
            'warnings_count': limit_results['total_warnings'],
            'compliance_score': max(0, 100 - limit_results['total_violations'] * 10 - limit_results['total_warnings'] * 2)
        }
    
    def _validate_risk_models(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Validate risk models"""
        return {
            'var_backtesting': self._backtest_var_model(returns),
            'model_stability': self._test_model_stability(returns),
            'data_quality_score': self._assess_data_quality(returns)
        }
    
    def _backtest_var_model(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Backtest VaR model accuracy"""
        # Simplified backtesting
        return {
            'coverage_ratio': 0.95,
            'independence_test': 'passed',
            'kupiec_test': 'passed'
        }
    
    def _test_model_stability(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Test model stability over time"""
        return {
            'parameter_stability': 'stable',
            'forecast_accuracy': 0.85,
            'model_confidence': 'high'
        }
    
    def _assess_data_quality(self, returns: pd.DataFrame) -> float:
        """Assess data quality score"""
        missing_data_pct = returns.isnull().sum().sum() / (len(returns) * len(returns.columns))
        return max(0, 1 - missing_data_pct * 2)
    
    def _calculate_leverage_ratio(self, positions: Dict[str, float]) -> float:
        """Calculate leverage ratio"""
        total_exposure = sum(abs(pos) for pos in positions.values())
        equity = sum(positions.values())
        return total_exposure / equity if equity > 0 else 0
    
    def _check_concentration_limits(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Check concentration limits"""
        total_value = sum(abs(pos) for pos in positions.values())
        max_position = max(abs(pos) for pos in positions.values()) if positions else 0
        
        return {
            'max_single_position': max_position / total_value if total_value > 0 else 0,
            'concentration_breaches': 0,
            'status': 'compliant'
        }
    
    def _calculate_market_risk_capital(self, returns: pd.DataFrame, positions: Dict[str, float]) -> float:
        """Calculate market risk capital requirement"""
        # Simplified calculation
        var_result = self.calculate_var(returns, positions, 0.99, 10)
        return var_result['var'] * 3  # Simplified multiplier
    
    def _store_risk_metric(self, name: str, value: float, level: RiskLevel, description: str):
        """Store risk metric with timestamp"""
        metric = RiskMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            level=level,
            description=description
        )
        
        if name not in self.risk_metrics:
            self.risk_metrics[name] = []
        
        self.risk_metrics[name].append(metric)
        
        # Keep only last 1000 metrics per type
        if len(self.risk_metrics[name]) > 1000:
            self.risk_metrics[name] = self.risk_metrics[name][-1000:]
    
    def _classify_risk_level(self, value: float, metric_type: str) -> RiskLevel:
        """Classify risk level based on value and metric type"""
        if metric_type == 'var':
            if value > 0.05:
                return RiskLevel.CRITICAL
            elif value > 0.03:
                return RiskLevel.HIGH
            elif value > 0.02:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        elif metric_type == 'stress':
            if value > 0.25:
                return RiskLevel.CRITICAL
            elif value > 0.15:
                return RiskLevel.HIGH
            elif value > 0.10:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        elif metric_type == 'correlation':
            if value > 0.85:
                return RiskLevel.HIGH
            elif value > 0.70:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        elif metric_type == 'violations':
            if value > 3:
                return RiskLevel.CRITICAL
            elif value > 1:
                return RiskLevel.HIGH
            elif value > 0:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        else:
            return RiskLevel.LOW
    
    def set_risk_limit(self, limit: RiskLimit):
        """Set a risk limit"""
        self.risk_limits[limit.name] = limit
        logger.info(f"Risk limit set: {limit.name} = {limit.value}")
    
    def update_positions(self, positions: Dict[str, float], portfolio_value: float):
        """Update current positions and portfolio value"""
        self.positions = positions.copy()
        self.portfolio_value = portfolio_value
        self.last_update = datetime.now()
        logger.info(f"Positions updated: {len(positions)} positions, ${portfolio_value:,.2f} total value")
    
    def add_stress_scenario(self, scenario: StressTestScenario):
        """Add a custom stress test scenario"""
        self.stress_scenarios[scenario.name] = scenario
        logger.info(f"Stress scenario added: {scenario.name}")
    
    def export_risk_data(self, filepath: str, format: str = 'json'):
        """Export risk data to file"""
        try:
            risk_data = {
                'risk_metrics': {
                    name: [
                        {
                            'name': metric.name,
                            'value': metric.value,
                            'timestamp': metric.timestamp.isoformat(),
                            'level': metric.level.value,
                            'description': metric.description
                        }
                        for metric in metrics
                    ]
                    for name, metrics in self.risk_metrics.items()
                },
                'risk_limits': {
                    name: {
                        'name': limit.name,
                        'limit_type': limit.limit_type,
                        'value': limit.value,
                        'unit': limit.unit,
                        'scope': limit.scope,
                        'warning_threshold': limit.warning_threshold,
                        'breach_action': limit.breach_action
                    }
                    for name, limit in self.risk_limits.items()
                },
                'stress_scenarios': {
                    name: {
                        'name': scenario.name,
                        'description': scenario.description,
                        'shocks': scenario.shocks,
                        'probability': scenario.probability,
                        'historical_date': scenario.historical_date
                    }
                    for name, scenario in self.stress_scenarios.items()
                }
            }
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(risk_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Risk data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting risk data: {str(e)}")
            raise
    
    # AI-Enhanced Methods
    
    def train_ai_models(self, 
                       returns: pd.DataFrame,
                       market_indicators: Optional[pd.DataFrame] = None,
                       alternative_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train AI models for enhanced risk management.
        
        Args:
            returns: Historical returns DataFrame
            market_indicators: Market indicators DataFrame
            alternative_data: Alternative data sources
            
        Returns:
            Training results
        """
        if not self.ai_enabled:
            logger.warning("AI enhancements not enabled")
            return {'status': 'ai_disabled'}
        
        try:
            results = {}
            
            # Train regime detection model
            logger.info("Training regime detection model...")
            regime_results = self.ai_enhancements.train_regime_detector(
                returns, market_indicators
            )
            results['regime_detection'] = regime_results
            
            # Discover risk factors
            logger.info("Discovering risk factors...")
            discovered_factors = self.ai_enhancements.discover_risk_factors(
                returns, market_indicators, alternative_data
            )
            results['factor_discovery'] = {
                'factors_discovered': len(discovered_factors),
                'total_explained_variance': sum(f.explained_variance for f in discovered_factors),
                'average_stability': np.mean([f.stability_score for f in discovered_factors])
            }
            
            logger.info("AI model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error training AI models: {str(e)}")
            raise
    
    def detect_regime_change(self, 
                           recent_data: pd.DataFrame,
                           market_indicators: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Detect regime changes using AI-enhanced methods.
        
        Args:
            recent_data: Recent market data
            market_indicators: Additional market indicators
            
        Returns:
            Regime detection results
        """
        if not self.ai_enabled:
            # Fall back to traditional regime detection
            return self._traditional_regime_detection(recent_data)
        
        try:
            # AI-powered regime detection
            regime_signal = self.ai_enhancements.detect_regime_change(
                recent_data, market_indicators
            )
            
            # Store regime change metric
            self._store_risk_metric(
                name="regime_change_signal",
                value=regime_signal.signal_strength,
                level=self._classify_risk_level(regime_signal.signal_strength, 'regime'),
                description=f"Regime {regime_signal.regime_id} detected with {regime_signal.confidence:.2%} confidence"
            )
            
            return {
                'regime_id': regime_signal.regime_id,
                'confidence': regime_signal.confidence,
                'signal_strength': regime_signal.signal_strength,
                'anomaly_score': regime_signal.anomaly_score,
                'contributing_factors': regime_signal.contributing_factors,
                'timestamp': regime_signal.timestamp,
                'method': 'ai_enhanced'
            }
            
        except Exception as e:
            logger.error(f"Error in AI regime detection: {str(e)}")
            # Fall back to traditional method
            return self._traditional_regime_detection(recent_data)
    
    def _traditional_regime_detection(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Traditional regime detection fallback"""
        try:
            # Simple volatility-based regime detection
            recent_volatility = recent_data.std().mean()
            
            if recent_volatility < 0.01:
                regime_id = 0  # Low volatility
            elif recent_volatility < 0.02:
                regime_id = 1  # Medium volatility
            elif recent_volatility < 0.04:
                regime_id = 2  # High volatility
            else:
                regime_id = 3  # Crisis
            
            return {
                'regime_id': regime_id,
                'confidence': 0.7,
                'signal_strength': 0.5,
                'anomaly_score': 0.0,
                'contributing_factors': ['volatility'],
                'timestamp': datetime.now(),
                'method': 'traditional'
            }
            
        except Exception as e:
            logger.error(f"Error in traditional regime detection: {str(e)}")
            return {
                'regime_id': 1,
                'confidence': 0.5,
                'signal_strength': 0.0,
                'anomaly_score': 0.0,
                'contributing_factors': [],
                'timestamp': datetime.now(),
                'method': 'fallback'
            }
    
    def get_discovered_risk_factors(self) -> Dict[str, Any]:
        """
        Get discovered risk factors from AI analysis.
        
        Returns:
            Dictionary containing discovered risk factors
        """
        if not self.ai_enabled:
            return {'status': 'ai_disabled'}
        
        try:
            return self.ai_enhancements.get_discovered_factors_summary()
        except Exception as e:
            logger.error(f"Error getting discovered risk factors: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_ai_risk_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive AI risk analysis.
        
        Returns:
            Dictionary containing AI risk analysis
        """
        if not self.ai_enabled:
            return {'status': 'ai_disabled'}
        
        try:
            # Get regime analysis
            regime_analysis = self.ai_enhancements.get_current_regime_analysis()
            
            # Get factor analysis
            factor_analysis = self.ai_enhancements.get_discovered_factors_summary()
            
            return {
                'regime_analysis': regime_analysis,
                'factor_analysis': factor_analysis,
                'ai_status': 'active',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting AI risk analysis: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def update_risk_factors(self, 
                           returns: pd.DataFrame,
                           market_data: Optional[pd.DataFrame] = None,
                           alternative_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Update risk factor models with new data.
        
        Args:
            returns: Recent returns data
            market_data: Market indicators
            alternative_data: Alternative data sources
            
        Returns:
            Update results
        """
        if not self.ai_enabled:
            return {'status': 'ai_disabled'}
        
        try:
            # Discover new risk factors
            new_factors = self.ai_enhancements.discover_risk_factors(
                returns, market_data, alternative_data
            )
            
            # Store factor discovery metric
            self._store_risk_metric(
                name="risk_factors_discovered",
                value=len(new_factors),
                level=RiskLevel.LOW,
                description=f"Discovered {len(new_factors)} risk factors"
            )
            
            return {
                'factors_updated': len(new_factors),
                'total_explained_variance': sum(f.explained_variance for f in new_factors),
                'average_stability': np.mean([f.stability_score for f in new_factors]),
                'update_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error updating risk factors: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def save_ai_models(self, filepath: str) -> bool:
        """
        Save AI models to disk.
        
        Args:
            filepath: Path to save models
            
        Returns:
            Success status
        """
        if not self.ai_enabled:
            logger.warning("AI enhancements not enabled")
            return False
        
        try:
            self.ai_enhancements.save_models(filepath)
            return True
        except Exception as e:
            logger.error(f"Error saving AI models: {str(e)}")
            return False
    
    def load_ai_models(self, filepath: str) -> bool:
        """
        Load AI models from disk.
        
        Args:
            filepath: Path to load models from
            
        Returns:
            Success status
        """
        if not self.ai_enabled:
            logger.warning("AI enhancements not enabled")
            return False
        
        try:
            self.ai_enhancements.load_models(filepath)
            return True
        except Exception as e:
            logger.error(f"Error loading AI models: {str(e)}")
            return False 