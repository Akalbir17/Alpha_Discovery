"""
Market Regime Detection Agent for Alpha Discovery Platform

This agent uses advanced econometric models to detect market regimes,
structural breaks, and volatility patterns using CrewAI framework.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from pathlib import Path
import pickle
import joblib
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
# Remove direct sklearn imports - now using ML client
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import arch
from arch import arch_model
from hmmlearn import hmm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import ruptures as rpt
from ruptures import Pelt, Binseg
import warnings
warnings.filterwarnings('ignore')

# ML client for remote inference (Phase 4 upgrade)
try:
    from src.scrapers.ml_client import ml_client
    ML_CLIENT_AVAILABLE = True
except ImportError:
    ML_CLIENT_AVAILABLE = False
    logger.warning("ML client not available - using fallback analysis")

# Import our existing components
from src.utils.model_manager import ModelManager
from src.utils.error_handling import handle_errors, AlphaDiscoveryError
from src.utils.monitoring import monitor_performance, track_metrics
from src.utils.config_manager import get_config_section

tool_config = get_config_section('tools')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    SIDEWAYS = "SIDEWAYS"
    CRISIS = "CRISIS"
    RECOVERY = "RECOVERY"
    TRANSITION = "TRANSITION"

class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW_VOL = "LOW_VOL"
    NORMAL_VOL = "NORMAL_VOL"
    HIGH_VOL = "HIGH_VOL"
    CRISIS_VOL = "CRISIS_VOL"

class LiquidityRegime(Enum):
    """Liquidity regime classification"""
    HIGH_LIQUIDITY = "HIGH_LIQUIDITY"
    NORMAL_LIQUIDITY = "NORMAL_LIQUIDITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"

@dataclass
class RegimeState:
    """Current market regime state"""
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    liquidity_regime: LiquidityRegime
    regime_probability: float
    transition_probabilities: Dict[str, float]
    regime_duration: int
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StructuralBreak:
    """Structural break detection result"""
    break_date: datetime
    break_type: str
    test_statistic: float
    p_value: float
    confidence_level: float
    affected_series: List[str]
    regime_change: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VolatilityForecast:
    """Volatility forecast result"""
    symbol: str
    forecast_horizon: int
    volatility_forecast: List[float]
    confidence_intervals: List[Tuple[float, float]]
    model_type: str
    forecast_accuracy: float
    timestamp: datetime

class HiddenMarkovModelTool(BaseTool):
    """Tool for Hidden Markov Model regime detection"""
    
    name: str = "hmm_regime_detector"
    description: str = "Detects market regimes using Hidden Markov Models"
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize after pydantic validation"""
        super().model_post_init(__context)
        # Initialize attributes that aren't pydantic fields
        object.__setattr__(self, 'models', {})
        object.__setattr__(self, 'scalers', {})
    
    @handle_errors
    async def _run(self, returns: pd.Series, n_states: int = 3, model_type: str = "gaussian") -> Dict[str, Any]:
        """Detect market regimes using HMM"""
        try:
            if not ML_CLIENT_AVAILABLE:
                logger.warning("ML client not available - using fallback HMM analysis")
                return self._fallback_hmm_analysis(returns, n_states)
            
            # Prepare data for ML client
            returns_clean = returns.dropna()
            feature_list = returns_clean.values.reshape(-1, 1).tolist()
            
            # Fit HMM model
            if model_type == "gaussian":
                model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", random_state=42)
            elif model_type == "gmm":
                model = hmm.GMMHMM(n_components=n_states, random_state=42)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            model.fit(X_scaled)
            
            # Predict states
            states = model.predict(X_scaled)
            state_probs = model.predict_proba(X_scaled)
            
            # Calculate regime statistics
            regime_stats = {}
            for state in range(n_states):
                state_mask = states == state
                state_returns = returns[state_mask]
                
                regime_stats[f"regime_{state}"] = {
                    "mean_return": float(state_returns.mean()),
                    "volatility": float(state_returns.std()),
                    "duration": int(state_mask.sum()),
                    "frequency": float(state_mask.mean()),
                    "sharpe_ratio": float(state_returns.mean() / state_returns.std()) if state_returns.std() > 0 else 0
                }
            
            # Get transition matrix
            transition_matrix = model.transmat_
            
            # Current state
            current_state = states[-1]
            current_prob = state_probs[-1, current_state]
            
            # Classify regimes based on return characteristics
            regime_classification = self._classify_regimes(regime_stats, n_states)
            
            return {
                "model_type": model_type,
                "n_states": n_states,
                "current_state": int(current_state),
                "current_probability": float(current_prob),
                "states": states.tolist(),
                "state_probabilities": state_probs.tolist(),
                "transition_matrix": transition_matrix.tolist(),
                "regime_stats": regime_stats,
                "regime_classification": regime_classification,
                "model_score": float(model.score(X_scaled)),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}")
            return {"error": str(e)}
    
    def _classify_regimes(self, regime_stats: Dict, n_states: int) -> Dict[str, str]:
        """Classify regimes based on statistical properties"""
        classification = {}
        
        # Sort regimes by mean return
        sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['mean_return'])
        
        if n_states == 2:
            classification[sorted_regimes[0][0]] = "BEAR_MARKET"
            classification[sorted_regimes[1][0]] = "BULL_MARKET"
        elif n_states == 3:
            classification[sorted_regimes[0][0]] = "BEAR_MARKET"
            classification[sorted_regimes[1][0]] = "SIDEWAYS"
            classification[sorted_regimes[2][0]] = "BULL_MARKET"
        elif n_states == 4:
            classification[sorted_regimes[0][0]] = "CRISIS"
            classification[sorted_regimes[1][0]] = "BEAR_MARKET"
            classification[sorted_regimes[2][0]] = "SIDEWAYS"
            classification[sorted_regimes[3][0]] = "BULL_MARKET"
        
        return classification

class StructuralBreakTool(BaseTool):
    """Tool for structural break detection"""
    
    name: str = "structural_break_detector"
    description: str = "Detects structural breaks using CUSUM and Bai-Perron tests"
    
    def __init__(self):
        super().__init__()
    
    @handle_errors
    async def _run(self, data: pd.Series, method: str = "cusum") -> Dict[str, Any]:
        """Detect structural breaks in time series"""
        try:
            if method == "cusum":
                return await self._cusum_test(data)
            elif method == "bai_perron":
                return await self._bai_perron_test(data)
            elif method == "pelt":
                return await self._pelt_test(data)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
        except Exception as e:
            logger.error(f"Structural break detection failed: {e}")
            return {"error": str(e)}
    
    async def _cusum_test(self, data: pd.Series) -> Dict[str, Any]:
        """CUSUM test for structural breaks"""
        try:
            # Calculate CUSUM statistic
            mean_val = data.mean()
            cusum = np.cumsum(data - mean_val)
            
            # Standardize CUSUM
            cusum_std = cusum / data.std()
            
            # Critical values (approximate)
            n = len(data)
            critical_value = 0.948 * np.sqrt(n)  # 5% significance level
            
            # Find breaks
            breaks = []
            for i, val in enumerate(cusum_std):
                if abs(val) > critical_value:
                    breaks.append({
                        "index": i,
                        "date": data.index[i] if hasattr(data, 'index') else i,
                        "cusum_value": float(val),
                        "critical_value": float(critical_value),
                        "significant": True
                    })
            
            return {
                "method": "cusum",
                "cusum_values": cusum_std.tolist(),
                "critical_value": float(critical_value),
                "breaks": breaks,
                "max_cusum": float(np.max(np.abs(cusum_std))),
                "break_detected": len(breaks) > 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"CUSUM test failed: {e}")
            return {"error": str(e)}
    
    async def _bai_perron_test(self, data: pd.Series) -> Dict[str, Any]:
        """Bai-Perron test for multiple structural breaks"""
        try:
            # Use ruptures library for Bai-Perron test
            model = "l2"  # Least squares
            algo = rpt.Pelt(model=model).fit(data.values)
            
            # Detect breaks
            breaks = algo.predict(pen=10)
            
            # Convert to dates if index available
            break_dates = []
            if hasattr(data, 'index'):
                for break_point in breaks[:-1]:  # Exclude last point
                    break_dates.append({
                        "index": break_point,
                        "date": data.index[break_point],
                        "value": float(data.iloc[break_point])
                    })
            
            return {
                "method": "bai_perron",
                "breaks": breaks,
                "break_dates": break_dates,
                "num_breaks": len(breaks) - 1,
                "break_detected": len(breaks) > 1,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Bai-Perron test failed: {e}")
            return {"error": str(e)}
    
    async def _pelt_test(self, data: pd.Series) -> Dict[str, Any]:
        """PELT algorithm for changepoint detection"""
        try:
            # Use ruptures PELT algorithm
            algo = rpt.Pelt(model="rbf").fit(data.values)
            breaks = algo.predict(pen=10)
            
            # Calculate test statistics
            break_info = []
            for i, break_point in enumerate(breaks[:-1]):
                if i == 0:
                    segment_start = 0
                else:
                    segment_start = breaks[i-1]
                
                segment_data = data.iloc[segment_start:break_point]
                
                break_info.append({
                    "break_point": break_point,
                    "date": data.index[break_point] if hasattr(data, 'index') else break_point,
                    "segment_mean": float(segment_data.mean()),
                    "segment_std": float(segment_data.std()),
                    "segment_length": len(segment_data)
                })
            
            return {
                "method": "pelt",
                "breaks": breaks,
                "break_info": break_info,
                "num_breaks": len(breaks) - 1,
                "break_detected": len(breaks) > 1,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"PELT test failed: {e}")
            return {"error": str(e)}

class VolatilityRegimeTool(BaseTool):
    """Tool for volatility regime classification"""
    
    name: str = "volatility_regime_classifier"
    description: str = "Classifies volatility regimes using statistical methods"
    
    def __init__(self):
        super().__init__()
    
    @handle_errors
    async def _run(self, returns: pd.Series, window: int = 22) -> Dict[str, Any]:
        """Classify volatility regimes"""
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            
            # Remove NaN values
            rolling_vol = rolling_vol.dropna()
            
            # Use Gaussian Mixture Model for regime classification
            vol_data = rolling_vol.values.reshape(-1, 1)
            
            # Fit GMM with 4 components (low, normal, high, crisis)
            gmm = GaussianMixture(n_components=4, random_state=42)
            vol_regimes = gmm.fit_predict(vol_data)
            
            # Calculate regime statistics
            regime_stats = {}
            for regime in range(4):
                regime_mask = vol_regimes == regime
                regime_vols = rolling_vol[regime_mask]
                
                regime_stats[f"regime_{regime}"] = {
                    "mean_volatility": float(regime_vols.mean()),
                    "median_volatility": float(regime_vols.median()),
                    "std_volatility": float(regime_vols.std()),
                    "frequency": float(regime_mask.mean()),
                    "duration": int(regime_mask.sum())
                }
            
            # Classify regimes based on volatility levels
            vol_classification = self._classify_vol_regimes(regime_stats)
            
            # Current regime
            current_regime = vol_regimes[-1]
            current_vol = rolling_vol.iloc[-1]
            
            # Calculate transition probabilities
            transition_probs = self._calculate_vol_transitions(vol_regimes)
            
            return {
                "current_regime": int(current_regime),
                "current_volatility": float(current_vol),
                "volatility_regimes": vol_regimes.tolist(),
                "regime_stats": regime_stats,
                "vol_classification": vol_classification,
                "transition_probabilities": transition_probs,
                "rolling_volatility": rolling_vol.tolist(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Volatility regime classification failed: {e}")
            return {"error": str(e)}
    
    def _classify_vol_regimes(self, regime_stats: Dict) -> Dict[str, str]:
        """Classify volatility regimes based on levels"""
        # Sort regimes by mean volatility
        sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['mean_volatility'])
        
        classification = {}
        classification[sorted_regimes[0][0]] = "LOW_VOL"
        classification[sorted_regimes[1][0]] = "NORMAL_VOL"
        classification[sorted_regimes[2][0]] = "HIGH_VOL"
        classification[sorted_regimes[3][0]] = "CRISIS_VOL"
        
        return classification
    
    def _calculate_vol_transitions(self, regimes: np.ndarray) -> Dict[str, float]:
        """Calculate volatility regime transition probabilities"""
        n_regimes = len(np.unique(regimes))
        transitions = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            transitions[current_regime, next_regime] += 1
        
        # Normalize to probabilities
        transition_probs = {}
        for i in range(n_regimes):
            row_sum = transitions[i, :].sum()
            if row_sum > 0:
                for j in range(n_regimes):
                    transition_probs[f"regime_{i}_to_{j}"] = float(transitions[i, j] / row_sum)
        
        return transition_probs

class CorrelationBreakdownTool(BaseTool):
    """Tool for monitoring correlation breakdowns"""
    
    name: str = "correlation_breakdown_monitor"
    description: str = "Monitors correlation breakdowns between assets"
    
    def __init__(self):
        super().__init__()
    
    @handle_errors
    async def _run(self, returns_data: pd.DataFrame, window: int = 60) -> Dict[str, Any]:
        """Monitor correlation breakdowns"""
        try:
            # Calculate rolling correlations
            rolling_corr = returns_data.rolling(window=window).corr()
            
            # Get correlation matrix for each time point
            correlation_breakdowns = []
            
            for date in returns_data.index[window:]:
                corr_matrix = rolling_corr.loc[date]
                
                # Calculate average correlation
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                avg_corr = corr_matrix.values[mask].mean()
                
                # Detect breakdown (correlation drops significantly)
                correlation_breakdowns.append({
                    "date": date,
                    "avg_correlation": float(avg_corr),
                    "correlation_matrix": corr_matrix.to_dict()
                })
            
            # Calculate correlation regime changes
            avg_corrs = [cb["avg_correlation"] for cb in correlation_breakdowns]
            corr_changes = np.diff(avg_corrs)
            
            # Detect significant correlation drops
            breakdown_threshold = -0.1  # 10% drop in correlation
            breakdowns = []
            
            for i, change in enumerate(corr_changes):
                if change < breakdown_threshold:
                    breakdowns.append({
                        "date": correlation_breakdowns[i+1]["date"],
                        "correlation_drop": float(change),
                        "new_correlation": float(avg_corrs[i+1]),
                        "previous_correlation": float(avg_corrs[i])
                    })
            
            # Current correlation state
            current_corr = avg_corrs[-1] if avg_corrs else 0
            
            return {
                "current_avg_correlation": float(current_corr),
                "correlation_breakdowns": breakdowns,
                "num_breakdowns": len(breakdowns),
                "correlation_history": correlation_breakdowns,
                "correlation_volatility": float(np.std(avg_corrs)),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Correlation breakdown monitoring failed: {e}")
            return {"error": str(e)}

class LiquidityRegimeTool(BaseTool):
    """Tool for tracking liquidity regime changes"""
    
    name: str = "liquidity_regime_tracker"
    description: str = "Tracks liquidity regime changes using market microstructure indicators"
    
    def __init__(self):
        super().__init__()
    
    @handle_errors
    async def _run(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict[str, Any]:
        """Track liquidity regime changes"""
        try:
            # Calculate liquidity metrics
            liquidity_metrics = self._calculate_liquidity_metrics(price_data, volume_data)
            
            # Classify liquidity regimes
            liquidity_regimes = self._classify_liquidity_regimes(liquidity_metrics)
            
            # Current liquidity state
            current_regime = liquidity_regimes[-1] if liquidity_regimes else 0
            
            # Calculate regime statistics
            regime_stats = {}
            for regime in range(4):  # 4 liquidity regimes
                regime_mask = np.array(liquidity_regimes) == regime
                if regime_mask.any():
                    regime_data = liquidity_metrics[regime_mask]
                    regime_stats[f"regime_{regime}"] = {
                        "mean_spread": float(regime_data['bid_ask_spread'].mean()),
                        "mean_depth": float(regime_data['market_depth'].mean()),
                        "mean_impact": float(regime_data['price_impact'].mean()),
                        "frequency": float(regime_mask.mean())
                    }
            
            # Classify regimes
            liquidity_classification = self._classify_liquidity_regimes_labels(regime_stats)
            
            return {
                "current_regime": int(current_regime),
                "liquidity_regimes": liquidity_regimes,
                "regime_stats": regime_stats,
                "liquidity_classification": liquidity_classification,
                "liquidity_metrics": liquidity_metrics.to_dict('records'),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Liquidity regime tracking failed: {e}")
            return {"error": str(e)}
    
    def _calculate_liquidity_metrics(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity metrics"""
        metrics = pd.DataFrame(index=price_data.index)
        
        # Bid-ask spread proxy (using high-low spread)
        metrics['bid_ask_spread'] = (price_data['High'] - price_data['Low']) / price_data['Close']
        
        # Market depth proxy (using volume)
        metrics['market_depth'] = volume_data['Volume'] / volume_data['Volume'].rolling(20).mean()
        
        # Price impact proxy (using return-to-volume ratio)
        returns = price_data['Close'].pct_change()
        metrics['price_impact'] = abs(returns) / (volume_data['Volume'] / volume_data['Volume'].rolling(20).mean())
        
        # Amihud illiquidity measure
        metrics['amihud_illiquidity'] = abs(returns) / (volume_data['Volume'] * price_data['Close'])
        
        return metrics.dropna()
    
    def _classify_liquidity_regimes(self, liquidity_metrics: pd.DataFrame) -> List[int]:
        """Classify liquidity regimes using clustering"""
        # Use KMeans clustering for regime classification
        features = ['bid_ask_spread', 'market_depth', 'price_impact', 'amihud_illiquidity']
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(liquidity_metrics[features])
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        regimes = kmeans.fit_predict(scaled_features)
        
        return regimes.tolist()
    
    def _classify_liquidity_regimes_labels(self, regime_stats: Dict) -> Dict[str, str]:
        """Assign labels to liquidity regimes"""
        # Sort regimes by liquidity quality (lower spread = better liquidity)
        sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['mean_spread'])
        
        classification = {}
        if len(sorted_regimes) >= 4:
            classification[sorted_regimes[0][0]] = "HIGH_LIQUIDITY"
            classification[sorted_regimes[1][0]] = "NORMAL_LIQUIDITY"
            classification[sorted_regimes[2][0]] = "LOW_LIQUIDITY"
            classification[sorted_regimes[3][0]] = "LIQUIDITY_CRISIS"
        
        return classification

class GARCHForecastTool(BaseTool):
    """Tool for GARCH volatility forecasting"""
    
    name: str = "garch_volatility_forecaster"
    description: str = "Forecasts volatility using GARCH models"
    
    def __init__(self):
        super().__init__()
    
    @handle_errors
    async def _run(self, returns: pd.Series, model_type: str = "GARCH", horizon: int = 5) -> Dict[str, Any]:
        """Forecast volatility using GARCH models"""
        try:
            # Prepare data
            returns_clean = returns.dropna() * 100  # Convert to percentage
            
            # Fit GARCH model
            if model_type == "GARCH":
                model = arch_model(returns_clean, vol='Garch', p=1, q=1)
            elif model_type == "EGARCH":
                model = arch_model(returns_clean, vol='EGARCH', p=1, q=1)
            elif model_type == "GJR-GARCH":
                model = arch_model(returns_clean, vol='GARCH', p=1, o=1, q=1)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Fit model
            fitted_model = model.fit(disp='off')
            
            # Generate forecasts
            forecasts = fitted_model.forecast(horizon=horizon)
            
            # Extract forecast values
            vol_forecast = np.sqrt(forecasts.variance.values[-1, :]) / 100  # Convert back to decimal
            
            # Calculate confidence intervals (approximate)
            std_error = vol_forecast * 0.1  # Simplified standard error
            conf_intervals = [(vol - 1.96 * std_error[i], vol + 1.96 * std_error[i]) 
                            for i, vol in enumerate(vol_forecast)]
            
            # Model diagnostics
            residuals = fitted_model.resid
            standardized_residuals = residuals / fitted_model.conditional_volatility
            
            # Ljung-Box test for residual autocorrelation
            lb_stat, lb_pvalue = acorr_ljungbox(standardized_residuals, lags=10, return_df=False)
            
            return {
                "model_type": model_type,
                "horizon": horizon,
                "volatility_forecast": vol_forecast.tolist(),
                "confidence_intervals": conf_intervals,
                "current_volatility": float(fitted_model.conditional_volatility.iloc[-1] / 100),
                "model_aic": float(fitted_model.aic),
                "model_bic": float(fitted_model.bic),
                "ljung_box_stat": float(lb_stat),
                "ljung_box_pvalue": float(lb_pvalue),
                "model_params": fitted_model.params.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"GARCH forecasting failed: {e}")
            return {"error": str(e)}

class RegimeDetectionAgent:
    """
    Advanced Market Regime Detection Agent using CrewAI
    
    Uses Hidden Markov Models, structural break detection, and volatility
    regime classification to identify market regimes and forecast transitions.
    """
    
    def __init__(self):
        self.model_manager = ModelManager()
        
        # Initialize tools
        self.hmm_tool = HiddenMarkovModelTool()
        self.break_tool = StructuralBreakTool()
        self.vol_tool = VolatilityRegimeTool()
        self.corr_tool = CorrelationBreakdownTool()
        self.liquidity_tool = LiquidityRegimeTool()
        self.garch_tool = GARCHForecastTool()
        
        # Initialize CrewAI agents
        self._setup_crew()
        
        # Model storage
        self.models = {}
        self.regime_history = []
        
        logger.info("RegimeDetectionAgent initialized successfully")
    
    def _setup_crew(self):
        """Setup CrewAI agents and crew"""
        
        # Regime Detection Agent
        self.regime_detector = Agent(
            role='Market Regime Detector',
            goal='Detect market regimes using advanced econometric models',
            backstory="""You are an expert econometrician specializing in market regime 
            detection. You have deep knowledge of Hidden Markov Models, structural break 
            tests, and volatility modeling. You excel at identifying regime changes and 
            forecasting market transitions.""",
            verbose=True,
            allow_delegation=False,
            tools=[
                self.hmm_tool,
                self.break_tool,
                self.vol_tool
            ]
        )
        
        # Volatility Analyst
        self.volatility_analyst = Agent(
            role='Volatility Regime Analyst',
            goal='Analyze volatility patterns and forecast future volatility',
            backstory="""You are a volatility modeling expert with extensive experience 
            in GARCH models and volatility forecasting. You specialize in identifying 
            volatility regimes and predicting volatility clustering patterns.""",
            verbose=True,
            allow_delegation=False,
            tools=[
                self.vol_tool,
                self.garch_tool
            ]
        )
        
        # Market Structure Analyst
        self.structure_analyst = Agent(
            role='Market Structure Analyst',
            goal='Monitor market microstructure and liquidity conditions',
            backstory="""You are a market microstructure expert who monitors correlation 
            breakdowns, liquidity regimes, and market stress indicators. You excel at 
            detecting structural changes in market behavior.""",
            verbose=True,
            allow_delegation=False,
            tools=[
                self.corr_tool,
                self.liquidity_tool
            ]
        )
        
        # Setup crew
        self.crew = Crew(
            agents=[self.regime_detector, self.volatility_analyst, self.structure_analyst],
            verbose=True,
            process=Process.sequential
        )
    
    @handle_errors
    @monitor_performance
    async def detect_current_regime(self, symbols: List[str], lookback_days: int = 252) -> RegimeState:
        """
        Detect current market regime using multiple models
        
        Args:
            symbols: List of symbols to analyze
            lookback_days: Number of days to look back for analysis
            
        Returns:
            Current regime state with transition probabilities
        """
        try:
            logger.info(f"Detecting current regime for {len(symbols)} symbols")
            
            # Get market data
            market_data = await self._get_market_data(symbols, lookback_days)
            
            # Run regime detection models in parallel
            detection_tasks = [
                self._run_hmm_analysis(market_data),
                self._run_volatility_analysis(market_data),
                self._run_correlation_analysis(market_data),
                self._run_liquidity_analysis(market_data)
            ]
            
            results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            
            # Combine results
            regime_state = await self._combine_regime_signals(results, symbols)
            
            # Store in history
            self.regime_history.append(regime_state)
            
            logger.info(f"Detected regime: {regime_state.market_regime.value}")
            return regime_state
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            raise AlphaDiscoveryError(f"Failed to detect regime: {e}")
    
    @handle_errors
    async def detect_structural_breaks(self, symbol: str, method: str = "cusum") -> List[StructuralBreak]:
        """
        Detect structural breaks in time series
        
        Args:
            symbol: Symbol to analyze
            method: Detection method (cusum, bai_perron, pelt)
            
        Returns:
            List of detected structural breaks
        """
        try:
            logger.info(f"Detecting structural breaks for {symbol} using {method}")
            
            # Get price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y")
            
            if hist.empty:
                raise AlphaDiscoveryError(f"No data available for {symbol}")
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            # Detect breaks
            break_results = await self.break_tool._run(returns, method=method)
            
            if "error" in break_results:
                raise AlphaDiscoveryError(f"Break detection failed: {break_results['error']}")
            
            # Convert to StructuralBreak objects
            structural_breaks = []
            
            if method == "cusum":
                for break_info in break_results.get("breaks", []):
                    structural_breaks.append(StructuralBreak(
                        break_date=break_info["date"],
                        break_type="cusum",
                        test_statistic=break_info["cusum_value"],
                        p_value=0.05,  # Approximate
                        confidence_level=0.95,
                        affected_series=[symbol],
                        regime_change=break_info["significant"],
                        metadata=break_info
                    ))
            
            elif method in ["bai_perron", "pelt"]:
                for break_info in break_results.get("break_info", []):
                    structural_breaks.append(StructuralBreak(
                        break_date=break_info["date"],
                        break_type=method,
                        test_statistic=0.0,  # Not available in this implementation
                        p_value=0.05,
                        confidence_level=0.95,
                        affected_series=[symbol],
                        regime_change=True,
                        metadata=break_info
                    ))
            
            logger.info(f"Detected {len(structural_breaks)} structural breaks")
            return structural_breaks
            
        except Exception as e:
            logger.error(f"Structural break detection failed: {e}")
            raise AlphaDiscoveryError(f"Failed to detect structural breaks: {e}")
    
    @handle_errors
    async def classify_volatility_regime(self, symbol: str) -> VolatilityRegime:
        """
        Classify current volatility regime
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Current volatility regime
        """
        try:
            logger.info(f"Classifying volatility regime for {symbol}")
            
            # Get price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                raise AlphaDiscoveryError(f"No data available for {symbol}")
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            # Classify volatility regime
            vol_results = await self.vol_tool._run(returns)
            
            if "error" in vol_results:
                raise AlphaDiscoveryError(f"Volatility classification failed: {vol_results['error']}")
            
            # Map to volatility regime
            current_regime = vol_results["current_regime"]
            vol_classification = vol_results["vol_classification"]
            
            regime_mapping = {
                "LOW_VOL": VolatilityRegime.LOW_VOL,
                "NORMAL_VOL": VolatilityRegime.NORMAL_VOL,
                "HIGH_VOL": VolatilityRegime.HIGH_VOL,
                "CRISIS_VOL": VolatilityRegime.CRISIS_VOL
            }
            
            regime_label = vol_classification.get(f"regime_{current_regime}", "NORMAL_VOL")
            volatility_regime = regime_mapping.get(regime_label, VolatilityRegime.NORMAL_VOL)
            
            logger.info(f"Volatility regime: {volatility_regime.value}")
            return volatility_regime
            
        except Exception as e:
            logger.error(f"Volatility regime classification failed: {e}")
            raise AlphaDiscoveryError(f"Failed to classify volatility regime: {e}")
    
    @handle_errors
    async def forecast_volatility(self, symbol: str, horizon: int = 5, model_type: str = "GARCH") -> VolatilityForecast:
        """
        Forecast volatility using GARCH models
        
        Args:
            symbol: Symbol to analyze
            horizon: Forecast horizon in days
            model_type: GARCH model type
            
        Returns:
            Volatility forecast
        """
        try:
            logger.info(f"Forecasting volatility for {symbol} using {model_type}")
            
            # Get price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                raise AlphaDiscoveryError(f"No data available for {symbol}")
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            # Generate forecast
            forecast_results = await self.garch_tool._run(returns, model_type=model_type, horizon=horizon)
            
            if "error" in forecast_results:
                raise AlphaDiscoveryError(f"Volatility forecasting failed: {forecast_results['error']}")
            
            # Create forecast object
            volatility_forecast = VolatilityForecast(
                symbol=symbol,
                forecast_horizon=horizon,
                volatility_forecast=forecast_results["volatility_forecast"],
                confidence_intervals=forecast_results["confidence_intervals"],
                model_type=model_type,
                forecast_accuracy=1.0 - forecast_results.get("ljung_box_pvalue", 0.5),  # Simplified accuracy
                timestamp=datetime.now()
            )
            
            logger.info(f"Generated volatility forecast for {horizon} days")
            return volatility_forecast
            
        except Exception as e:
            logger.error(f"Volatility forecasting failed: {e}")
            raise AlphaDiscoveryError(f"Failed to forecast volatility: {e}")
    
    @handle_errors
    async def get_regime_transition_probabilities(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get regime transition probabilities
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary of transition probabilities
        """
        try:
            logger.info(f"Calculating transition probabilities for {len(symbols)} symbols")
            
            # Get market data
            market_data = await self._get_market_data(symbols, 252)
            
            # Run HMM analysis
            hmm_results = await self._run_hmm_analysis(market_data)
            
            if "error" in hmm_results:
                raise AlphaDiscoveryError(f"HMM analysis failed: {hmm_results['error']}")
            
            # Extract transition probabilities
            transition_matrix = hmm_results["transition_matrix"]
            current_state = hmm_results["current_state"]
            
            # Convert to named probabilities
            transition_probs = {}
            regime_names = ["bear", "sideways", "bull"]  # Assuming 3-state model
            
            for i, from_regime in enumerate(regime_names):
                for j, to_regime in enumerate(regime_names):
                    if i < len(transition_matrix) and j < len(transition_matrix[i]):
                        transition_probs[f"{from_regime}_to_{to_regime}"] = float(transition_matrix[i][j])
            
            # Add current state probabilities
            if current_state < len(regime_names):
                transition_probs["current_regime"] = regime_names[current_state]
                for i, regime in enumerate(regime_names):
                    if i < len(transition_matrix[current_state]):
                        transition_probs[f"next_{regime}_prob"] = float(transition_matrix[current_state][i])
            
            logger.info("Calculated transition probabilities")
            return transition_probs
            
        except Exception as e:
            logger.error(f"Transition probability calculation failed: {e}")
            raise AlphaDiscoveryError(f"Failed to calculate transition probabilities: {e}")
    
    async def _get_market_data(self, symbols: List[str], lookback_days: int) -> Dict[str, pd.DataFrame]:
        """Get market data for analysis"""
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{lookback_days}d")
                
                if not hist.empty:
                    market_data[symbol] = hist
                    
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
        
        return market_data
    
    async def _run_hmm_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run HMM analysis on market data"""
        try:
            # Combine returns from all symbols
            all_returns = []
            for symbol, data in market_data.items():
                returns = data['Close'].pct_change().dropna()
                all_returns.extend(returns.tolist())
            
            if not all_returns:
                return {"error": "No return data available"}
            
            # Create returns series
            returns_series = pd.Series(all_returns)
            
            # Run HMM analysis
            return await self.hmm_tool._run(returns_series, n_states=3)
            
        except Exception as e:
            logger.error(f"HMM analysis failed: {e}")
            return {"error": str(e)}
    
    async def _run_volatility_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run volatility analysis"""
        try:
            # Use first symbol for volatility analysis
            if not market_data:
                return {"error": "No market data available"}
            
            symbol, data = next(iter(market_data.items()))
            returns = data['Close'].pct_change().dropna()
            
            return await self.vol_tool._run(returns)
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {"error": str(e)}
    
    async def _run_correlation_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run correlation analysis"""
        try:
            if len(market_data) < 2:
                return {"error": "Need at least 2 symbols for correlation analysis"}
            
            # Create returns dataframe
            returns_df = pd.DataFrame()
            for symbol, data in market_data.items():
                returns_df[symbol] = data['Close'].pct_change()
            
            returns_df = returns_df.dropna()
            
            return await self.corr_tool._run(returns_df)
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {"error": str(e)}
    
    async def _run_liquidity_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run liquidity analysis"""
        try:
            if not market_data:
                return {"error": "No market data available"}
            
            # Use first symbol for liquidity analysis
            symbol, data = next(iter(market_data.items()))
            
            # Create volume dataframe
            volume_df = pd.DataFrame()
            volume_df['Volume'] = data['Volume']
            
            return await self.liquidity_tool._run(data, volume_df)
            
        except Exception as e:
            logger.error(f"Liquidity analysis failed: {e}")
            return {"error": str(e)}
    
    async def _combine_regime_signals(self, results: List[Any], symbols: List[str]) -> RegimeState:
        """Combine signals from different models"""
        try:
            hmm_result, vol_result, corr_result, liquidity_result = results
            
            # Default values
            market_regime = MarketRegime.SIDEWAYS
            volatility_regime = VolatilityRegime.NORMAL_VOL
            liquidity_regime = LiquidityRegime.NORMAL_LIQUIDITY
            confidence = 0.5
            transition_probs = {}
            
            # Process HMM results
            if isinstance(hmm_result, dict) and "error" not in hmm_result:
                regime_classification = hmm_result.get("regime_classification", {})
                current_state = hmm_result.get("current_state", 0)
                
                regime_label = regime_classification.get(f"regime_{current_state}", "SIDEWAYS")
                market_regime = MarketRegime(regime_label)
                confidence = hmm_result.get("current_probability", 0.5)
                
                # Extract transition probabilities
                transition_matrix = hmm_result.get("transition_matrix", [])
                if transition_matrix and current_state < len(transition_matrix):
                    transition_probs = {
                        f"regime_{i}": float(prob) 
                        for i, prob in enumerate(transition_matrix[current_state])
                    }
            
            # Process volatility results
            if isinstance(vol_result, dict) and "error" not in vol_result:
                vol_classification = vol_result.get("vol_classification", {})
                current_vol_regime = vol_result.get("current_regime", 0)
                
                vol_label = vol_classification.get(f"regime_{current_vol_regime}", "NORMAL_VOL")
                volatility_regime = VolatilityRegime(vol_label)
            
            # Process liquidity results
            if isinstance(liquidity_result, dict) and "error" not in liquidity_result:
                liq_classification = liquidity_result.get("liquidity_classification", {})
                current_liq_regime = liquidity_result.get("current_regime", 0)
                
                liq_label = liq_classification.get(f"regime_{current_liq_regime}", "NORMAL_LIQUIDITY")
                liquidity_regime = LiquidityRegime(liq_label)
            
            # Calculate regime duration (simplified)
            regime_duration = len(self.regime_history) if self.regime_history else 1
            
            return RegimeState(
                market_regime=market_regime,
                volatility_regime=volatility_regime,
                liquidity_regime=liquidity_regime,
                regime_probability=confidence,
                transition_probabilities=transition_probs,
                regime_duration=regime_duration,
                confidence_score=confidence,
                timestamp=datetime.now(),
                metadata={
                    "symbols": symbols,
                    "hmm_result": hmm_result if isinstance(hmm_result, dict) else None,
                    "vol_result": vol_result if isinstance(vol_result, dict) else None,
                    "corr_result": corr_result if isinstance(corr_result, dict) else None,
                    "liquidity_result": liquidity_result if isinstance(liquidity_result, dict) else None
                }
            )
            
        except Exception as e:
            logger.error(f"Signal combination failed: {e}")
            # Return default regime state
            return RegimeState(
                market_regime=MarketRegime.SIDEWAYS,
                volatility_regime=VolatilityRegime.NORMAL_VOL,
                liquidity_regime=LiquidityRegime.NORMAL_LIQUIDITY,
                regime_probability=0.5,
                transition_probabilities={},
                regime_duration=1,
                confidence_score=0.5,
                timestamp=datetime.now()
            )

# Example usage and testing
async def main():
    """Example usage of RegimeDetectionAgent"""
    
    # Initialize agent
    agent = RegimeDetectionAgent()
    
    # Test symbols
    symbols = ["SPY", "QQQ", "IWM"]
    
    try:
        # Detect current regime
        print("Detecting current market regime...")
        regime_state = await agent.detect_current_regime(symbols)
        
        print(f"\nCurrent Market Regime:")
        print(f"- Market: {regime_state.market_regime.value}")
        print(f"- Volatility: {regime_state.volatility_regime.value}")
        print(f"- Liquidity: {regime_state.liquidity_regime.value}")
        print(f"- Confidence: {regime_state.confidence_score:.2f}")
        print(f"- Duration: {regime_state.regime_duration} periods")
        
        # Detect structural breaks
        print("\nDetecting structural breaks...")
        breaks = await agent.detect_structural_breaks("SPY", method="cusum")
        print(f"Detected {len(breaks)} structural breaks")
        
        # Classify volatility regime
        print("\nClassifying volatility regime...")
        vol_regime = await agent.classify_volatility_regime("SPY")
        print(f"Volatility regime: {vol_regime.value}")
        
        # Forecast volatility
        print("\nForecasting volatility...")
        vol_forecast = await agent.forecast_volatility("SPY", horizon=5)
        print(f"5-day volatility forecast: {vol_forecast.volatility_forecast}")
        
        # Get transition probabilities
        print("\nCalculating transition probabilities...")
        transition_probs = await agent.get_regime_transition_probabilities(symbols)
        print(f"Transition probabilities: {transition_probs}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 