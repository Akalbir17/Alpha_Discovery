"""
Performance Analytics System for Alpha Discovery Platform

This module provides comprehensive performance analytics including:
- Real-time strategy performance tracking
- Alpha decomposition by source and factor
- Factor attribution analysis with Fama-French and custom factors
- Risk-adjusted return calculations
- Strategy correlation monitoring
- Performance degradation detection
- Daily performance reporting with detailed analytics

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

# Statistical analysis
from scipy import stats
from scipy.optimize import minimize
# ML models offloaded to MCP server (Phase 5.1)
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.metrics import r2_score

# ML Client integration for offloaded models
try:
    from ..scrapers.ml_client import ml_client, ML_CLIENT_AVAILABLE
except ImportError:
    ML_CLIENT_AVAILABLE = False
    ml_client = None

# Time series analysis
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Financial metrics
import empyrical as ep
import pyfolio as pf
import quantstats as qs

# Data handling
import asyncio
import json
from pathlib import Path
import sqlite3

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Performance metric types"""
    TOTAL_RETURN = "total_return"
    ALPHA = "alpha"
    BETA = "beta"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    TRACKING_ERROR = "tracking_error"
    HIT_RATE = "hit_rate"
    PROFIT_FACTOR = "profit_factor"

class AttributionMethod(Enum):
    """Attribution analysis methods"""
    FAMA_FRENCH = "fama_french"
    CARHART = "carhart"
    CUSTOM_FACTORS = "custom_factors"
    BARRA = "barra"
    RISK_MODEL = "risk_model"

@dataclass
class PerformanceRecord:
    """Performance record data structure"""
    timestamp: datetime
    strategy_id: str
    pnl: float
    returns: float
    cumulative_returns: float
    benchmark_returns: float
    alpha: float
    beta: float
    sharpe_ratio: float
    drawdown: float
    volatility: float
    positions: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttributionResult:
    """Factor attribution result"""
    factor_name: str
    exposure: float
    contribution: float
    t_statistic: float
    p_value: float
    r_squared: float
    explanation: str

@dataclass
class AlphaSource:
    """Alpha source decomposition"""
    source_name: str
    contribution: float
    confidence: float
    persistence: float
    description: str

class PerformanceAnalytics:
    """
    Comprehensive performance analytics system for the Alpha Discovery platform.
    
    This class provides institutional-grade performance analytics including:
    - Real-time strategy performance tracking with millisecond precision
    - Alpha decomposition by source, factor, and time period
    - Factor attribution analysis using multiple methodologies
    - Risk-adjusted return calculations with multiple metrics
    - Strategy correlation monitoring and regime detection
    - Performance degradation detection with early warning systems
    - Daily performance reporting with detailed analytics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Performance Analytics system.
        
        Args:
            config: Configuration dictionary with analytics parameters
        """
        self.config = config or self._default_config()
        
        # Performance data storage
        self.performance_records: Dict[str, List[PerformanceRecord]] = {}
        self.benchmark_data: Dict[str, pd.Series] = {}
        self.factor_data: Dict[str, pd.DataFrame] = {}
        
        # Attribution models
        self.attribution_models: Dict[str, Any] = {}
        self.alpha_sources: Dict[str, List[AlphaSource]] = {}
        
        # Performance monitoring
        self.performance_alerts: List[Dict[str, Any]] = []
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
        
        # Database connection for persistence
        self.db_path = self.config.get('database_path', 'performance_analytics.db')
        self._initialize_database()
        
        # Load factor data
        self._load_factor_data()
        
        logger.info("PerformanceAnalytics initialized with comprehensive tracking")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for performance analytics"""
        return {
            'risk_free_rate': 0.02,
            'benchmark_symbol': 'SPY',
            'attribution_window': 252,  # trading days
            'performance_window': 60,   # days for performance monitoring
            'correlation_threshold': 0.7,
            'degradation_threshold': -0.05,  # 5% performance degradation
            'min_observations': 30,
            'confidence_level': 0.95,
            'factor_models': ['fama_french', 'carhart', 'custom'],
            'update_frequency': 'daily',
            'database_path': 'performance_analytics.db',
            'report_formats': ['html', 'pdf', 'json'],
            'real_time_tracking': True,
            'performance_metrics': [
                'total_return', 'alpha', 'beta', 'sharpe_ratio',
                'sortino_ratio', 'maximum_drawdown', 'volatility',
                'information_ratio', 'hit_rate'
            ]
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for performance data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Performance records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_id TEXT NOT NULL,
                        pnl REAL NOT NULL,
                        returns REAL NOT NULL,
                        cumulative_returns REAL NOT NULL,
                        benchmark_returns REAL NOT NULL,
                        alpha REAL,
                        beta REAL,
                        sharpe_ratio REAL,
                        drawdown REAL,
                        volatility REAL,
                        positions TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Attribution results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attribution_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        factor_name TEXT NOT NULL,
                        exposure REAL NOT NULL,
                        contribution REAL NOT NULL,
                        t_statistic REAL,
                        p_value REAL,
                        r_squared REAL,
                        analysis_date TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Performance alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Performance analytics database initialized")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _load_factor_data(self):
        """Load factor data for attribution analysis"""
        try:
            # Load Fama-French factors (would typically come from data provider)
            # For now, we'll create placeholder factor data
            dates = pd.date_range(start='2020-01-01', end='2025-01-01', freq='D')
            
            # Fama-French 3-factor model
            ff_factors = pd.DataFrame({
                'Mkt-RF': np.random.normal(0.0008, 0.012, len(dates)),
                'SMB': np.random.normal(0.0002, 0.008, len(dates)),
                'HML': np.random.normal(0.0001, 0.007, len(dates)),
                'RF': np.full(len(dates), self.config['risk_free_rate'] / 252)
            }, index=dates)
            
            # Carhart 4-factor model (add momentum)
            carhart_factors = ff_factors.copy()
            carhart_factors['MOM'] = np.random.normal(0.0003, 0.009, len(dates))
            
            # Custom factors
            custom_factors = pd.DataFrame({
                'Quality': np.random.normal(0.0002, 0.006, len(dates)),
                'Low_Vol': np.random.normal(0.0001, 0.005, len(dates)),
                'Profitability': np.random.normal(0.0003, 0.007, len(dates)),
                'Investment': np.random.normal(-0.0001, 0.006, len(dates))
            }, index=dates)
            
            self.factor_data = {
                'fama_french': ff_factors,
                'carhart': carhart_factors,
                'custom': custom_factors
            }
            
            logger.info("Factor data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading factor data: {str(e)}")
            # Create minimal factor data as fallback
            self.factor_data = {'fama_french': pd.DataFrame()}
    
    def track_performance(self, 
                         strategy_id: str,
                         returns: pd.Series,
                         positions: Dict[str, float],
                         benchmark_returns: Optional[pd.Series] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> PerformanceRecord:
        """
        Track strategy performance in real-time.
        
        Args:
            strategy_id: Unique strategy identifier
            returns: Strategy returns series
            positions: Current positions dictionary
            benchmark_returns: Benchmark returns for comparison
            metadata: Additional metadata
            
        Returns:
            Performance record with calculated metrics
        """
        try:
            current_time = datetime.now()
            
            # Calculate basic metrics
            latest_return = returns.iloc[-1] if len(returns) > 0 else 0.0
            cumulative_returns = (1 + returns).cumprod().iloc[-1] - 1 if len(returns) > 0 else 0.0
            
            # Calculate PnL (assuming position values)
            pnl = sum(positions.values()) * latest_return if positions else 0.0
            
            # Benchmark comparison
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                benchmark_return = benchmark_returns.iloc[-1]
                alpha, beta = self._calculate_alpha_beta(returns, benchmark_returns)
            else:
                benchmark_return = 0.0
                alpha, beta = 0.0, 1.0
            
            # Risk metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            drawdown = self._calculate_drawdown(returns)
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
            
            # Create performance record
            record = PerformanceRecord(
                timestamp=current_time,
                strategy_id=strategy_id,
                pnl=pnl,
                returns=latest_return,
                cumulative_returns=cumulative_returns,
                benchmark_returns=benchmark_return,
                alpha=alpha,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                drawdown=drawdown,
                volatility=volatility,
                positions=positions.copy(),
                metadata=metadata or {}
            )
            
            # Store record
            if strategy_id not in self.performance_records:
                self.performance_records[strategy_id] = []
            
            self.performance_records[strategy_id].append(record)
            
            # Keep only recent records in memory
            max_records = 10000
            if len(self.performance_records[strategy_id]) > max_records:
                self.performance_records[strategy_id] = self.performance_records[strategy_id][-max_records:]
            
            # Save to database
            self._save_performance_record(record)
            
            # Check for performance alerts
            self._check_performance_alerts(strategy_id, record)
            
            self.last_update = current_time
            
            logger.info(f"Performance tracked for {strategy_id}: Return={latest_return:.4f}, Alpha={alpha:.4f}")
            return record
            
        except Exception as e:
            logger.error(f"Error tracking performance: {str(e)}")
            raise
    
    def calculate_alpha(self, 
                       strategy_returns: pd.Series,
                       benchmark_returns: pd.Series,
                       method: str = 'capm',
                       factor_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate alpha using various methodologies.
        
        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series
            method: Alpha calculation method ('capm', 'fama_french', 'carhart')
            factor_data: Factor data for multi-factor models
            
        Returns:
            Dictionary containing alpha analysis results
        """
        try:
            # Align data
            aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
            if len(aligned_data) < self.config['min_observations']:
                raise ValueError(f"Insufficient data: need at least {self.config['min_observations']} observations")
            
            strategy_ret = aligned_data.iloc[:, 0]
            benchmark_ret = aligned_data.iloc[:, 1]
            
            results = {}
            
            if method == 'capm':
                # CAPM Alpha
                excess_strategy = strategy_ret - self.config['risk_free_rate'] / 252
                excess_benchmark = benchmark_ret - self.config['risk_free_rate'] / 252
                
                # Regression: R_s - R_f = α + β(R_m - R_f) + ε
                X = add_constant(excess_benchmark)
                model = OLS(excess_strategy, X).fit()
                
                alpha = model.params[0] * 252  # Annualized
                beta = model.params[1]
                
                results = {
                    'alpha': alpha,
                    'beta': beta,
                    'alpha_tstat': model.tvalues[0],
                    'alpha_pvalue': model.pvalues[0],
                    'r_squared': model.rsquared,
                    'tracking_error': np.sqrt(model.mse_resid * 252),
                    'information_ratio': alpha / np.sqrt(model.mse_resid * 252) if model.mse_resid > 0 else 0
                }
                
            elif method == 'fama_french' and 'fama_french' in self.factor_data:
                # Fama-French 3-factor model
                ff_data = self.factor_data['fama_french']
                
                # Align with strategy returns
                common_dates = strategy_ret.index.intersection(ff_data.index)
                if len(common_dates) < self.config['min_observations']:
                    raise ValueError("Insufficient overlapping data with Fama-French factors")
                
                strategy_excess = strategy_ret.loc[common_dates] - ff_data.loc[common_dates, 'RF']
                
                # Regression: R_s - R_f = α + β₁(R_m - R_f) + β₂SMB + β₃HML + ε
                X = ff_data.loc[common_dates, ['Mkt-RF', 'SMB', 'HML']]
                X = add_constant(X)
                model = OLS(strategy_excess, X).fit()
                
                results = {
                    'alpha': model.params[0] * 252,
                    'market_beta': model.params[1],
                    'smb_beta': model.params[2],
                    'hml_beta': model.params[3],
                    'alpha_tstat': model.tvalues[0],
                    'alpha_pvalue': model.pvalues[0],
                    'r_squared': model.rsquared,
                    'tracking_error': np.sqrt(model.mse_resid * 252)
                }
                
            elif method == 'carhart' and 'carhart' in self.factor_data:
                # Carhart 4-factor model
                carhart_data = self.factor_data['carhart']
                
                common_dates = strategy_ret.index.intersection(carhart_data.index)
                if len(common_dates) < self.config['min_observations']:
                    raise ValueError("Insufficient overlapping data with Carhart factors")
                
                strategy_excess = strategy_ret.loc[common_dates] - carhart_data.loc[common_dates, 'RF']
                
                # Regression: R_s - R_f = α + β₁(R_m - R_f) + β₂SMB + β₃HML + β₄MOM + ε
                X = carhart_data.loc[common_dates, ['Mkt-RF', 'SMB', 'HML', 'MOM']]
                X = add_constant(X)
                model = OLS(strategy_excess, X).fit()
                
                results = {
                    'alpha': model.params[0] * 252,
                    'market_beta': model.params[1],
                    'smb_beta': model.params[2],
                    'hml_beta': model.params[3],
                    'mom_beta': model.params[4],
                    'alpha_tstat': model.tvalues[0],
                    'alpha_pvalue': model.pvalues[0],
                    'r_squared': model.rsquared,
                    'tracking_error': np.sqrt(model.mse_resid * 252)
                }
            
            # Calculate additional metrics
            results.update({
                'method': method,
                'observations': len(aligned_data),
                'sharpe_ratio': self._calculate_sharpe_ratio(strategy_ret),
                'information_ratio': results.get('information_ratio', 
                    results['alpha'] / results['tracking_error'] if results.get('tracking_error', 0) > 0 else 0),
                'analysis_date': datetime.now()
            })
            
            logger.info(f"Alpha calculated using {method}: {results['alpha']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error calculating alpha: {str(e)}")
            raise
    
    def factor_analysis(self, 
                       strategy_returns: pd.Series,
                       method: AttributionMethod = AttributionMethod.FAMA_FRENCH,
                       custom_factors: Optional[pd.DataFrame] = None) -> List[AttributionResult]:
        """
        Perform factor attribution analysis.
        
        Args:
            strategy_returns: Strategy returns series
            method: Attribution method to use
            custom_factors: Custom factor data
            
        Returns:
            List of attribution results
        """
        try:
            attribution_results = []
            
            if method == AttributionMethod.FAMA_FRENCH and 'fama_french' in self.factor_data:
                ff_data = self.factor_data['fama_french']
                
                # Align data
                common_dates = strategy_returns.index.intersection(ff_data.index)
                if len(common_dates) < self.config['min_observations']:
                    raise ValueError("Insufficient data for factor analysis")
                
                strategy_excess = strategy_returns.loc[common_dates] - ff_data.loc[common_dates, 'RF']
                factors = ff_data.loc[common_dates, ['Mkt-RF', 'SMB', 'HML']]
                
                # Run factor regression
                X = add_constant(factors)
                model = OLS(strategy_excess, X).fit()
                
                # Create attribution results
                factor_names = ['Alpha', 'Market', 'Size', 'Value']
                for i, factor_name in enumerate(factor_names):
                    if i == 0:  # Alpha
                        contribution = model.params[i] * 252
                        exposure = 1.0
                    else:
                        contribution = model.params[i] * factors.iloc[:, i-1].mean() * 252
                        exposure = model.params[i]
                    
                    result = AttributionResult(
                        factor_name=factor_name,
                        exposure=exposure,
                        contribution=contribution,
                        t_statistic=model.tvalues[i],
                        p_value=model.pvalues[i],
                        r_squared=model.rsquared,
                        explanation=self._explain_factor(factor_name, exposure, contribution)
                    )
                    attribution_results.append(result)
                    
            elif method == AttributionMethod.CARHART and 'carhart' in self.factor_data:
                carhart_data = self.factor_data['carhart']
                
                common_dates = strategy_returns.index.intersection(carhart_data.index)
                strategy_excess = strategy_returns.loc[common_dates] - carhart_data.loc[common_dates, 'RF']
                factors = carhart_data.loc[common_dates, ['Mkt-RF', 'SMB', 'HML', 'MOM']]
                
                X = add_constant(factors)
                model = OLS(strategy_excess, X).fit()
                
                factor_names = ['Alpha', 'Market', 'Size', 'Value', 'Momentum']
                for i, factor_name in enumerate(factor_names):
                    if i == 0:
                        contribution = model.params[i] * 252
                        exposure = 1.0
                    else:
                        contribution = model.params[i] * factors.iloc[:, i-1].mean() * 252
                        exposure = model.params[i]
                    
                    result = AttributionResult(
                        factor_name=factor_name,
                        exposure=exposure,
                        contribution=contribution,
                        t_statistic=model.tvalues[i],
                        p_value=model.pvalues[i],
                        r_squared=model.rsquared,
                        explanation=self._explain_factor(factor_name, exposure, contribution)
                    )
                    attribution_results.append(result)
                    
            elif method == AttributionMethod.CUSTOM_FACTORS and custom_factors is not None:
                # Custom factor analysis
                common_dates = strategy_returns.index.intersection(custom_factors.index)
                strategy_ret = strategy_returns.loc[common_dates]
                factors = custom_factors.loc[common_dates]
                
                X = add_constant(factors)
                model = OLS(strategy_ret, X).fit()
                
                factor_names = ['Alpha'] + list(factors.columns)
                for i, factor_name in enumerate(factor_names):
                    if i == 0:
                        contribution = model.params[i] * 252
                        exposure = 1.0
                    else:
                        contribution = model.params[i] * factors.iloc[:, i-1].mean() * 252
                        exposure = model.params[i]
                    
                    result = AttributionResult(
                        factor_name=factor_name,
                        exposure=exposure,
                        contribution=contribution,
                        t_statistic=model.tvalues[i],
                        p_value=model.pvalues[i],
                        r_squared=model.rsquared,
                        explanation=self._explain_factor(factor_name, exposure, contribution)
                    )
                    attribution_results.append(result)
            
            # Store attribution results
            self._save_attribution_results(attribution_results, strategy_returns.name or 'unknown')
            
            logger.info(f"Factor analysis completed using {method.value}: {len(attribution_results)} factors")
            return attribution_results
            
        except Exception as e:
            logger.error(f"Error in factor analysis: {str(e)}")
            raise
    
    def generate_pnl(self, 
                    strategy_id: str,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive P&L analysis.
        
        Args:
            strategy_id: Strategy identifier
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Dictionary containing P&L analysis
        """
        try:
            # Get performance records
            if strategy_id not in self.performance_records:
                raise ValueError(f"No performance data found for strategy {strategy_id}")
            
            records = self.performance_records[strategy_id]
            
            # Filter by date range
            if start_date or end_date:
                filtered_records = []
                for record in records:
                    if start_date and record.timestamp < start_date:
                        continue
                    if end_date and record.timestamp > end_date:
                        continue
                    filtered_records.append(record)
                records = filtered_records
            
            if not records:
                raise ValueError("No records found in specified date range")
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([
                {
                    'timestamp': r.timestamp,
                    'pnl': r.pnl,
                    'returns': r.returns,
                    'cumulative_returns': r.cumulative_returns,
                    'alpha': r.alpha,
                    'sharpe_ratio': r.sharpe_ratio,
                    'drawdown': r.drawdown,
                    'volatility': r.volatility
                }
                for r in records
            ])
            
            df.set_index('timestamp', inplace=True)
            
            # Calculate P&L metrics
            total_pnl = df['pnl'].sum()
            total_return = df['cumulative_returns'].iloc[-1]
            
            # Daily statistics
            daily_pnl = df['pnl'].resample('D').sum()
            daily_returns = df['returns'].resample('D').last()
            
            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            sortino_ratio = self._calculate_sortino_ratio(daily_returns)
            calmar_ratio = self._calculate_calmar_ratio(daily_returns)
            max_drawdown = df['drawdown'].min()
            
            # Win/Loss analysis
            winning_days = (daily_pnl > 0).sum()
            losing_days = (daily_pnl < 0).sum()
            hit_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
            
            avg_win = daily_pnl[daily_pnl > 0].mean() if winning_days > 0 else 0
            avg_loss = daily_pnl[daily_pnl < 0].mean() if losing_days > 0 else 0
            profit_factor = abs(avg_win * winning_days / (avg_loss * losing_days)) if avg_loss != 0 else float('inf')
            
            # Monthly analysis
            monthly_pnl = daily_pnl.resample('M').sum()
            monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # Volatility analysis
            volatility = daily_returns.std() * np.sqrt(252)
            downside_vol = daily_returns[daily_returns < 0].std() * np.sqrt(252)
            
            # Risk metrics
            var_95 = daily_pnl.quantile(0.05)
            var_99 = daily_pnl.quantile(0.01)
            cvar_95 = daily_pnl[daily_pnl <= var_95].mean()
            
            pnl_analysis = {
                'summary': {
                    'total_pnl': total_pnl,
                    'total_return': total_return,
                    'start_date': records[0].timestamp,
                    'end_date': records[-1].timestamp,
                    'trading_days': len(daily_pnl),
                    'total_trades': len(records)
                },
                'performance_metrics': {
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'calmar_ratio': calmar_ratio,
                    'maximum_drawdown': max_drawdown,
                    'volatility': volatility,
                    'downside_volatility': downside_vol
                },
                'win_loss_analysis': {
                    'hit_rate': hit_rate,
                    'winning_days': int(winning_days),
                    'losing_days': int(losing_days),
                    'average_win': avg_win,
                    'average_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'largest_win': daily_pnl.max(),
                    'largest_loss': daily_pnl.min()
                },
                'risk_metrics': {
                    'var_95': var_95,
                    'var_99': var_99,
                    'cvar_95': cvar_95,
                    'skewness': daily_pnl.skew(),
                    'kurtosis': daily_pnl.kurtosis()
                },
                'time_series_data': {
                    'daily_pnl': daily_pnl.to_dict(),
                    'daily_returns': daily_returns.to_dict(),
                    'monthly_pnl': monthly_pnl.to_dict(),
                    'monthly_returns': monthly_returns.to_dict(),
                    'cumulative_pnl': daily_pnl.cumsum().to_dict()
                }
            }
            
            logger.info(f"P&L analysis generated for {strategy_id}: Total PnL = {total_pnl:.2f}")
            return pnl_analysis
            
        except Exception as e:
            logger.error(f"Error generating P&L analysis: {str(e)}")
            raise
    
    def monitor_strategy_correlation(self, 
                                   strategy_returns: Dict[str, pd.Series],
                                   window: int = 60) -> Dict[str, Any]:
        """
        Monitor correlation between strategies.
        
        Args:
            strategy_returns: Dictionary of strategy returns
            window: Rolling window for correlation calculation
            
        Returns:
            Correlation analysis results
        """
        try:
            # Combine returns into DataFrame
            returns_df = pd.DataFrame(strategy_returns).dropna()
            
            if len(returns_df) < window:
                raise ValueError(f"Insufficient data: need at least {window} observations")
            
            # Calculate rolling correlations
            rolling_corr = returns_df.rolling(window=window).corr()
            
            # Current correlation matrix
            current_corr = returns_df.tail(window).corr()
            self.correlation_matrix = current_corr
            
            # Identify high correlations
            high_corr_pairs = []
            strategies = list(strategy_returns.keys())
            
            for i in range(len(strategies)):
                for j in range(i + 1, len(strategies)):
                    corr = current_corr.loc[strategies[i], strategies[j]]
                    if abs(corr) > self.config['correlation_threshold']:
                        high_corr_pairs.append({
                            'strategy1': strategies[i],
                            'strategy2': strategies[j],
                            'correlation': corr,
                            'risk_level': 'high' if abs(corr) > 0.85 else 'medium'
                        })
            
            # Correlation stability analysis
            corr_stability = self._analyze_correlation_stability(rolling_corr, strategies)
            
            # Diversification metrics
            diversification_metrics = self._calculate_diversification_metrics(current_corr)
            
            correlation_analysis = {
                'current_correlations': current_corr.to_dict(),
                'high_correlation_pairs': high_corr_pairs,
                'correlation_stability': corr_stability,
                'diversification_metrics': diversification_metrics,
                'analysis_timestamp': datetime.now(),
                'window_size': window
            }
            
            # Generate correlation alerts
            self._generate_correlation_alerts(high_corr_pairs, strategies)
            
            logger.info(f"Correlation analysis completed: {len(high_corr_pairs)} high correlation pairs detected")
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error monitoring strategy correlation: {str(e)}")
            raise
    
    def detect_performance_degradation(self, 
                                     strategy_id: str,
                                     lookback_days: int = 30) -> Dict[str, Any]:
        """
        Detect performance degradation using statistical methods.
        
        Args:
            strategy_id: Strategy identifier
            lookback_days: Number of days to look back for analysis
            
        Returns:
            Performance degradation analysis
        """
        try:
            if strategy_id not in self.performance_records:
                raise ValueError(f"No performance data found for strategy {strategy_id}")
            
            records = self.performance_records[strategy_id]
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Filter recent records
            recent_records = [r for r in records if r.timestamp >= cutoff_date]
            
            if len(recent_records) < 10:
                raise ValueError("Insufficient recent data for degradation analysis")
            
            # Extract performance metrics
            returns = [r.returns for r in recent_records]
            sharpe_ratios = [r.sharpe_ratio for r in recent_records if r.sharpe_ratio is not None]
            alphas = [r.alpha for r in recent_records if r.alpha is not None]
            
            # Statistical tests for degradation
            degradation_signals = []
            
            # 1. Trend analysis
            if len(returns) > 5:
                trend_test = self._test_performance_trend(returns)
                if trend_test['p_value'] < 0.05 and trend_test['slope'] < 0:
                    degradation_signals.append({
                        'type': 'negative_trend',
                        'severity': 'high' if trend_test['slope'] < self.config['degradation_threshold'] else 'medium',
                        'p_value': trend_test['p_value'],
                        'description': f"Significant negative trend detected (slope: {trend_test['slope']:.4f})"
                    })
            
            # 2. Sharpe ratio decline
            if len(sharpe_ratios) > 5:
                recent_sharpe = np.mean(sharpe_ratios[-5:])
                historical_sharpe = np.mean(sharpe_ratios[:-5]) if len(sharpe_ratios) > 10 else np.mean(sharpe_ratios)
                
                if recent_sharpe < historical_sharpe * 0.7:  # 30% decline
                    degradation_signals.append({
                        'type': 'sharpe_decline',
                        'severity': 'high' if recent_sharpe < historical_sharpe * 0.5 else 'medium',
                        'recent_sharpe': recent_sharpe,
                        'historical_sharpe': historical_sharpe,
                        'description': f"Sharpe ratio declined from {historical_sharpe:.2f} to {recent_sharpe:.2f}"
                    })
            
            # 3. Alpha decay
            if len(alphas) > 5:
                recent_alpha = np.mean(alphas[-5:])
                historical_alpha = np.mean(alphas[:-5]) if len(alphas) > 10 else np.mean(alphas)
                
                if recent_alpha < historical_alpha - 0.02:  # 2% alpha decay
                    degradation_signals.append({
                        'type': 'alpha_decay',
                        'severity': 'high' if recent_alpha < historical_alpha - 0.05 else 'medium',
                        'recent_alpha': recent_alpha,
                        'historical_alpha': historical_alpha,
                        'description': f"Alpha declined from {historical_alpha:.4f} to {recent_alpha:.4f}"
                    })
            
            # 4. Volatility increase
            recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
            historical_vol = np.std(returns[:-10]) if len(returns) > 20 else recent_vol
            
            if recent_vol > historical_vol * 1.5:  # 50% volatility increase
                degradation_signals.append({
                    'type': 'volatility_increase',
                    'severity': 'medium',
                    'recent_volatility': recent_vol,
                    'historical_volatility': historical_vol,
                    'description': f"Volatility increased from {historical_vol:.4f} to {recent_vol:.4f}"
                })
            
            # Overall degradation score
            degradation_score = len(degradation_signals) / 4.0  # Normalize to 0-1
            
            # Generate alerts for significant degradation
            if degradation_score > 0.5:
                self._generate_degradation_alert(strategy_id, degradation_signals, degradation_score)
            
            degradation_analysis = {
                'strategy_id': strategy_id,
                'degradation_score': degradation_score,
                'degradation_signals': degradation_signals,
                'analysis_period': lookback_days,
                'records_analyzed': len(recent_records),
                'analysis_timestamp': datetime.now(),
                'overall_status': self._classify_degradation_status(degradation_score)
            }
            
            logger.info(f"Performance degradation analysis for {strategy_id}: Score = {degradation_score:.2f}")
            return degradation_analysis
            
        except Exception as e:
            logger.error(f"Error detecting performance degradation: {str(e)}")
            raise
    
    def generate_daily_report(self, 
                            strategy_id: str,
                            report_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate comprehensive daily performance report.
        
        Args:
            strategy_id: Strategy identifier
            report_date: Report date (defaults to today)
            
        Returns:
            Daily performance report
        """
        try:
            report_date = report_date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get performance data
            if strategy_id not in self.performance_records:
                raise ValueError(f"No performance data found for strategy {strategy_id}")
            
            records = self.performance_records[strategy_id]
            
            # Filter records for the report date
            daily_records = [
                r for r in records 
                if r.timestamp.date() == report_date.date()
            ]
            
            if not daily_records:
                logger.warning(f"No records found for {strategy_id} on {report_date.date()}")
                return {'status': 'no_data', 'date': report_date.date()}
            
            # Calculate daily metrics
            daily_pnl = sum(r.pnl for r in daily_records)
            daily_return = sum(r.returns for r in daily_records)
            
            # Get latest record for current state
            latest_record = daily_records[-1]
            
            # Historical comparison (last 30 days)
            historical_cutoff = report_date - timedelta(days=30)
            historical_records = [
                r for r in records 
                if historical_cutoff <= r.timestamp < report_date
            ]
            
            # Performance metrics
            if historical_records:
                historical_returns = [r.returns for r in historical_records]
                historical_pnl = [r.pnl for r in historical_records]
                
                avg_daily_return = np.mean(historical_returns)
                avg_daily_pnl = np.mean(historical_pnl)
                volatility = np.std(historical_returns) * np.sqrt(252)
                sharpe_ratio = self._calculate_sharpe_ratio(pd.Series(historical_returns))
            else:
                avg_daily_return = daily_return
                avg_daily_pnl = daily_pnl
                volatility = 0.0
                sharpe_ratio = 0.0
            
            # Position analysis
            position_analysis = self._analyze_positions(latest_record.positions)
            
            # Risk metrics
            risk_metrics = {
                'current_drawdown': latest_record.drawdown,
                'volatility': volatility,
                'beta': latest_record.beta,
                'var_95': np.percentile([r.pnl for r in historical_records], 5) if historical_records else 0,
                'sharpe_ratio': sharpe_ratio
            }
            
            # Performance attribution
            attribution_analysis = None
            try:
                if len(historical_records) >= 20:
                    returns_series = pd.Series(
                        [r.returns for r in historical_records],
                        index=[r.timestamp for r in historical_records]
                    )
                    attribution_analysis = self.factor_analysis(returns_series)
            except Exception as e:
                logger.warning(f"Could not perform attribution analysis: {str(e)}")
            
            # Generate alerts and recommendations
            alerts = self._generate_daily_alerts(daily_records, historical_records)
            recommendations = self._generate_daily_recommendations(
                daily_return, avg_daily_return, volatility, latest_record
            )
            
            daily_report = {
                'report_metadata': {
                    'strategy_id': strategy_id,
                    'report_date': report_date.date(),
                    'generation_time': datetime.now(),
                    'records_analyzed': len(daily_records),
                    'historical_period_days': 30
                },
                'daily_performance': {
                    'pnl': daily_pnl,
                    'return': daily_return,
                    'cumulative_return': latest_record.cumulative_returns,
                    'vs_average': daily_return - avg_daily_return,
                    'vs_benchmark': daily_return - latest_record.benchmark_returns
                },
                'risk_metrics': risk_metrics,
                'position_analysis': position_analysis,
                'attribution_analysis': attribution_analysis,
                'alerts': alerts,
                'recommendations': recommendations,
                'market_context': {
                    'benchmark_return': latest_record.benchmark_returns,
                    'market_volatility': 'normal',  # Would be calculated from market data
                    'regime': 'normal'  # Would come from regime detection
                }
            }
            
            # Save report to database
            self._save_daily_report(daily_report)
            
            logger.info(f"Daily report generated for {strategy_id}: PnL = {daily_pnl:.2f}, Return = {daily_return:.4f}")
            return daily_report
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            raise
    
    # Helper methods
    
    def _calculate_alpha_beta(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate alpha and beta using linear regression"""
        try:
            aligned_data = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
            if len(aligned_data) < 2:
                return 0.0, 1.0
            
            X = aligned_data.iloc[:, 1].values.reshape(-1, 1)  # Benchmark
            y = aligned_data.iloc[:, 0].values  # Strategy
            
            model = LinearRegression().fit(X, y)
            beta = model.coef_[0]
            alpha = model.intercept_ * 252  # Annualized
            
            return alpha, beta
            
        except Exception:
            return 0.0, 1.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - self.config['risk_free_rate'] / 252
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - self.config['risk_free_rate'] / 252
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_deviation = downside_returns.std()
            return excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            annual_return = returns.mean() * 252
            return annual_return / max_drawdown if max_drawdown > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_drawdown(self, returns: pd.Series) -> float:
        """Calculate current drawdown"""
        try:
            if len(returns) < 2:
                return 0.0
            
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            return drawdown.iloc[-1]
            
        except Exception:
            return 0.0
    
    def _explain_factor(self, factor_name: str, exposure: float, contribution: float) -> str:
        """Generate explanation for factor attribution"""
        explanations = {
            'Alpha': f"Strategy generates {contribution:.2%} alpha annually",
            'Market': f"Market beta of {exposure:.2f} contributes {contribution:.2%} annually",
            'Size': f"Small cap exposure of {exposure:.2f} contributes {contribution:.2%} annually",
            'Value': f"Value factor exposure of {exposure:.2f} contributes {contribution:.2%} annually",
            'Momentum': f"Momentum factor exposure of {exposure:.2f} contributes {contribution:.2%} annually"
        }
        
        return explanations.get(factor_name, f"{factor_name} factor contributes {contribution:.2%} annually")
    
    def _analyze_correlation_stability(self, rolling_corr: pd.DataFrame, strategies: List[str]) -> Dict[str, Any]:
        """Analyze correlation stability over time"""
        stability_metrics = {}
        
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                pair = f"{strategies[i]}-{strategies[j]}"
                
                # Get correlation time series for this pair
                corr_series = rolling_corr.loc[(slice(None), strategies[i]), strategies[j]]
                
                if len(corr_series) > 10:
                    stability_metrics[pair] = {
                        'mean_correlation': corr_series.mean(),
                        'correlation_volatility': corr_series.std(),
                        'stability_score': 1 - corr_series.std(),  # Higher is more stable
                        'trend': 'increasing' if corr_series.diff().mean() > 0 else 'decreasing'
                    }
        
        return stability_metrics
    
    def _calculate_diversification_metrics(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio diversification metrics"""
        try:
            # Average correlation
            corr_values = correlation_matrix.values
            upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
            avg_correlation = np.mean(upper_triangle)
            
            # Diversification ratio
            n_strategies = len(correlation_matrix)
            diversification_ratio = 1 - avg_correlation
            
            # Effective number of strategies
            eigenvalues = np.linalg.eigvals(correlation_matrix)
            effective_strategies = 1 / np.sum((eigenvalues / np.sum(eigenvalues)) ** 2)
            
            return {
                'average_correlation': avg_correlation,
                'diversification_ratio': diversification_ratio,
                'effective_strategies': effective_strategies,
                'concentration_risk': 1 - effective_strategies / n_strategies
            }
            
        except Exception as e:
            logger.error(f"Error calculating diversification metrics: {str(e)}")
            return {}
    
    def _test_performance_trend(self, returns: List[float]) -> Dict[str, Any]:
        """Test for performance trend using linear regression"""
        try:
            x = np.arange(len(returns))
            y = np.array(returns)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err
            }
            
        except Exception:
            return {'slope': 0, 'p_value': 1.0}
    
    def _classify_degradation_status(self, degradation_score: float) -> str:
        """Classify degradation status"""
        if degradation_score >= 0.75:
            return 'critical'
        elif degradation_score >= 0.5:
            return 'high'
        elif degradation_score >= 0.25:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_positions(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Analyze position composition"""
        if not positions:
            return {}
        
        total_value = sum(abs(pos) for pos in positions.values())
        
        return {
            'total_positions': len(positions),
            'total_exposure': total_value,
            'net_exposure': sum(positions.values()),
            'long_exposure': sum(pos for pos in positions.values() if pos > 0),
            'short_exposure': sum(pos for pos in positions.values() if pos < 0),
            'largest_position': max(positions.values(), key=abs),
            'position_concentration': max(abs(pos) for pos in positions.values()) / total_value if total_value > 0 else 0
        }
    
    def _generate_daily_alerts(self, daily_records: List[PerformanceRecord], historical_records: List[PerformanceRecord]) -> List[Dict[str, Any]]:
        """Generate daily performance alerts"""
        alerts = []
        
        if not daily_records:
            return alerts
        
        daily_pnl = sum(r.pnl for r in daily_records)
        latest_record = daily_records[-1]
        
        # Large loss alert
        if daily_pnl < -10000:  # Configurable threshold
            alerts.append({
                'type': 'large_loss',
                'severity': 'high',
                'message': f"Large daily loss: ${daily_pnl:,.2f}",
                'value': daily_pnl
            })
        
        # High drawdown alert
        if latest_record.drawdown < -0.05:  # 5% drawdown
            alerts.append({
                'type': 'high_drawdown',
                'severity': 'medium',
                'message': f"High drawdown: {latest_record.drawdown:.2%}",
                'value': latest_record.drawdown
            })
        
        # Low Sharpe ratio alert
        if latest_record.sharpe_ratio < 0.5:
            alerts.append({
                'type': 'low_sharpe',
                'severity': 'medium',
                'message': f"Low Sharpe ratio: {latest_record.sharpe_ratio:.2f}",
                'value': latest_record.sharpe_ratio
            })
        
        return alerts
    
    def _generate_daily_recommendations(self, daily_return: float, avg_return: float, volatility: float, latest_record: PerformanceRecord) -> List[str]:
        """Generate daily performance recommendations"""
        recommendations = []
        
        # Performance recommendations
        if daily_return < avg_return - 2 * volatility:
            recommendations.append("Consider reviewing strategy parameters due to underperformance")
        
        if latest_record.drawdown < -0.03:
            recommendations.append("Monitor risk exposure due to elevated drawdown")
        
        if volatility > 0.20:
            recommendations.append("Consider reducing position sizes due to high volatility")
        
        if latest_record.sharpe_ratio < 1.0:
            recommendations.append("Focus on improving risk-adjusted returns")
        
        return recommendations
    
    def _check_performance_alerts(self, strategy_id: str, record: PerformanceRecord):
        """Check for performance alerts and store them"""
        alerts = []
        
        # Check for significant losses
        if record.pnl < -5000:  # Configurable threshold
            alerts.append({
                'strategy_id': strategy_id,
                'alert_type': 'large_loss',
                'severity': 'high',
                'message': f"Large loss detected: ${record.pnl:,.2f}",
                'timestamp': record.timestamp
            })
        
        # Check for high drawdown
        if record.drawdown < -0.05:
            alerts.append({
                'strategy_id': strategy_id,
                'alert_type': 'high_drawdown',
                'severity': 'medium',
                'message': f"High drawdown: {record.drawdown:.2%}",
                'timestamp': record.timestamp
            })
        
        # Store alerts
        for alert in alerts:
            self.performance_alerts.append(alert)
            self._save_performance_alert(alert)
    
    def _generate_correlation_alerts(self, high_corr_pairs: List[Dict[str, Any]], strategies: List[str]):
        """Generate correlation alerts"""
        for pair in high_corr_pairs:
            if pair['risk_level'] == 'high':
                alert = {
                    'strategy_id': f"{pair['strategy1']}-{pair['strategy2']}",
                    'alert_type': 'high_correlation',
                    'severity': 'medium',
                    'message': f"High correlation detected: {pair['correlation']:.3f}",
                    'timestamp': datetime.now()
                }
                self.performance_alerts.append(alert)
                self._save_performance_alert(alert)
    
    def _generate_degradation_alert(self, strategy_id: str, signals: List[Dict[str, Any]], score: float):
        """Generate performance degradation alert"""
        alert = {
            'strategy_id': strategy_id,
            'alert_type': 'performance_degradation',
            'severity': 'high' if score > 0.75 else 'medium',
            'message': f"Performance degradation detected (score: {score:.2f})",
            'timestamp': datetime.now(),
            'details': signals
        }
        self.performance_alerts.append(alert)
        self._save_performance_alert(alert)
    
    # Database operations
    
    def _save_performance_record(self, record: PerformanceRecord):
        """Save performance record to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_records (
                        timestamp, strategy_id, pnl, returns, cumulative_returns,
                        benchmark_returns, alpha, beta, sharpe_ratio, drawdown,
                        volatility, positions, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.timestamp.isoformat(),
                    record.strategy_id,
                    record.pnl,
                    record.returns,
                    record.cumulative_returns,
                    record.benchmark_returns,
                    record.alpha,
                    record.beta,
                    record.sharpe_ratio,
                    record.drawdown,
                    record.volatility,
                    json.dumps(record.positions),
                    json.dumps(record.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving performance record: {str(e)}")
    
    def _save_attribution_results(self, results: List[AttributionResult], strategy_id: str):
        """Save attribution results to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for result in results:
                    cursor.execute('''
                        INSERT INTO attribution_results (
                            strategy_id, factor_name, exposure, contribution,
                            t_statistic, p_value, r_squared, analysis_date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        strategy_id,
                        result.factor_name,
                        result.exposure,
                        result.contribution,
                        result.t_statistic,
                        result.p_value,
                        result.r_squared,
                        datetime.now().isoformat()
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving attribution results: {str(e)}")
    
    def _save_performance_alert(self, alert: Dict[str, Any]):
        """Save performance alert to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_alerts (
                        strategy_id, alert_type, severity, message, timestamp
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    alert['strategy_id'],
                    alert['alert_type'],
                    alert['severity'],
                    alert['message'],
                    alert['timestamp'].isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving performance alert: {str(e)}")
    
    def _save_daily_report(self, report: Dict[str, Any]):
        """Save daily report to database (simplified)"""
        try:
            # In a real implementation, this would save to a reports table
            report_path = Path(f"reports/daily_report_{report['report_metadata']['strategy_id']}_{report['report_metadata']['report_date']}.json")
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving daily report: {str(e)}")
    
    # Public utility methods
    
    def get_performance_summary(self, strategy_id: str) -> Dict[str, Any]:
        """Get performance summary for a strategy"""
        if strategy_id not in self.performance_records:
            return {'status': 'no_data'}
        
        records = self.performance_records[strategy_id]
        if not records:
            return {'status': 'no_data'}
        
        latest = records[-1]
        
        return {
            'strategy_id': strategy_id,
            'total_records': len(records),
            'latest_update': latest.timestamp,
            'cumulative_return': latest.cumulative_returns,
            'current_alpha': latest.alpha,
            'current_sharpe': latest.sharpe_ratio,
            'current_drawdown': latest.drawdown,
            'total_pnl': sum(r.pnl for r in records),
            'status': 'active'
        }
    
    def get_alerts(self, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        if strategy_id:
            return [alert for alert in self.performance_alerts if alert['strategy_id'] == strategy_id]
        return self.performance_alerts.copy()
    
    def clear_alerts(self, strategy_id: Optional[str] = None):
        """Clear performance alerts"""
        if strategy_id:
            self.performance_alerts = [
                alert for alert in self.performance_alerts 
                if alert['strategy_id'] != strategy_id
            ]
        else:
            self.performance_alerts.clear()
    
    def export_performance_data(self, strategy_id: str, filepath: str, format: str = 'csv'):
        """Export performance data to file"""
        try:
            if strategy_id not in self.performance_records:
                raise ValueError(f"No data found for strategy {strategy_id}")
            
            records = self.performance_records[strategy_id]
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': r.timestamp,
                    'pnl': r.pnl,
                    'returns': r.returns,
                    'cumulative_returns': r.cumulative_returns,
                    'alpha': r.alpha,
                    'beta': r.beta,
                    'sharpe_ratio': r.sharpe_ratio,
                    'drawdown': r.drawdown,
                    'volatility': r.volatility
                }
                for r in records
            ])
            
            if format.lower() == 'csv':
                df.to_csv(filepath, index=False)
            elif format.lower() == 'excel':
                df.to_excel(filepath, index=False)
            elif format.lower() == 'json':
                df.to_json(filepath, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Performance data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {str(e)}")
            raise 