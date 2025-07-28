"""
Dashboard configuration and utility functions
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

@dataclass
class DashboardConfig:
    """Configuration for dashboard components"""
    
    # Database connections
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "alpha_discovery")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    # API endpoints
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # Dashboard settings
    refresh_interval: int = int(os.getenv("REFRESH_INTERVAL", "5"))
    max_data_points: int = int(os.getenv("MAX_DATA_POINTS", "1000"))
    
    # Display settings
    default_timezone: str = os.getenv("TIMEZONE", "UTC")
    currency_symbol: str = os.getenv("CURRENCY_SYMBOL", "$")
    
    # Chart settings
    chart_height: int = 400
    chart_theme: str = "plotly"
    
    # Risk thresholds
    var_threshold: float = 0.05
    correlation_threshold: float = 0.8
    concentration_threshold: float = 0.3
    
    # Alert settings
    enable_alerts: bool = True
    alert_email: Optional[str] = os.getenv("ALERT_EMAIL")
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.refresh_interval < 1:
            self.refresh_interval = 1
        
        if self.max_data_points < 100:
            self.max_data_points = 100

# Color schemes for different components
COLORS = {
    'positive': '#28a745',
    'negative': '#dc3545',
    'neutral': '#6c757d',
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'success': '#28a745',
    'info': '#17a2b8'
}

# Market regime colors
REGIME_COLORS = {
    'Bull Market': '#28a745',
    'Bear Market': '#dc3545',
    'High Volatility': '#ffc107',
    'Low Volatility': '#17a2b8'
}

# Agent signal colors
SIGNAL_COLORS = {
    'BUY': '#28a745',
    'SELL': '#dc3545',
    'HOLD': '#ffc107'
}

# Sentiment colors
SENTIMENT_COLORS = {
    'positive': '#28a745',
    'negative': '#dc3545',
    'neutral': '#6c757d'
}

def get_color_for_value(value: float, positive_threshold: float = 0, 
                       negative_threshold: float = 0) -> str:
    """Get color based on value thresholds"""
    if value > positive_threshold:
        return COLORS['positive']
    elif value < negative_threshold:
        return COLORS['negative']
    else:
        return COLORS['neutral']

def format_currency(value: float, symbol: str = "$") -> str:
    """Format currency values"""
    if abs(value) >= 1e9:
        return f"{symbol}{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{symbol}{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{symbol}{value/1e3:.1f}K"
    else:
        return f"{symbol}{value:.2f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage values"""
    return f"{value*100:.{decimals}f}%"

def get_risk_level_color(risk_level: str) -> str:
    """Get color for risk level"""
    risk_colors = {
        'LOW': COLORS['success'],
        'MEDIUM': COLORS['warning'],
        'HIGH': COLORS['danger']
    }
    return risk_colors.get(risk_level.upper(), COLORS['neutral'])

# Logging configuration
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dashboard.log')
        ]
    )
    return logging.getLogger(__name__)

# Mock data generators for development
def generate_mock_portfolio_data() -> Dict:
    """Generate mock portfolio data for development"""
    import random
    import numpy as np
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
    positions = []
    
    for symbol in random.sample(symbols, 5):
        quantity = random.randint(10, 200)
        price = random.uniform(50, 300)
        value = quantity * price
        pnl = random.uniform(-value * 0.1, value * 0.15)
        
        positions.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': value,
            'pnl': pnl
        })
    
    total_value = sum(pos['value'] for pos in positions)
    total_pnl = sum(pos['pnl'] for pos in positions)
    
    return {
        'total_value': total_value,
        'pnl_today': total_pnl * 0.3,
        'pnl_total': total_pnl,
        'positions': positions
    }

def generate_mock_time_series(days: int = 30, frequency: str = '1D') -> Dict:
    """Generate mock time series data"""
    import pandas as pd
    import numpy as np
    
    timestamps = pd.date_range(
        start=pd.Timestamp.now() - pd.Timedelta(days=days),
        end=pd.Timestamp.now(),
        freq=frequency
    )
    
    # Generate random walk
    returns = np.random.normal(0.001, 0.02, len(timestamps))
    prices = 100 * np.cumprod(1 + returns)
    
    return {
        'timestamps': [t.isoformat() for t in timestamps],
        'prices': prices.tolist(),
        'returns': returns.tolist(),
        'volume': np.random.poisson(1000, len(timestamps)).tolist()
    }

# Dashboard themes
THEMES = {
    'default': {
        'background_color': '#ffffff',
        'text_color': '#000000',
        'accent_color': '#1f77b4'
    },
    'dark': {
        'background_color': '#2e2e2e',
        'text_color': '#ffffff',
        'accent_color': '#00d4ff'
    }
}

# Performance metrics definitions
PERFORMANCE_METRICS = {
    'sharpe_ratio': {
        'name': 'Sharpe Ratio',
        'description': 'Risk-adjusted return measure',
        'good_threshold': 1.0,
        'excellent_threshold': 2.0
    },
    'sortino_ratio': {
        'name': 'Sortino Ratio',
        'description': 'Downside risk-adjusted return',
        'good_threshold': 1.5,
        'excellent_threshold': 2.5
    },
    'max_drawdown': {
        'name': 'Maximum Drawdown',
        'description': 'Largest peak-to-trough decline',
        'good_threshold': -0.1,
        'excellent_threshold': -0.05
    },
    'calmar_ratio': {
        'name': 'Calmar Ratio',
        'description': 'Return to max drawdown ratio',
        'good_threshold': 1.0,
        'excellent_threshold': 2.0
    }
}

# Risk metrics definitions
RISK_METRICS = {
    'var_95': {
        'name': 'Value at Risk (95%)',
        'description': '95% confidence loss threshold',
        'format': 'currency'
    },
    'var_99': {
        'name': 'Value at Risk (99%)',
        'description': '99% confidence loss threshold',
        'format': 'currency'
    },
    'expected_shortfall': {
        'name': 'Expected Shortfall',
        'description': 'Expected loss beyond VaR',
        'format': 'currency'
    },
    'beta': {
        'name': 'Portfolio Beta',
        'description': 'Market sensitivity measure',
        'format': 'decimal'
    }
}

# Alert configurations
ALERT_CONFIGS = {
    'portfolio_loss': {
        'threshold': -0.05,
        'message': 'Portfolio loss exceeds 5%',
        'severity': 'HIGH'
    },
    'var_breach': {
        'threshold': 0.9,
        'message': 'VaR limit approaching',
        'severity': 'MEDIUM'
    },
    'correlation_spike': {
        'threshold': 0.8,
        'message': 'Portfolio correlation above threshold',
        'severity': 'HIGH'
    }
} 