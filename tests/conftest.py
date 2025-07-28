"""
Pytest configuration and fixtures for Alpha Discovery testing suite
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import redis
import psycopg2
from typing import Dict, List, Any, Optional
import sys
import os
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Test configuration
TEST_CONFIG = {
    "redis_host": "localhost",
    "redis_port": 6379,
    "postgres_host": "localhost",
    "postgres_port": 5432,
    "postgres_db": "alpha_discovery_test",
    "postgres_user": "postgres",
    "postgres_password": "postgres",
    "api_base_url": "http://localhost:8000",
    "test_data_dir": current_dir / "data",
    "benchmark_timeout": 30,  # seconds
    "load_test_duration": 60,  # seconds
    "max_memory_usage": 1024,  # MB
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return TEST_CONFIG

@pytest.fixture(scope="session")
def sample_market_data():
    """Generate sample market data for testing"""
    np.random.seed(42)  # For reproducible tests
    
    # Generate 1 year of daily data
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        freq='D'
    )
    
    # Generate realistic price data
    initial_price = 100.0
    returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual volatility
    prices = initial_price * np.cumprod(1 + returns)
    
    # Generate volume data
    volume = np.random.lognormal(10, 1, len(dates)).astype(int)
    
    # Generate OHLC data
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
    
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
        'returns': returns
    })
    
    return market_data

@pytest.fixture(scope="session")
def sample_fundamental_data():
    """Generate sample fundamental data for testing"""
    return {
        'AAPL': {
            'pe_ratio': 25.5,
            'pb_ratio': 8.2,
            'debt_to_equity': 1.73,
            'roe': 0.147,
            'revenue_growth': 0.08,
            'earnings_growth': 0.12,
            'free_cash_flow': 92.95e9,
            'market_cap': 2.8e12,
            'dividend_yield': 0.0044,
            'beta': 1.24
        },
        'GOOGL': {
            'pe_ratio': 22.1,
            'pb_ratio': 4.8,
            'debt_to_equity': 0.12,
            'roe': 0.186,
            'revenue_growth': 0.13,
            'earnings_growth': 0.15,
            'free_cash_flow': 67.0e9,
            'market_cap': 1.7e12,
            'dividend_yield': 0.0,
            'beta': 1.05
        }
    }

@pytest.fixture(scope="session")
def sample_sentiment_data():
    """Generate sample sentiment data for testing"""
    np.random.seed(42)
    
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='H'
    )
    
    sentiment_data = []
    for timestamp in timestamps:
        sentiment_data.append({
            'timestamp': timestamp,
            'symbol': 'AAPL',
            'sentiment_score': np.random.normal(0.1, 0.3),
            'volume': np.random.poisson(100),
            'source': 'reddit'
        })
    
    return pd.DataFrame(sentiment_data)

@pytest.fixture(scope="session")
def sample_options_data():
    """Generate sample options data for testing"""
    return {
        'AAPL': {
            'calls': {
                'strike_150': {'price': 5.2, 'volume': 1000, 'open_interest': 5000, 'iv': 0.25},
                'strike_155': {'price': 2.8, 'volume': 800, 'open_interest': 3000, 'iv': 0.28},
                'strike_160': {'price': 1.2, 'volume': 500, 'open_interest': 2000, 'iv': 0.32},
            },
            'puts': {
                'strike_145': {'price': 3.1, 'volume': 900, 'open_interest': 4000, 'iv': 0.24},
                'strike_140': {'price': 1.8, 'volume': 600, 'open_interest': 2500, 'iv': 0.26},
                'strike_135': {'price': 0.9, 'volume': 400, 'open_interest': 1500, 'iv': 0.29},
            }
        }
    }

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    mock_client = Mock()
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.publish.return_value = 1
    mock_client.ping.return_value = True
    return mock_client

@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL connection for testing"""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.execute.return_value = None
    return mock_conn

@pytest.fixture
def mock_api_client():
    """Mock API client for testing"""
    mock_client = AsyncMock()
    mock_client.get.return_value = {"status": "success", "data": {}}
    mock_client.post.return_value = {"status": "success", "data": {}}
    mock_client.put.return_value = {"status": "success", "data": {}}
    mock_client.delete.return_value = {"status": "success", "data": {}}
    return mock_client

@pytest.fixture
def sample_portfolio():
    """Sample portfolio for testing"""
    return {
        'cash': 100000,
        'positions': {
            'AAPL': {'quantity': 100, 'avg_price': 150.0, 'current_price': 155.0},
            'GOOGL': {'quantity': 50, 'avg_price': 2500.0, 'current_price': 2520.0},
            'MSFT': {'quantity': 75, 'avg_price': 300.0, 'current_price': 305.0}
        },
        'total_value': 165000,
        'pnl': 15000
    }

@pytest.fixture
def sample_trade_signals():
    """Sample trade signals for testing"""
    return [
        {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence': 0.8,
            'quantity': 100,
            'price': 150.0,
            'timestamp': datetime.now(),
            'agent': 'technical_agent',
            'reasoning': 'Strong momentum indicators'
        },
        {
            'symbol': 'GOOGL',
            'signal': 'SELL',
            'confidence': 0.7,
            'quantity': 25,
            'price': 2500.0,
            'timestamp': datetime.now(),
            'agent': 'fundamental_agent',
            'reasoning': 'Overvalued based on P/E ratio'
        }
    ]

@pytest.fixture
def sample_agent_debates():
    """Sample agent debates for testing"""
    return [
        {
            'topic': 'Market Direction',
            'participants': ['technical_agent', 'fundamental_agent'],
            'arguments': {
                'technical_agent': {
                    'position': 'BULLISH',
                    'confidence': 0.8,
                    'evidence': ['RSI oversold', 'MACD crossover', 'Volume spike']
                },
                'fundamental_agent': {
                    'position': 'BEARISH',
                    'confidence': 0.7,
                    'evidence': ['High P/E ratios', 'Declining earnings', 'Economic uncertainty']
                }
            },
            'resolution': 'NEUTRAL',
            'consensus_confidence': 0.6,
            'timestamp': datetime.now()
        }
    ]

@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing"""
    return {
        'agent_response_time': 0.1,  # seconds
        'signal_generation_time': 0.5,  # seconds
        'debate_resolution_time': 2.0,  # seconds
        'backtest_time_per_year': 5.0,  # seconds
        'api_response_time': 0.05,  # seconds
        'memory_usage_limit': 512,  # MB
        'cpu_usage_limit': 80,  # percent
    }

@pytest.fixture
def risk_limits():
    """Risk limits for testing"""
    return {
        'max_position_size': 0.1,  # 10% of portfolio
        'max_sector_exposure': 0.3,  # 30% of portfolio
        'max_daily_loss': 0.02,  # 2% of portfolio
        'max_drawdown': 0.1,  # 10% of portfolio
        'min_cash_reserve': 0.05,  # 5% of portfolio
        'max_leverage': 1.5,  # 1.5x leverage
        'var_95_limit': 0.05,  # 5% VaR
        'correlation_limit': 0.8,  # 80% correlation
    }

@pytest.fixture
def historical_scenarios():
    """Historical market scenarios for testing"""
    return {
        '2008_crisis': {
            'start_date': '2008-09-01',
            'end_date': '2009-03-01',
            'market_return': -0.45,
            'volatility': 0.65,
            'max_drawdown': -0.55
        },
        'covid_crash': {
            'start_date': '2020-02-01',
            'end_date': '2020-04-01',
            'market_return': -0.35,
            'volatility': 0.82,
            'max_drawdown': -0.38
        },
        'flash_crash': {
            'start_date': '2010-05-06',
            'end_date': '2010-05-07',
            'market_return': -0.09,
            'volatility': 1.2,
            'max_drawdown': -0.12
        }
    }

@pytest.fixture
def load_test_config():
    """Load testing configuration"""
    return {
        'concurrent_users': 100,
        'ramp_up_time': 10,  # seconds
        'test_duration': 60,  # seconds
        'request_rate': 10,  # requests per second
        'endpoints': [
            '/api/v1/portfolio',
            '/api/v1/signals',
            '/api/v1/agents/consensus',
            '/api/v1/risk/metrics',
            '/api/v1/performance'
        ]
    }

# Test utilities
class TestDataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def generate_price_series(length: int = 100, volatility: float = 0.02) -> pd.Series:
        """Generate realistic price series"""
        np.random.seed(42)
        returns = np.random.normal(0.0008, volatility, length)
        prices = 100 * np.cumprod(1 + returns)
        return pd.Series(prices)
    
    @staticmethod
    def generate_order_book(symbol: str = 'AAPL', depth: int = 10) -> Dict:
        """Generate realistic order book data"""
        mid_price = 150.0
        spread = 0.01
        
        bids = []
        asks = []
        
        for i in range(depth):
            bid_price = mid_price - spread/2 - i * 0.01
            ask_price = mid_price + spread/2 + i * 0.01
            
            bid_size = np.random.exponential(1000)
            ask_size = np.random.exponential(1000)
            
            bids.append([bid_price, bid_size])
            asks.append([ask_price, ask_size])
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def generate_trade_data(count: int = 100) -> List[Dict]:
        """Generate realistic trade data"""
        trades = []
        base_price = 150.0
        
        for i in range(count):
            price = base_price + np.random.normal(0, 0.5)
            size = np.random.exponential(100)
            side = np.random.choice(['buy', 'sell'])
            
            trades.append({
                'symbol': 'AAPL',
                'price': price,
                'size': size,
                'side': side,
                'timestamp': datetime.now() - timedelta(seconds=i)
            })
        
        return trades

class TestMetrics:
    """Utility class for collecting test metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_timing(self, operation: str, duration: float):
        """Record timing metrics"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for operation"""
        if operation in self.metrics:
            return np.mean(self.metrics[operation])
        return 0.0
    
    def get_percentile(self, operation: str, percentile: float) -> float:
        """Get percentile for operation"""
        if operation in self.metrics:
            return np.percentile(self.metrics[operation], percentile)
        return 0.0

@pytest.fixture
def test_metrics():
    """Test metrics collector"""
    return TestMetrics()

# Async test utilities
def async_test(coro):
    """Decorator for async tests"""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

# Performance testing decorators
def benchmark(max_time: float = 1.0):
    """Decorator for benchmarking test performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            assert duration <= max_time, f"Test took {duration:.2f}s, expected <= {max_time}s"
            return result
        return wrapper
    return decorator

def memory_limit(max_mb: int = 512):
    """Decorator for memory usage testing"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            
            assert memory_used <= max_mb, f"Test used {memory_used:.1f}MB, expected <= {max_mb}MB"
            return result
        return wrapper
    return decorator

# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Cleanup test data after each test"""
    yield
    # Cleanup logic here if needed
    pass

# Skip markers for optional dependencies
redis_available = pytest.mark.skipif(
    not hasattr(redis, 'Redis'),
    reason="Redis not available"
)

postgres_available = pytest.mark.skipif(
    not hasattr(psycopg2, 'connect'),
    reason="PostgreSQL not available"
)

# Test categories
unit_test = pytest.mark.unit
integration_test = pytest.mark.integration
performance_test = pytest.mark.performance
load_test = pytest.mark.load
slow_test = pytest.mark.slow 