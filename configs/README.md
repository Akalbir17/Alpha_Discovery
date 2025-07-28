# Alpha Discovery Configuration Management

This directory contains the comprehensive configuration management system for the Alpha Discovery algorithmic trading platform. The configuration system provides centralized, validated, and environment-aware configuration management.

## 📁 Configuration Files

### Core Configuration Files

| File | Description | Purpose |
|------|-------------|---------|
| `strategies.yaml` | Trading strategies and agent parameters | Defines agent behavior, strategy allocations, risk controls, and execution parameters |
| `market_data.yaml` | Market data sources and processing | Configures data feeds, symbols, processing pipelines, and storage |
| `risk.yaml` | Risk management parameters | Sets risk limits, VaR parameters, stress testing, and circuit breakers |
| `monitoring.yaml` | System monitoring and Reddit sentiment | Configures alerts, Reddit monitoring, and system health checks |
| `api.yaml` | API configuration and rate limits | Defines endpoints, authentication, rate limiting, and external APIs |
| `models.yaml` | Machine learning model parameters | Sets hyperparameters for all trading and analysis models |

### Environment-Specific Overrides

| File | Description |
|------|-------------|
| `strategies.development.yaml` | Development environment overrides for strategies |
| `strategies.production.yaml` | Production environment overrides for strategies |
| `*.development.yaml` | Development overrides for any configuration |
| `*.production.yaml` | Production overrides for any configuration |

### Configuration Management

| File | Description |
|------|-------------|
| `config_loader.py` | Configuration loader with validation and hot-reloading |
| `README.md` | This documentation file |

## 🚀 Quick Start

### Basic Usage

```python
from configs.config_loader import get_config, get_config_value

# Load complete configuration
strategies_config = get_config('strategies')
risk_config = get_config('risk')

# Get specific values using dot notation
initial_capital = get_config_value('strategies.global.initial_capital')
max_daily_var = get_config_value('risk.portfolio_limits.max_daily_var')
reddit_keywords = get_config_value('monitoring.reddit.keywords.stock_symbols.high_priority')
```

### Advanced Usage

```python
from configs.config_loader import ConfigLoader

# Initialize with specific environment
config_loader = ConfigLoader(environment='production')

# Watch for configuration changes
def on_config_change(config_name, config_data):
    print(f"Configuration {config_name} changed!")

config_loader.add_callback(on_config_change)

# Validate all configurations
validation_results = config_loader.validate_all_configs()

# Export configuration
yaml_export = config_loader.export_config('strategies', 'yaml')
```

## 📋 Configuration Details

### 1. Strategies Configuration (`strategies.yaml`)

**Purpose**: Central configuration for all trading strategies, agents, and execution parameters.

**Key Sections**:
- **Global Settings**: Environment, capital, trading modes
- **Agent Configuration**: Technical, fundamental, sentiment, risk, and options agents
- **Strategy Parameters**: Momentum, mean reversion, arbitrage, pairs trading, options
- **Portfolio Management**: Position limits, diversification, rebalancing
- **Risk Management**: VaR limits, circuit breakers, stress testing
- **Execution**: Order management, market impact, algorithms
- **Backtesting**: Periods, parameters, analysis settings
- **Performance**: Targets, benchmarks, reporting

**Example Configuration**:
```yaml
global:
  initial_capital: 1000000
  enable_live_trading: true
  max_leverage: 1.5

agents:
  technical_agent:
    confidence_threshold: 0.6
    indicators:
      rsi:
        period: 14
        overbought_threshold: 70
```

### 2. Market Data Configuration (`market_data.yaml`)

**Purpose**: Configure market data sources, symbols, processing, and storage.

**Key Sections**:
- **Data Sources**: Primary exchanges, alternative data, crypto, options
- **Symbol Configuration**: Equities, ETFs, cryptocurrencies with metadata
- **Data Processing**: Real-time and historical processing pipelines
- **Storage**: Time series database, caching, backup
- **Feeds**: Market data, news, economic data
- **Monitoring**: Data quality, performance, cost monitoring
- **Failover**: Data source failover, geographic redundancy

**Example Configuration**:
```yaml
data_sources:
  exchanges:
    nasdaq:
      enabled: true
      api_key: "${NASDAQ_API_KEY}"
      rate_limit: 100
      
symbols:
  equities:
    large_cap:
      - symbol: "AAPL"
        sector: "Technology"
        market_cap: 2800000000000
```

### 3. Risk Management Configuration (`risk.yaml`)

**Purpose**: Comprehensive risk management framework with limits and controls.

**Key Sections**:
- **VaR Configuration**: Multiple calculation methods, time horizons
- **Position Limits**: Individual positions, sectors, asset classes
- **Portfolio Limits**: Total risk, volatility, correlation, drawdown
- **Liquidity Risk**: ADV ratios, market impact, stress testing
- **Credit Risk**: Counterparty limits, credit quality
- **Operational Risk**: System risk, key person risk, processes
- **Stress Testing**: Historical and hypothetical scenarios
- **Circuit Breakers**: Automatic risk controls and recovery

**Example Configuration**:
```yaml
var_config:
  methods:
    historical_simulation:
      enabled: true
      lookback_period: 252
      
position_limits:
  individual:
    max_position_size: 0.10
    max_position_var: 0.01
```

### 4. Monitoring Configuration (`monitoring.yaml`)

**Purpose**: System monitoring, Reddit sentiment analysis, and alerting.

**Key Sections**:
- **Reddit Monitoring**: Subreddits, keywords, sentiment analysis
- **Alert Configuration**: Thresholds, severity levels, routing
- **System Monitoring**: Infrastructure, applications, logs
- **Notifications**: Email, Slack, PagerDuty, SMS
- **Dashboards**: Grafana, custom dashboards

**Reddit Keywords Monitored**:
- **Stock Symbols**: AAPL, MSFT, GOOGL, AMZN, TSLA, etc.
- **Trading Terms**: moon, rocket, diamond hands, paper hands, etc.
- **Market Events**: earnings, FDA approval, merger, etc.

**Example Configuration**:
```yaml
reddit:
  subreddits:
    trading:
      - name: "wallstreetbets"
        weight: 0.3
        min_upvotes: 100
        
  keywords:
    stock_symbols:
      high_priority: ["AAPL", "MSFT", "GOOGL"]
```

### 5. API Configuration (`api.yaml`)

**Purpose**: API endpoints, authentication, rate limiting, and external integrations.

**Key Sections**:
- **Authentication**: JWT, API keys, OAuth 2.0
- **Rate Limiting**: Global, user-based, endpoint-specific limits
- **Endpoints**: Market data, trading, analytics, system endpoints
- **External APIs**: Market data providers, brokers, social media
- **WebSocket**: Real-time data streaming
- **Security**: CORS, input validation, request logging

**Example Configuration**:
```yaml
rate_limiting:
  user_based:
    premium_tier:
      requests_per_minute: 600
      concurrent_requests: 20
      
endpoints:
  trading:
    orders:
      methods: ["GET", "POST", "PUT", "DELETE"]
      authentication: "required"
```

### 6. Models Configuration (`models.yaml`)

**Purpose**: Machine learning model hyperparameters and deployment settings.

**Key Sections**:
- **Technical Models**: LSTM, Transformer, CNN for price prediction
- **Fundamental Models**: XGBoost, Random Forest, Neural Networks
- **Sentiment Models**: BERT, VADER, FinBERT
- **Regime Models**: Hidden Markov Models, Markov Switching
- **Options Models**: Black-Scholes, Heston
- **Risk Models**: VaR Neural Networks, GARCH
- **Reinforcement Learning**: DQN, PPO
- **Ensemble Models**: Voting, Stacking
- **Evaluation**: Metrics, validation, backtesting

**Example Configuration**:
```yaml
technical_models:
  lstm_price_predictor:
    architecture:
      hidden_size: 128
      num_layers: 3
      dropout: 0.2
    training:
      epochs: 100
      batch_size: 64
      learning_rate: 0.001
```

## 🔧 Environment Management

### Environment Variables

The configuration system supports environment variable substitution using the `${ENV_VAR}` syntax:

```yaml
api_key: "${NASDAQ_API_KEY}"
database_url: "${DATABASE_URL}"
secret_key: "${JWT_SECRET_KEY}"
```

### Environment-Specific Overrides

Override configurations for different environments:

- **Development**: `*.development.yaml`
  - Paper trading enabled
  - Relaxed risk limits
  - More verbose logging
  - Higher error tolerance

- **Production**: `*.production.yaml`
  - Live trading enabled
  - Strict risk controls
  - Conservative parameters
  - Enhanced monitoring

### Setting Environment

```bash
# Set environment variable
export ENVIRONMENT=production

# Or in code
import os
os.environ['ENVIRONMENT'] = 'production'
```

## 🛡️ Configuration Validation

### Automatic Validation

The configuration loader automatically validates configurations using Pydantic models:

```python
# Validation happens automatically on load
config_loader = ConfigLoader()

# Manual validation
validation_results = config_loader.validate_all_configs()
```

### Validation Rules

- **Required Fields**: All required configuration sections must be present
- **Data Types**: Numeric values, strings, booleans validated
- **Ranges**: Percentages (0-1), positive numbers, valid dates
- **Formats**: Symbol formats, date formats, URL formats
- **Environment Variables**: Resolved and validated

### Custom Validators

```python
def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format."""
    import re
    return bool(re.match(r'^[A-Z]{1,5}$', symbol))

def validate_percentage(value: float) -> bool:
    """Validate percentage value (0-1)."""
    return 0 <= value <= 1
```

## 🔄 Hot Reloading

### Development Mode

In development, configuration files are watched for changes and automatically reloaded:

```python
# Automatic in development
config_loader = ConfigLoader(environment='development')

# Manual reload
config_loader.reload_config('strategies')  # Specific config
config_loader.reload_config()             # All configs
```

### Configuration Callbacks

Register callbacks to respond to configuration changes:

```python
def on_risk_config_change(config_name, config_data):
    if config_name == 'risk':
        # Update risk limits in real-time
        update_risk_limits(config_data)

config_loader.add_callback(on_risk_config_change)
```

## 📊 Configuration Examples

### Trading Strategy Setup

```python
from configs.config_loader import get_config_value

# Get strategy weights
momentum_weight = get_config_value('strategies.strategies.momentum_strategy.weight')
mean_reversion_weight = get_config_value('strategies.strategies.mean_reversion_strategy.weight')

# Get risk limits
max_position_size = get_config_value('strategies.portfolio.position_limits.max_position_size')
max_daily_var = get_config_value('risk.portfolio_limits.max_daily_var')

# Get execution parameters
max_slippage = get_config_value('strategies.execution.order_management.max_slippage')
```

### Reddit Sentiment Monitoring

```python
# Get Reddit configuration
reddit_config = get_config('monitoring')['reddit']

# Get keywords to monitor
stock_symbols = reddit_config['keywords']['stock_symbols']['high_priority']
trading_terms = reddit_config['keywords']['trading_terms']['bullish']

# Get subreddits
subreddits = [sub['name'] for sub in reddit_config['subreddits']['trading']]
```

### Model Configuration

```python
# Get LSTM model parameters
lstm_config = get_config_value('models.technical_models.lstm_price_predictor')

hidden_size = lstm_config['architecture']['hidden_size']
num_layers = lstm_config['architecture']['num_layers']
learning_rate = lstm_config['training']['learning_rate']
```

## 🚨 Best Practices

### 1. Configuration Security

- **Never commit secrets**: Use environment variables for API keys and secrets
- **Validate inputs**: Always validate configuration values
- **Use minimal permissions**: Grant only necessary access to configuration files

### 2. Configuration Management

- **Version control**: Keep configuration files in version control
- **Environment separation**: Use environment-specific overrides
- **Documentation**: Document all configuration parameters
- **Testing**: Test configuration changes in development first

### 3. Performance Optimization

- **Caching**: Configuration is cached for performance
- **Lazy loading**: Load configurations only when needed
- **Hot reloading**: Use only in development, not production

### 4. Error Handling

- **Graceful degradation**: Handle missing or invalid configurations
- **Fallback values**: Provide sensible defaults
- **Logging**: Log configuration errors and changes

## 🔍 Troubleshooting

### Common Issues

1. **Environment Variable Not Found**
   ```
   Error: Environment variable 'API_KEY' not found
   Solution: Set the environment variable or provide a default
   ```

2. **Configuration Validation Error**
   ```
   Error: Invalid percentage value: 1.5
   Solution: Use values between 0 and 1 for percentages
   ```

3. **File Permission Error**
   ```
   Error: Permission denied reading config file
   Solution: Check file permissions and ownership
   ```

### Debug Mode

Enable debug logging for configuration issues:

```python
import logging
logging.getLogger('configs.config_loader').setLevel(logging.DEBUG)
```

### Configuration Backup

Backup configurations before making changes:

```python
backup_path = config_loader.backup_configs()
print(f"Backup created: {backup_path}")
```

## 📈 Configuration Monitoring

### Performance Metrics

- Configuration load time
- Validation time
- File change detection latency
- Memory usage

### Health Checks

- Configuration file integrity
- Environment variable availability
- Validation status
- Last reload timestamp

### Alerts

- Configuration validation failures
- Missing environment variables
- File permission issues
- Performance degradation

## 🔮 Future Enhancements

### Planned Features

1. **Configuration UI**: Web interface for configuration management
2. **A/B Testing**: Configuration experiments and rollouts
3. **Audit Trail**: Track all configuration changes
4. **Encryption**: Encrypt sensitive configuration values
5. **Remote Configuration**: Load configurations from remote sources
6. **Configuration Templates**: Template system for common configurations

### Contributing

To add new configuration options:

1. Update the appropriate YAML file
2. Update the Pydantic model in `config_loader.py`
3. Add validation rules if needed
4. Update this documentation
5. Add tests for the new configuration

---

For questions or issues with configuration management, please contact the development team or create an issue in the project repository. 