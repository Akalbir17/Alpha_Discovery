# Alpha Discovery Dashboard

A comprehensive Streamlit dashboard for monitoring quantitative trading strategies, market data, and risk metrics in real-time.

## Features

### 📊 Portfolio Overview
- Real-time P&L tracking
- Position monitoring
- Portfolio value metrics
- Performance visualization

### 🤖 Agent Consensus
- Multi-agent trading signals
- Consensus confidence levels
- Agent debate tracking
- Individual agent reasoning

### 📈 Market Microstructure
- Order book visualization
- Trade flow analysis
- Bid-ask spread monitoring
- Market depth charts

### 💬 Sentiment Analysis
- Reddit sentiment trends
- Social media mention tracking
- Sentiment score visualization
- Top mentioned symbols

### 🔄 Regime Monitoring
- Market regime detection
- Regime transition probabilities
- Historical regime analysis
- Confidence scoring

### 📊 Performance Metrics
- Strategy performance tracking
- Risk-adjusted returns
- Drawdown analysis
- Statistical measures

### 💼 Trade Execution
- Manual trade interface
- Order management
- Strategy controls
- Execution monitoring

### ⚠️ Risk Monitoring
- VaR calculations
- Stress testing
- Sector exposure
- Risk alerts

## Installation

### Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
python run_dashboard.py
```

Or directly with Streamlit:
```bash
streamlit run app.py
```

### Docker Installation

1. Build the Docker image:
```bash
docker build -t alpha-discovery-dashboard .
```

2. Run the container:
```bash
docker run -p 8501:8501 \
  -e REDIS_HOST=localhost \
  -e POSTGRES_HOST=localhost \
  alpha-discovery-dashboard
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | localhost | Redis server hostname |
| `REDIS_PORT` | 6379 | Redis server port |
| `POSTGRES_HOST` | localhost | PostgreSQL hostname |
| `POSTGRES_PORT` | 5432 | PostgreSQL port |
| `POSTGRES_DB` | alpha_discovery | Database name |
| `POSTGRES_USER` | postgres | Database user |
| `POSTGRES_PASSWORD` | postgres | Database password |
| `API_BASE_URL` | http://localhost:8000 | API server URL |
| `REFRESH_INTERVAL` | 5 | Auto-refresh interval (seconds) |
| `MAX_DATA_POINTS` | 1000 | Maximum data points to display |
| `TIMEZONE` | UTC | Default timezone |
| `CURRENCY_SYMBOL` | $ | Currency symbol |

### Dashboard Configuration

The dashboard can be configured through the `config.py` file:

```python
from config import DashboardConfig

config = DashboardConfig(
    refresh_interval=10,
    chart_height=500,
    enable_alerts=True
)
```

## Usage

### Navigation

Use the sidebar to navigate between different dashboard sections:

1. **Portfolio Overview** - Monitor P&L and positions
2. **Agent Consensus** - View agent signals and debates
3. **Market Microstructure** - Analyze order flow and spreads
4. **Sentiment Analysis** - Track social media sentiment
5. **Regime Monitoring** - Monitor market regime changes
6. **Performance Metrics** - Analyze strategy performance
7. **Trade Execution** - Execute trades manually
8. **Risk Monitoring** - Monitor portfolio risk
9. **All Dashboards** - View all sections in tabs

### Auto-Refresh

Enable auto-refresh in the sidebar to automatically update data at specified intervals. Adjust the refresh rate using the slider.

### Debug Mode

Enable debug mode in the sidebar to view connection status and diagnostic information.

## Data Sources

The dashboard connects to multiple data sources:

- **Redis** - Real-time data cache
- **PostgreSQL** - Historical data storage
- **REST API** - Trading system integration
- **WebSocket** - Live market data feeds

## Development

### Mock Data

For development and testing, the dashboard includes mock data generators. When database connections are unavailable, the dashboard automatically switches to mock data mode.

### Adding New Panels

To add new dashboard panels:

1. Create a new method in `DashboardComponents` class
2. Add data retrieval logic in `DataManager` class
3. Update the sidebar navigation
4. Add any required configuration in `config.py`

### Customization

The dashboard supports theming and customization:

- Modify colors in `config.py`
- Update chart configurations
- Add custom CSS styling
- Configure alert thresholds

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Check Redis and PostgreSQL connectivity
   - Verify environment variables
   - Check firewall settings

2. **Performance Issues**
   - Reduce refresh interval
   - Limit data points
   - Optimize database queries

3. **Display Issues**
   - Clear browser cache
   - Check browser compatibility
   - Verify Streamlit version

### Logs

Dashboard logs are written to `dashboard.log` and console output. Enable debug mode for detailed logging.

## Security

- Dashboard runs on localhost by default
- Use environment variables for sensitive configuration
- Enable authentication for production deployments
- Configure HTTPS for secure connections

## Performance

- Data is cached in Redis for fast access
- Database queries are optimized
- Charts are rendered client-side
- Auto-refresh can be disabled for better performance

## License

This dashboard is part of the Alpha Discovery platform and follows the same licensing terms. 