# Alpha Discovery MCP Server

A state-of-the-art Model Context Protocol (MCP) server for financial analysis and alpha discovery, built with the latest free AI models and modern async patterns.

## Features

- **Latest MCP Protocol**: Implements the most recent MCP protocol (2024-11-05) with full tool discovery and execution
- **Free AI Models**: Integrated with the latest free AI models (Claude, Gemini, Llama, Mixtral, etc.)
- **Real-time Data**: WebSocket-based real-time market data streaming
- **Comprehensive Tools**: 15+ financial analysis tools including market data, sentiment analysis, and risk metrics
- **High Performance**: Async/await architecture with rate limiting and caching
- **Production Ready**: Docker support, monitoring, and comprehensive error handling

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Copy the example environment file and configure your API keys:

```bash
cp env.example .env
```

Edit `.env` with your API keys for free AI models and financial data sources.

### 3. Start the MCP Server

```bash
# Using the startup script
python src/mcp/run_server.py

# Or using Docker
docker-compose up mcp-server
```

### 4. Test the Server

```bash
# Run the test client
python src/mcp/test_client.py
```

## Available Tools

### Market Data Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_orderbook` | Get real-time order book data | `symbol`, `exchange` |
| `get_market_data` | Get comprehensive market data | `symbol`, `timeframe` |
| `get_technical_indicators` | Get technical indicators | `symbol`, `timeframe` |
| `stream_market_data` | Stream real-time data | `symbol`, `duration_seconds` |

### Sentiment Analysis Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_reddit_sentiment` | Reddit sentiment analysis | `symbol`, `timeframe` |
| `get_news_sentiment` | News sentiment analysis | `symbol`, `timeframe` |

### Analysis Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `calculate_microstructure_features` | Market microstructure analysis | `symbol`, `timeframe` |
| `analyze_order_flow` | Order flow analysis | `symbol`, `timeframe` |
| `detect_anomalies` | Anomaly detection | `symbol`, `anomaly_type`, `sensitivity` |

### Prediction Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `detect_regime_change` | Market regime detection | `symbol`, `lookback_days` |
| `forecast_volatility` | Volatility forecasting | `symbol`, `forecast_days`, `confidence_level` |

### Risk Management Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `calculate_risk_metrics` | Risk metrics calculation | `symbol`, `portfolio_value` |
| `optimize_portfolio` | Portfolio optimization | `symbols`, `portfolio_value`, `risk_tolerance` |

### Alternative Data Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_alternative_data` | Alternative data sources | `symbol`, `data_type` |
| `analyze_correlation` | Correlation analysis | `symbols`, `timeframe` |

## API Usage

### WebSocket Connection

Connect to the MCP server via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8001');

ws.onopen = function() {
    console.log('Connected to MCP server');
    
    // Initialize connection
    ws.send(JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: {}
    }));
};
```

### Tool Discovery

List all available tools:

```javascript
ws.send(JSON.stringify({
    jsonrpc: "2.0",
    id: 2,
    method: "tools/list",
    params: {}
}));
```

### Tool Execution

Execute a tool:

```javascript
ws.send(JSON.stringify({
    jsonrpc: "2.0",
    id: 3,
    method: "tools/call",
    params: {
        calls: [{
            name: "get_market_data",
            arguments: {
                symbol: "AAPL",
                timeframe: "1d"
            },
            id: "call_1"
        }]
    }
}));
```

### Real-time Streaming

Subscribe to real-time data:

```javascript
ws.send(JSON.stringify({
    jsonrpc: "2.0",
    id: 4,
    method: "resources/subscribe",
    params: {
        uris: ["alpha-discovery://market-data"]
    }
}));
```

## Python Client Example

```python
import asyncio
import websockets
import json

async def mcp_client():
    uri = "ws://localhost:8001"
    async with websockets.connect(uri) as websocket:
        # Initialize
        await websocket.send(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }))
        
        # Get market data
        await websocket.send(json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "calls": [{
                    "name": "get_market_data",
                    "arguments": {"symbol": "TSLA"},
                    "id": "call_1"
                }]
            }
        }))
        
        # Receive responses
        async for message in websocket:
            response = json.loads(message)
            print(f"Received: {response}")

asyncio.run(mcp_client())
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_HOST` | Server host | `0.0.0.0` |
| `MCP_PORT` | Server port | `8001` |
| `MCP_LOG_LEVEL` | Logging level | `INFO` |
| `MCP_RATE_LIMIT_PER_MINUTE` | Rate limit per minute | `60` |
| `MCP_ENABLE_CACHING` | Enable caching | `true` |

### API Keys Required

#### Free AI Models
- `GROQ_API_KEY` - Groq API key
- `GOOGLE_API_KEY` - Google Gemini API key
- `ANTHROPIC_API_KEY` - Anthropic Claude API key
- `TOGETHER_API_KEY` - Together AI API key
- `FIREWORKS_API_KEY` - Fireworks AI API key
- `MISTRAL_API_KEY` - Mistral AI API key

#### Financial Data
- `ALPACA_API_KEY` - Alpaca Markets API key
- `POLYGON_API_KEY` - Polygon.io API key
- `ALPHA_VANTAGE_API_KEY` - Alpha Vantage API key

#### Alternative Data
- `REDDIT_CLIENT_ID` - Reddit API client ID
- `REDDIT_CLIENT_SECRET` - Reddit API client secret
- `NEWS_API_KEY` - News API key

## Architecture

### Components

1. **MCP Server** (`mcp_server.py`): Main server implementation
2. **Tool Definitions** (`tool_definitions.py`): Tool schemas and definitions
3. **Configuration** (`config.py`): Server and tool configuration
4. **Model Manager** (`../utils/model_manager.py`): AI model routing and management
5. **Tools** (`../tools/`): Individual tool implementations

### Data Flow

```
Client Request → MCP Server → Tool Execution → AI Model → Response
                ↓
            Rate Limiting
                ↓
            Caching Layer
                ↓
            Error Handling
```

### Performance Features

- **Async/await**: Non-blocking I/O for high concurrency
- **Rate Limiting**: Per-client and per-tool rate limits
- **Caching**: Redis-based caching for expensive operations
- **Connection Pooling**: Efficient resource management
- **Error Recovery**: Automatic retries and fallbacks

## Monitoring and Logging

### Logging

The server uses structured logging with JSON format:

```json
{
    "timestamp": "2025-01-15T10:30:00Z",
    "level": "INFO",
    "logger": "mcp_server",
    "event": "tool_execution",
    "tool": "get_market_data",
    "symbol": "AAPL",
    "duration_ms": 150
}
```

### Metrics

Key metrics tracked:
- Request rate and latency
- Tool execution times
- Error rates by tool
- Model usage statistics
- Cache hit rates

### Health Checks

```bash
# Health check endpoint
curl http://localhost:8001/health

# Metrics endpoint
curl http://localhost:8001/metrics
```

## Development

### Running Tests

```bash
# Run test client
python src/mcp/test_client.py

# Run unit tests
pytest tests/mcp/

# Run integration tests
pytest tests/integration/
```

### Adding New Tools

1. Define tool schema in `tool_definitions.py`
2. Implement tool function in `mcp_server.py`
3. Add tool to the tools dictionary
4. Update documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check if server is running on correct port
2. **Authentication Errors**: Verify API keys in environment variables
3. **Rate Limit Exceeded**: Reduce request frequency or increase limits
4. **Tool Not Found**: Check tool name spelling and availability

### Debug Mode

Enable debug logging:

```bash
export MCP_LOG_LEVEL=DEBUG
python src/mcp/run_server.py
```

### Performance Tuning

- Adjust rate limits based on your needs
- Configure caching TTL for different tools
- Monitor memory usage and adjust connection limits
- Use connection pooling for external APIs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide 