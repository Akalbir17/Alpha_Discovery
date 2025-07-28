# Microstructure Analysis Agent

A state-of-the-art market microstructure analysis agent using CrewAI framework, implementing advanced algorithms for order flow analysis, price impact measurement, and alpha signal generation.

## Features

- **VPIN Analysis**: Volume-synchronized Probability of Informed Trading
- **Kyle's Lambda**: Price impact measurement using linear regression
- **Lee-Ready Algorithm**: Order flow classification and imbalance detection
- **Flow Pattern Detection**: Institutional vs retail flow identification
- **Market Depth Analysis**: Bid-ask spread dynamics and liquidity measurement
- **Memory-Based Pattern Recognition**: Historical pattern matching and learning
- **AI-Enhanced Signal Generation**: Integration with free AI models (Claude, Gemini, Llama)
- **CrewAI Framework**: Multi-agent collaboration for complex analysis

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# AI Models (for enhanced analysis)
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Market Data APIs
ALPACA_API_KEY=your_alpaca_key
POLYGON_API_KEY=your_polygon_key
```

### 3. Basic Usage

```python
import asyncio
from src.agents.microstructure_agent import microstructure_agent

async def main():
    # Analyze microstructure for a symbol
    analysis = await microstructure_agent.analyze_microstructure("AAPL", "1h")
    print(f"VPIN: {analysis.vpin:.4f}")
    print(f"Kyle's Lambda: {analysis.kyle_lambda:.6f}")
    print(f"Flow Imbalance: {analysis.flow_imbalance:.4f}")
    
    # Generate alpha signals
    signals = await microstructure_agent.generate_signals("TSLA", "1h")
    for signal in signals:
        print(f"{signal.signal_type}: {signal.alpha_score:.2f}")

asyncio.run(main())
```

### 4. Run Tests

```bash
python src/agents/test_microstructure_agent.py
```

## Algorithms Implemented

### VPIN (Volume-synchronized Probability of Informed Trading)

VPIN measures the probability of informed trading by analyzing order flow imbalances in volume-synchronized buckets.

**Implementation:**
```python
vpin_calculator = VPINCalculator()
vpin = vpin_calculator.calculate_vpin(trades_data)
```

**Key Features:**
- Volume-synchronized bucketing (50 buckets by default)
- Order flow imbalance calculation
- Normalized VPIN values (0-1 range)
- High VPIN indicates informed trading

**Interpretation:**
- VPIN > 0.7: High probability of informed trading (often bearish)
- VPIN < 0.3: Low probability of informed trading
- VPIN 0.3-0.7: Normal market conditions

### Kyle's Lambda

Kyle's Lambda measures the price impact of trades, representing market depth and liquidity.

**Implementation:**
```python
kyle_calculator = KylesLambdaCalculator()
lambda_val = kyle_calculator.calculate_kyle_lambda(orderbook_data, trades_data)
```

**Key Features:**
- Linear regression of price changes on signed volume
- Price impact parameter estimation
- Market depth measurement
- Liquidity assessment

**Interpretation:**
- High Lambda: Low liquidity, high price impact
- Low Lambda: High liquidity, low price impact
- Lambda typically ranges from 0.0001 to 0.01

### Lee-Ready Algorithm

The Lee-Ready algorithm classifies trades as buyer or seller initiated using quote-based and tick-based methods.

**Implementation:**
```python
lee_ready = LeeReadyAlgorithm()
classified_trades = lee_ready.classify_trades(trades_data, quotes_data)
```

**Key Features:**
- Quote-based classification (within 5-second threshold)
- Tick test fallback for older quotes
- Trade initiation identification
- Order flow imbalance calculation

**Classification Methods:**
1. **Quote-Based**: Compare trade price to mid-quote
2. **Tick Test**: Compare to previous trade price
3. **Hybrid**: Use quote when available, tick test otherwise

## API Reference

### Core Methods

#### `analyze_microstructure(symbol: str, timeframe: str = "1h")`

Perform comprehensive microstructure analysis.

**Parameters:**
- `symbol`: Trading symbol (e.g., "AAPL", "TSLA")
- `timeframe`: Analysis timeframe ("1h", "4h", "1d")

**Returns:**
```python
OrderFlowAnalysis(
    symbol="AAPL",
    timestamp=datetime.now(),
    vpin=0.65,
    kyle_lambda=0.0012,
    flow_imbalance=0.3,
    bid_ask_spread=0.15,
    market_depth={"bid_depth": 5000, "ask_depth": 4500},
    institutional_ratio=0.7,
    retail_ratio=0.3,
    toxicity_score=0.45,
    price_impact=0.0008
)
```

#### `generate_signals(symbol: str, timeframe: str = "1h")`

Generate alpha signals using AI-enhanced analysis.

**Parameters:**
- `symbol`: Trading symbol
- `timeframe`: Analysis timeframe

**Returns:**
```python
[
    MicrostructureSignal(
        symbol="AAPL",
        timestamp=datetime.now(),
        signal_type="high_vpin",
        value=0.8,
        confidence=0.9,
        description="High VPIN indicates informed trading",
        evidence=["VPIN > 0.7", "High toxicity score"],
        alpha_score=-0.3
    )
]
```

#### `get_signal_performance(signal_type: str = None)`

Get historical performance metrics for signals.

**Parameters:**
- `signal_type`: Optional signal type filter

**Returns:**
```python
{
    "high_vpin": {
        "accuracy": 0.75,
        "avg_alpha": -0.2,
        "count": 45
    },
    "flow_imbalance": {
        "accuracy": 0.68,
        "avg_alpha": 0.15,
        "count": 32
    }
}
```

### Tool Methods

#### `calculate_vpin(symbol: str, timeframe: str = "1h")`

Calculate VPIN for a specific symbol.

#### `measure_price_impact(symbol: str, timeframe: str = "1h")`

Measure price impact using Kyle's Lambda.

#### `analyze_flow_imbalance(symbol: str, timeframe: str = "1h")`

Analyze order flow imbalance using Lee-Ready algorithm.

#### `detect_flow_patterns(symbol: str, timeframe: str = "1h")`

Detect institutional vs retail flow patterns.

#### `analyze_market_depth(symbol: str)`

Analyze market depth and bid-ask spread dynamics.

## Advanced Features

### Memory-Based Pattern Recognition

The agent maintains a memory system for pattern recognition and learning:

```python
# Add patterns to memory
microstructure_agent.memory.add_pattern({
    "symbol": "AAPL",
    "vpin": 0.8,
    "kyle_lambda": 0.001,
    "flow_imbalance": 0.3,
    "bid_ask_spread": 0.1
})

# Find similar patterns
similar_patterns = microstructure_agent.memory.find_similar_patterns(
    current_analysis, threshold=0.8
)
```

### AI-Enhanced Signal Generation

The agent uses free AI models to generate sophisticated signals:

```python
# AI analyzes microstructure data and generates signals
signals = await microstructure_agent._generate_ai_signals(analysis)

# Fallback signals when AI is unavailable
fallback_signals = microstructure_agent._generate_fallback_signals(analysis)
```

### Toxicity Scoring

Combined toxicity score based on multiple metrics:

```python
toxicity_score = microstructure_agent._calculate_toxicity_score(
    vpin, kyle_lambda, flow_imbalance
)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for AI analysis | None |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | None |
| `GOOGLE_API_KEY` | Google Gemini API key | None |
| `ALPACA_API_KEY` | Alpaca Markets API key | None |
| `POLYGON_API_KEY` | Polygon.io API key | None |

### Agent Configuration

```python
# Access agent configuration
from src.mcp.config import tool_config

print(tool_config.market_data_cache_ttl)  # 60 seconds
print(tool_config.market_data_max_retries)  # 3
print(tool_config.market_data_timeout)  # 10 seconds
```

## Performance Optimization

### Concurrent Analysis

The agent performs multiple analyses concurrently:

```python
# Run all analyses in parallel
tasks = [
    tools.calculate_vpin(symbol, timeframe),
    tools.measure_price_impact(symbol, timeframe),
    tools.analyze_flow_imbalance(symbol, timeframe),
    tools.detect_flow_patterns(symbol, timeframe),
    tools.analyze_market_depth(symbol)
]

results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Caching Strategy

- **Market Data**: 60-second TTL for real-time data
- **Analysis Results**: 5-minute TTL for computed metrics
- **AI Responses**: 10-minute TTL for AI-generated signals
- **Pattern Memory**: Persistent storage with LRU eviction

### Error Handling

- **API Failures**: Graceful degradation with fallback data
- **Calculation Errors**: Default values and error logging
- **AI Failures**: Fallback to rule-based signal generation
- **Memory Issues**: Automatic cleanup and pattern eviction

## Integration Examples

### MCP Server Integration

```python
# In MCP server tool definitions
async def _calculate_microstructure_features(self, symbol: str, timeframe: str = "1m"):
    analysis = await microstructure_agent.analyze_microstructure(symbol, timeframe)
    return {
        "symbol": analysis.symbol,
        "vpin": analysis.vpin,
        "kyle_lambda": analysis.kyle_lambda,
        "flow_imbalance": analysis.flow_imbalance,
        "toxicity_score": analysis.toxicity_score
    }
```

### Multi-Agent Collaboration

```python
# Create a crew with multiple agents
from crewai import Crew

crew = Crew(
    agents=[
        microstructure_agent.agent,
        sentiment_agent.agent,
        regime_agent.agent
    ],
    tasks=[
        Task("Analyze microstructure", agent=microstructure_agent.agent),
        Task("Analyze sentiment", agent=sentiment_agent.agent),
        Task("Detect regime", agent=regime_agent.agent)
    ],
    process=Process.sequential
)

result = crew.kickoff()
```

### Real-time Monitoring

```python
import asyncio
import schedule
import time

async def monitor_microstructure():
    symbols = ["AAPL", "TSLA", "SPY", "QQQ"]
    
    for symbol in symbols:
        analysis = await microstructure_agent.analyze_microstructure(symbol, "1h")
        signals = await microstructure_agent.generate_signals(symbol, "1h")
        
        print(f"{symbol}: VPIN={analysis.vpin:.3f}, "
              f"Lambda={analysis.kyle_lambda:.6f}, "
              f"Signals={len(signals)}")

# Run every hour
schedule.every().hour.do(lambda: asyncio.run(monitor_microstructure()))

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Testing

### Run Comprehensive Tests

```bash
python src/agents/test_microstructure_agent.py
```

### Test Coverage

The test suite validates:

1. **Algorithm Accuracy**: VPIN, Kyle's Lambda, Lee-Ready calculations
2. **Data Validation**: Input/output validation and error handling
3. **Performance**: Concurrent operations and caching effectiveness
4. **AI Integration**: Signal generation and pattern recognition
5. **Memory System**: Pattern storage and retrieval
6. **Error Handling**: Graceful degradation and fallbacks

### Test Report

Tests generate a comprehensive report:

```json
{
    "timestamp": "2025-01-15T10:30:00Z",
    "test_summary": {
        "total_tests": 7,
        "tests_passed": 7,
        "agent_type": "MicrostructureAgent",
        "framework": "CrewAI"
    },
    "features_tested": [
        "VPIN Calculator",
        "Kyle's Lambda Calculator",
        "Lee-Ready Algorithm",
        "Microstructure Analysis",
        "Signal Generation",
        "Pattern Recognition",
        "Performance Analysis"
    ],
    "algorithms_implemented": [
        "VPIN (Volume-synchronized Probability of Informed Trading)",
        "Kyle's Lambda (Price Impact)",
        "Lee-Ready (Order Flow Classification)",
        "Flow Toxicity Scoring",
        "Market Depth Analysis",
        "Pattern Recognition with Memory"
    ]
}
```

## Troubleshooting

### Common Issues

1. **High VPIN Values**: Normal during high volatility or news events
2. **Low Kyle's Lambda**: Indicates high liquidity, not necessarily an error
3. **AI Signal Failures**: Falls back to rule-based signals automatically
4. **Memory Usage**: Automatic cleanup prevents memory leaks

### Debug Mode

Enable debug logging:

```python
import structlog
structlog.configure(processors=[structlog.processors.ConsoleRenderer()])
```

### Performance Issues

- **Slow Analysis**: Check market data API response times
- **High Memory Usage**: Reduce pattern memory size
- **AI Timeouts**: Increase timeout values or use fallback signals

## Contributing

### Adding New Algorithms

1. **Create Algorithm Class**: Implement new algorithm in separate class
2. **Add to Tools**: Integrate with MicrostructureTools
3. **Create Tool**: Add CrewAI tool for the algorithm
4. **Update Tests**: Add comprehensive tests
5. **Update Documentation**: Document algorithm and usage

### Extending Signal Types

1. **Define Signal Type**: Add new signal type to MicrostructureSignal
2. **Implement Logic**: Add signal generation logic
3. **AI Integration**: Update AI prompt for new signals
4. **Performance Tracking**: Add performance metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting guide
- Review the test examples
- Create an issue on GitHub
- Check the CrewAI documentation 