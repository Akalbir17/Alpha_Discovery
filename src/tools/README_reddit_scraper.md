# Reddit Scraper Tool

A state-of-the-art Reddit scraper for financial sentiment analysis and trend detection, built with the latest free AI models and modern NLP techniques.

## Features

- **Multi-Subreddit Monitoring**: Monitors r/wallstreetbets, r/stocks, r/investing, r/SecurityAnalysis, r/quant
- **Advanced Sentiment Analysis**: Uses transformers (RoBERTa) with TextBlob fallback
- **Trend Detection**: Identifies trending tickers with velocity tracking
- **Unusual Activity Detection**: Detects options activity, sentiment spikes, and volume anomalies
- **Redis Caching**: Intelligent caching with TTL for performance
- **Rate Limiting**: Respects Reddit API limits and implements smart throttling
- **AI Integration**: Uses free AI models for enhanced analysis
- **Real-time Processing**: Async/await architecture for high performance

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Reddit API (optional - will use mock data if not provided)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=AlphaDiscovery/1.0

# Redis (optional - caching disabled if not provided)
REDIS_URL=redis://localhost:6379

# AI Models (for enhanced analysis)
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

### 3. Basic Usage

```python
import asyncio
from src.scrapers.reddit_scraper import RedditScraper

async def main():
    # Initialize the scraper
    reddit_scraper = RedditScraper()
    await reddit_scraper.initialize()
    
    # Get sentiment for a symbol
    sentiment = await reddit_scraper.get_sentiment("TSLA", timeframe="1h")
    print(f"TSLA sentiment: {sentiment['sentiment_score']:.2f}")
    
    # Get trending tickers
    trending = await reddit_scraper.get_trending_tickers(hours_back=24)
    for ticker in trending[:5]:
        print(f"{ticker.symbol}: {ticker.mention_count} mentions")

asyncio.run(main())
```

### 4. Run Tests

```bash
python src/tools/test_reddit_scraper.py
```

## API Reference

### Core Methods

#### `get_sentiment(symbol: str, timeframe: str = "1h")`

Get sentiment analysis for a specific trading symbol.

**Parameters:**
- `symbol`: Trading symbol (e.g., "AAPL", "TSLA")
- `timeframe`: Time window ("1h", "4h", "1d", "1w")

**Returns:**
```json
{
    "symbol": "TSLA",
    "sentiment_score": 0.75,
    "sentiment_label": "positive",
    "mention_count": 45,
    "confidence": 0.85,
    "timeframe": "1h",
    "timestamp": "2025-01-15T10:30:00Z",
    "sentiment_distribution": {
        "positive": 30,
        "neutral": 10,
        "negative": 5
    }
}
```

#### `get_trending_tickers(hours_back: int = 24)`

Get trending tickers with sentiment analysis and velocity tracking.

**Parameters:**
- `hours_back`: Number of hours to analyze

**Returns:**
```json
[
    {
        "symbol": "GME",
        "mention_count": 150,
        "sentiment_score": 0.8,
        "sentiment_change": 0.3,
        "velocity": 6.25,
        "subreddits": ["wallstreetbets", "stocks"],
        "top_posts": [...],
        "unusual_activity": true,
        "trend_strength": 0.85
    }
]
```

#### `get_sentiment_history(symbol: str, days: int = 7)`

Get historical sentiment data for a symbol.

**Parameters:**
- `symbol`: Trading symbol
- `days`: Number of days to analyze

**Returns:**
```json
[
    {
        "date": "2025-01-14",
        "sentiment_score": 0.65,
        "mention_count": 45,
        "posts_count": 12
    }
]
```

#### `get_unusual_activity(hours_back: int = 24)`

Detect unusual activity patterns in Reddit discussions.

**Parameters:**
- `hours_back`: Number of hours to analyze

**Returns:**
```json
[
    {
        "symbol": "TSLA",
        "activity_type": "options_activity",
        "confidence": 0.85,
        "description": "Unusual options activity: 15 mentions",
        "evidence": ["Options mentions: 15", "Options: [{'symbol': 'TSLA', 'expiry': '1/15', 'strike': '150', 'type': 'C'}]"],
        "timestamp": "2025-01-15T10:30:00Z"
    }
]
```

#### `scrape_subreddits(hours_back: int = 24)`

Scrape posts from configured subreddits.

**Parameters:**
- `hours_back`: Number of hours to look back

**Returns:**
```json
[
    {
        "id": "post_id",
        "title": "TSLA to the moon! 🚀",
        "body": "Just bought more TSLA calls...",
        "author": "diamond_hands",
        "subreddit": "wallstreetbets",
        "score": 1500,
        "num_comments": 234,
        "created_utc": 1705312200.0,
        "url": "https://reddit.com/r/wallstreetbets/post_id",
        "tickers": ["TSLA"],
        "sentiment_score": 0.8,
        "sentiment_label": "positive",
        "confidence": 0.9
    }
]
```

## Advanced Features

### Sentiment Analysis

The scraper uses a multi-layered sentiment analysis approach:

1. **Transformers Pipeline**: Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` for high accuracy
2. **TextBlob Fallback**: Falls back to TextBlob if transformers fail
3. **Confidence Scoring**: Provides confidence scores for sentiment predictions
4. **Sentiment Distribution**: Breaks down sentiment into positive/neutral/negative ratios

### Trend Detection

Advanced trend detection algorithms:

- **Velocity Tracking**: Mentions per hour to identify accelerating interest
- **Sentiment Change**: Compare current vs historical sentiment
- **Cross-Subreddit Analysis**: Track mentions across multiple subreddits
- **Trend Strength**: Composite score based on velocity, sentiment, and volume

### Unusual Activity Detection

Detects various types of unusual patterns:

- **High Velocity**: Unusually high mention rates
- **Extreme Sentiment**: Very positive or negative sentiment spikes
- **Sentiment Volatility**: High sentiment variance
- **Options Activity**: Unusual options mentions and patterns
- **AI Enhancement**: Uses Claude for pattern analysis

### Caching Strategy

Intelligent Redis caching with TTL:

- **Post Cache**: 5 minutes for recent posts
- **Sentiment Cache**: 5 minutes for sentiment analysis
- **Trending Cache**: 10 minutes for trending tickers
- **History Cache**: 1 hour for historical data
- **Unusual Activity Cache**: 15 minutes for activity detection

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDDIT_CLIENT_ID` | Reddit API client ID | None (mock data) |
| `REDDIT_CLIENT_SECRET` | Reddit API client secret | None (mock data) |
| `REDDIT_USER_AGENT` | Reddit API user agent | "AlphaDiscovery/1.0" |
| `REDIS_URL` | Redis connection URL | None (no caching) |
| `REDDIT_RATE_LIMIT` | Requests per minute | 60 |
| `REDDIT_MAX_POSTS` | Max posts per subreddit | 100 |
| `SENTIMENT_CACHE_TTL` | Sentiment cache TTL (seconds) | 300 |

### Tool Configuration

```python
from src.mcp.config import tool_config

# Access configuration
print(tool_config.reddit_subreddits)  # ['wallstreetbets', 'stocks', 'investing', ...]
print(tool_config.reddit_rate_limit)  # 60
print(tool_config.sentiment_cache_ttl)  # 300
```

## Performance Optimization

### Rate Limiting

- **Reddit API**: Respects Reddit's rate limits
- **Internal Throttling**: Ensures minimum intervals between requests
- **Request Counting**: Tracks request frequency
- **Graceful Degradation**: Falls back to mock data if rate limited

### Caching Strategy

- **Multi-level Caching**: Redis + in-memory caching
- **TTL Management**: Different TTLs for different data types
- **Cache Invalidation**: Automatic cache expiration
- **Cache Warming**: Pre-loads frequently accessed data

### Async Processing

- **Concurrent Operations**: Multiple operations run simultaneously
- **Non-blocking I/O**: Uses async/await for all I/O operations
- **Resource Management**: Efficient connection pooling
- **Error Recovery**: Automatic retries and fallbacks

## Error Handling

### Graceful Degradation

- **API Failures**: Falls back to mock data
- **Network Issues**: Automatic retries with exponential backoff
- **Rate Limiting**: Respects limits and waits appropriately
- **Cache Failures**: Continues without caching

### Error Types

- **Reddit API Errors**: Handled with fallback to mock data
- **Redis Errors**: Logged but doesn't stop operation
- **Sentiment Analysis Errors**: Falls back to TextBlob
- **Network Timeouts**: Retried with backoff

## Testing

### Run Comprehensive Tests

```bash
python src/tools/test_reddit_scraper.py
```

### Test Coverage

The test suite covers:

1. **Basic Functionality**: Scraping, sentiment analysis, trend detection
2. **Error Handling**: API failures, network issues, invalid inputs
3. **Performance**: Concurrent operations, caching effectiveness
4. **Data Quality**: Ticker extraction, options detection, sentiment accuracy
5. **Integration**: Redis caching, AI model integration

### Test Report

Tests generate a comprehensive report:

```json
{
    "timestamp": "2025-01-15T10:30:00Z",
    "summary": {
        "posts_scraped": 150,
        "trending_tickers": 25,
        "unusual_activities": 8,
        "subreddits_monitored": ["wallstreetbets", "stocks", "investing"]
    },
    "top_trending_tickers": [...],
    "unusual_activities_summary": [...],
    "sentiment_distribution": {
        "positive": 90,
        "neutral": 40,
        "negative": 20
    }
}
```

## Integration Examples

### MCP Server Integration

```python
# In MCP server tool definitions
async def _get_reddit_sentiment(self, symbol: str, timeframe: str = "1h"):
    return await reddit_scraper.get_sentiment(symbol, timeframe)
```

### Agent Integration

```python
# In agent implementations
from src.scrapers.reddit_scraper import RedditScraper

class SentimentAgent:
    def __init__(self):
        self.reddit_scraper = RedditScraper()
    
    async def analyze_sentiment(self, symbol: str):
        sentiment = await self.reddit_scraper.get_sentiment(symbol)
        trending = await self.reddit_scraper.get_trending_tickers()
        unusual = await self.reddit_scraper.get_unusual_activity()
        
        return {
            "sentiment": sentiment,
            "trending": trending,
            "unusual_activity": unusual
        }
```

### Real-time Monitoring

```python
import asyncio
import schedule
import time

async def monitor_sentiment():
    symbols = ["TSLA", "AAPL", "GME", "SPY"]
    
    for symbol in symbols:
        sentiment = await reddit_scraper.get_sentiment(symbol, "1h")
        print(f"{symbol}: {sentiment['sentiment_score']:.2f}")

# Run every hour
schedule.every().hour.do(lambda: asyncio.run(monitor_sentiment()))

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Troubleshooting

### Common Issues

1. **No Reddit API Credentials**: Tool will use mock data automatically
2. **Redis Connection Failed**: Caching disabled, continues without cache
3. **Rate Limit Exceeded**: Tool waits and retries automatically
4. **Sentiment Analysis Failed**: Falls back to TextBlob

### Debug Mode

Enable debug logging:

```python
import structlog
structlog.configure(processors=[structlog.processors.ConsoleRenderer()])
```

### Performance Issues

- **Slow Response**: Check Redis connection and cache hit rates
- **High Memory Usage**: Reduce `max_posts` configuration
- **Rate Limiting**: Increase `rate_limit` or reduce request frequency

## Contributing

### Adding New Features

1. **New Subreddits**: Add to `tool_config.reddit_subreddits`
2. **New Sentiment Models**: Extend `_init_sentiment_analysis()`
3. **New Activity Types**: Add to `get_unusual_activity()`
4. **New Caching Strategies**: Extend caching methods

### Testing New Features

1. Add unit tests for new functionality
2. Update integration tests
3. Run performance benchmarks
4. Update documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting guide
- Review the test examples
- Create an issue on GitHub
- Check the MCP server integration 