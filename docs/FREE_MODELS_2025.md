# Free AI Models & Tools Guide - 2025

This document outlines the latest state-of-the-art free AI models and tools used in the Alpha Discovery Platform.

## 🤖 LLM Providers (Free Tiers)

### 1. Groq (Ultra-Fast Inference)
- **Model**: Llama 3.1 8B
- **Speed**: ~800 tokens/second
- **Free Tier**: 100 requests/day
- **Best For**: Real-time analysis, low-latency applications
- **Use Case**: Microstructure analysis, real-time signal processing

### 2. Google Gemini (Multimodal)
- **Model**: Gemini Pro
- **Features**: Text, image, code understanding
- **Free Tier**: 15 requests/minute
- **Best For**: Multimodal analysis, code generation
- **Use Case**: Alternative data analysis, chart pattern recognition

### 3. Anthropic Claude (Reasoning)
- **Model**: Claude 3 Haiku
- **Features**: Advanced reasoning, safety-focused
- **Free Tier**: 5 requests/minute
- **Best For**: Complex reasoning, risk assessment
- **Use Case**: Regime detection, risk management

### 4. Together AI (Open Source)
- **Model**: Llama 3.1 70B
- **Features**: Largest open-source model
- **Free Tier**: 100 requests/day
- **Best For**: Complex analysis, strategy generation
- **Use Case**: Strategy agent, consensus analysis

### 5. Fireworks AI (Specialized)
- **Model**: Llama 3.1 8B
- **Features**: Fine-tuned for specific tasks
- **Free Tier**: 100 requests/day
- **Best For**: Specialized financial analysis
- **Use Case**: Technical analysis, pattern recognition

### 6. Mistral AI (Efficient)
- **Model**: Mistral 7B
- **Features**: Efficient, multilingual
- **Free Tier**: 100 requests/day
- **Best For**: Multilingual sentiment analysis
- **Use Case**: Global market sentiment

## 📊 Financial Data APIs (Free Tiers)

### 1. Polygon.io
- **Free Tier**: 5 requests/minute
- **Data**: Real-time stock data, options, forex
- **Use Case**: Real-time market data

### 2. Finnhub
- **Free Tier**: 60 requests/minute
- **Data**: Stock data, news, sentiment
- **Use Case**: News sentiment analysis

### 3. Alpha Vantage
- **Free Tier**: 500 requests/day
- **Data**: Technical indicators, fundamental data
- **Use Case**: Technical analysis

### 4. IEX Cloud
- **Free Tier**: 50,000 messages/month
- **Data**: Real-time and historical data
- **Use Case**: Historical backtesting

### 5. Quandl
- **Free Tier**: 50,000 requests/month
- **Data**: Alternative data, economic indicators
- **Use Case**: Alternative data analysis

## 🛠️ Modern Tools & Libraries

### Data Processing
- **Polars**: Fast DataFrame library (Rust-based)
- **Vaex**: Big data processing
- **Dask**: Parallel computing
- **Xarray**: Multi-dimensional arrays

### Web Scraping
- **Scrapy**: Advanced web scraping framework
- **Playwright**: Modern browser automation
- **Newspaper3k**: News article extraction
- **YouTube Transcript API**: Video content analysis

### Monitoring & Logging
- **Sentry**: Error tracking and monitoring
- **Loguru**: Modern logging framework

### CLI & UI
- **Typer**: Modern CLI framework
- **Textual**: Terminal UI framework
- **Rich**: Rich text and formatting

## 🔄 Model Rotation Strategy

To maximize free tier usage and ensure reliability:

### Primary Models (High Priority)
1. **Groq** - Real-time analysis
2. **Google Gemini** - Multimodal tasks
3. **Anthropic Claude** - Complex reasoning

### Secondary Models (Backup)
1. **Together AI** - Large model tasks
2. **Fireworks AI** - Specialized analysis
3. **Mistral AI** - Multilingual tasks

### Fallback Strategy
- Local Ollama models for offline operation
- Model switching based on rate limits
- Caching responses to minimize API calls

## 📈 Performance Comparison

| Model | Speed | Quality | Cost | Best Use |
|-------|-------|---------|------|----------|
| Groq | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | Real-time |
| Gemini | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Free | Multimodal |
| Claude | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Free | Reasoning |
| Together | ⭐⭐ | ⭐⭐⭐⭐⭐ | Free | Complex |
| Fireworks | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | Specialized |
| Mistral | ⭐⭐⭐⭐ | ⭐⭐⭐ | Free | Efficient |

## 🚀 Getting Started

### 1. Sign Up for Free APIs
```bash
# Groq (Fastest)
curl -X POST https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1-8b-8192", "messages": [{"role": "user", "content": "Hello"}]}'

# Google Gemini
curl -X POST https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent \
  -H "Authorization: Bearer $GOOGLE_AI_API_KEY" \
  -d '{"contents": [{"parts": [{"text": "Hello"}]}]}'

# Anthropic Claude
curl -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-haiku-20240307", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}'
```

### 2. Environment Setup
```bash
# Copy environment template
cp env.example .env

# Add your free API keys
GROQ_API_KEY=your_groq_key_here
GOOGLE_AI_API_KEY=your_google_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
TOGETHER_API_KEY=your_together_key_here
FIREWORKS_API_KEY=your_fireworks_key_here
MISTRAL_API_KEY=your_mistral_key_here
```

### 3. Test Models
```python
from src.agents.microstructure_agent import MicrostructureAgent
from src.agents.altdata_agent import AltDataAgent

# Test Groq (fastest)
micro_agent = MicrostructureAgent()
analysis = await micro_agent.analyze_microstructure("AAPL")

# Test Gemini (multimodal)
alt_agent = AltDataAgent()
sentiment = await alt_agent.analyze_sentiment("TSLA")
```

## 💡 Best Practices

### Rate Limiting
- Implement exponential backoff
- Use multiple API keys
- Cache responses when possible
- Monitor usage limits

### Model Selection
- Use fastest model for real-time tasks
- Use largest model for complex analysis
- Use specialized models for specific tasks
- Implement fallback chains

### Cost Optimization
- Batch requests when possible
- Use streaming for long responses
- Implement smart caching
- Monitor API usage

## 🔮 Future Updates

This guide will be updated as new free models become available. Current trends:

- **Local Models**: Ollama, LM Studio
- **Edge Computing**: On-device inference
- **Specialized Models**: Finance-specific fine-tuning
- **Multimodal**: Image, audio, video analysis

## 📚 Resources

- [Groq Documentation](https://console.groq.com/docs)
- [Google AI Studio](https://aistudio.google.com/)
- [Anthropic Console](https://console.anthropic.com/)
- [Together AI](https://together.ai/)
- [Fireworks AI](https://fireworks.ai/)
- [Mistral AI](https://mistral.ai/)

---

*Last updated: January 2025*
*Free tier limits subject to change* 