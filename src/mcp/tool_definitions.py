"""
Tool Definitions for MCP Server

Comprehensive tool definitions using the latest OpenAI function calling format
with detailed schemas for all financial analysis tools.
"""

from typing import Dict, List, Any


class ToolDefinitions:
    """
    Tool definitions for the Alpha Discovery MCP Server.
    
    Provides standardized tool definitions that can be used by any AI model
    supporting OpenAI function calling format.
    """
    
    def __init__(self):
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> Dict[str, Dict[str, Any]]:
        """Initialize all tool definitions"""
        return {
            "get_orderbook": {
                "type": "function",
                "function": {
                    "name": "get_orderbook",
                    "description": "Get real-time order book data for a trading symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol (e.g., 'AAPL', 'TSLA', 'SPY')",
                                "examples": ["AAPL", "TSLA", "SPY", "QQQ"]
                            },
                            "exchange": {
                                "type": "string",
                                "description": "Exchange to get data from",
                                "enum": ["alpaca", "binance", "polygon"],
                                "default": "alpaca"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "get_reddit_sentiment": {
                "type": "function",
                "function": {
                    "name": "get_reddit_sentiment",
                    "description": "Get sentiment analysis from Reddit for a trading symbol",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to analyze",
                                "examples": ["AAPL", "TSLA", "GME", "AMC"]
                            },
                            "timeframe": {
                                "type": "string",
                                "description": "Time window for sentiment analysis",
                                "enum": ["1h", "4h", "1d", "1w"],
                                "default": "1h"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "calculate_microstructure_features": {
                "type": "function",
                "function": {
                    "name": "calculate_microstructure_features",
                    "description": "Calculate market microstructure features including bid-ask spread, order imbalance, and liquidity metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to analyze",
                                "examples": ["AAPL", "TSLA", "SPY"]
                            },
                            "timeframe": {
                                "type": "string",
                                "description": "Analysis timeframe",
                                "enum": ["1m", "5m", "15m", "1h"],
                                "default": "1m"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "detect_regime_change": {
                "type": "function",
                "function": {
                    "name": "detect_regime_change",
                    "description": "Detect market regime changes using AI analysis of price patterns, volatility, and trend strength",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to analyze",
                                "examples": ["SPY", "QQQ", "IWM", "AAPL"]
                            },
                            "lookback_days": {
                                "type": "integer",
                                "description": "Number of days to analyze for regime detection",
                                "minimum": 7,
                                "maximum": 365,
                                "default": 30
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "get_market_data": {
                "type": "function",
                "function": {
                    "name": "get_market_data",
                    "description": "Get comprehensive market data including price, volume, and basic indicators",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol",
                                "examples": ["AAPL", "TSLA", "SPY", "QQQ"]
                            },
                            "timeframe": {
                                "type": "string",
                                "description": "Data timeframe",
                                "enum": ["1m", "5m", "15m", "1h", "1d", "1w"],
                                "default": "1d"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "analyze_order_flow": {
                "type": "function",
                "function": {
                    "name": "analyze_order_flow",
                    "description": "Analyze order flow patterns and detect unusual activity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to analyze",
                                "examples": ["AAPL", "TSLA", "SPY"]
                            },
                            "timeframe": {
                                "type": "string",
                                "description": "Analysis timeframe",
                                "enum": ["1m", "5m", "15m", "1h"],
                                "default": "1m"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "get_technical_indicators": {
                "type": "function",
                "function": {
                    "name": "get_technical_indicators",
                    "description": "Get technical indicators including RSI, MACD, Bollinger Bands, and moving averages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol",
                                "examples": ["AAPL", "TSLA", "SPY", "QQQ"]
                            },
                            "timeframe": {
                                "type": "string",
                                "description": "Indicator timeframe",
                                "enum": ["1m", "5m", "15m", "1h", "1d"],
                                "default": "1d"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "stream_market_data": {
                "type": "function",
                "function": {
                    "name": "stream_market_data",
                    "description": "Stream real-time market data for a specified duration",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to stream",
                                "examples": ["AAPL", "TSLA", "SPY"]
                            },
                            "duration_seconds": {
                                "type": "integer",
                                "description": "Duration to stream data in seconds",
                                "minimum": 10,
                                "maximum": 3600,
                                "default": 60
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "get_news_sentiment": {
                "type": "function",
                "function": {
                    "name": "get_news_sentiment",
                    "description": "Get sentiment analysis from news sources and earnings reports",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to analyze",
                                "examples": ["AAPL", "TSLA", "MSFT", "GOOGL"]
                            },
                            "timeframe": {
                                "type": "string",
                                "description": "Time window for news analysis",
                                "enum": ["1h", "4h", "1d", "1w"],
                                "default": "1h"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "calculate_risk_metrics": {
                "type": "function",
                "function": {
                    "name": "calculate_risk_metrics",
                    "description": "Calculate comprehensive risk metrics including VaR, volatility, and position sizing recommendations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to analyze",
                                "examples": ["AAPL", "TSLA", "SPY", "QQQ"]
                            },
                            "portfolio_value": {
                                "type": "number",
                                "description": "Portfolio value for position sizing calculations",
                                "minimum": 1000,
                                "default": 100000
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "get_alternative_data": {
                "type": "function",
                "function": {
                    "name": "get_alternative_data",
                    "description": "Get alternative data sources including satellite imagery, credit card data, and social media trends",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to analyze",
                                "examples": ["TSLA", "AMZN", "WMT", "MCD"]
                            },
                            "data_type": {
                                "type": "string",
                                "description": "Type of alternative data to retrieve",
                                "enum": ["satellite", "credit_card", "social_media", "earnings_calls"],
                                "default": "social_media"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "analyze_correlation": {
                "type": "function",
                "function": {
                    "name": "analyze_correlation",
                    "description": "Analyze correlation between multiple assets and detect regime-dependent relationships",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of trading symbols to analyze",
                                "minItems": 2,
                                "maxItems": 10,
                                "examples": [["SPY", "QQQ"], ["AAPL", "TSLA", "MSFT"]]
                            },
                            "timeframe": {
                                "type": "string",
                                "description": "Analysis timeframe",
                                "enum": ["1d", "1w", "1m", "3m"],
                                "default": "1m"
                            }
                        },
                        "required": ["symbols"]
                    }
                }
            },
            
            "detect_anomalies": {
                "type": "function",
                "function": {
                    "name": "detect_anomalies",
                    "description": "Detect anomalous patterns in price, volume, or sentiment data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to analyze",
                                "examples": ["AAPL", "TSLA", "GME", "AMC"]
                            },
                            "anomaly_type": {
                                "type": "string",
                                "description": "Type of anomaly to detect",
                                "enum": ["price", "volume", "sentiment", "order_flow"],
                                "default": "price"
                            },
                            "sensitivity": {
                                "type": "number",
                                "description": "Anomaly detection sensitivity (0.1 to 1.0)",
                                "minimum": 0.1,
                                "maximum": 1.0,
                                "default": 0.5
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "forecast_volatility": {
                "type": "function",
                "function": {
                    "name": "forecast_volatility",
                    "description": "Forecast volatility using GARCH models and market regime analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to forecast",
                                "examples": ["SPY", "QQQ", "VIX", "AAPL"]
                            },
                            "forecast_days": {
                                "type": "integer",
                                "description": "Number of days to forecast",
                                "minimum": 1,
                                "maximum": 30,
                                "default": 5
                            },
                            "confidence_level": {
                                "type": "number",
                                "description": "Confidence level for volatility bands",
                                "minimum": 0.5,
                                "maximum": 0.99,
                                "default": 0.95
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            },
            
            "optimize_portfolio": {
                "type": "function",
                "function": {
                    "name": "optimize_portfolio",
                    "description": "Optimize portfolio allocation using modern portfolio theory and risk metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of assets to include in portfolio",
                                "minItems": 2,
                                "maxItems": 20,
                                "examples": [["SPY", "QQQ", "IWM"], ["AAPL", "TSLA", "MSFT", "GOOGL"]]
                            },
                            "portfolio_value": {
                                "type": "number",
                                "description": "Total portfolio value",
                                "minimum": 1000,
                                "default": 100000
                            },
                            "risk_tolerance": {
                                "type": "string",
                                "description": "Risk tolerance level",
                                "enum": ["conservative", "moderate", "aggressive"],
                                "default": "moderate"
                            },
                            "optimization_method": {
                                "type": "string",
                                "description": "Portfolio optimization method",
                                "enum": ["sharpe_ratio", "min_variance", "max_diversification", "black_litterman"],
                                "default": "sharpe_ratio"
                            }
                        },
                        "required": ["symbols"]
                    }
                }
            }
        }
    
    def get_tool_definition(self, tool_name: str) -> Dict[str, Any]:
        """Get tool definition by name"""
        return self.tools.get(tool_name, {})
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tool definitions"""
        return list(self.tools.values())
    
    def get_tool_names(self) -> List[str]:
        """Get list of all available tool names"""
        return list(self.tools.keys())
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get tools by category"""
        categories = {
            "market_data": ["get_orderbook", "get_market_data", "get_technical_indicators"],
            "sentiment": ["get_reddit_sentiment", "get_news_sentiment"],
            "analysis": ["calculate_microstructure_features", "analyze_order_flow", "detect_anomalies"],
            "prediction": ["detect_regime_change", "forecast_volatility"],
            "risk": ["calculate_risk_metrics", "optimize_portfolio"],
            "streaming": ["stream_market_data"],
            "alternative": ["get_alternative_data", "analyze_correlation"]
        }
        
        tool_names = categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_openai_format(self) -> List[Dict[str, Any]]:
        """Get tools in OpenAI function calling format"""
        return self.get_all_tools()
    
    def get_anthropic_format(self) -> List[Dict[str, Any]]:
        """Get tools in Anthropic Claude format"""
        tools = []
        for tool_name, tool_def in self.tools.items():
            function_def = tool_def["function"]
            tools.append({
                "name": function_def["name"],
                "description": function_def["description"],
                "input_schema": function_def["parameters"]
            })
        return tools
    
    def get_gemini_format(self) -> List[Dict[str, Any]]:
        """Get tools in Google Gemini format"""
        tools = []
        for tool_name, tool_def in self.tools.items():
            function_def = tool_def["function"]
            tools.append({
                "name": function_def["name"],
                "description": function_def["description"],
                "parameters": function_def["parameters"]
            })
        return tools
    
    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Validate tool call arguments"""
        if tool_name not in self.tools:
            return False
        
        tool_def = self.tools[tool_name]
        required_params = tool_def["function"]["parameters"].get("required", [])
        
        # Check required parameters
        for param in required_params:
            if param not in arguments:
                return False
        
        return True
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get JSON schema for a tool"""
        if tool_name not in self.tools:
            return {}
        
        tool_def = self.tools[tool_name]
        return tool_def["function"]["parameters"]
    
    def get_tool_description(self, tool_name: str) -> str:
        """Get description for a tool"""
        if tool_name not in self.tools:
            return ""
        
        tool_def = self.tools[tool_name]
        return tool_def["function"]["description"]


# Global tool definitions instance
tool_definitions = ToolDefinitions() 