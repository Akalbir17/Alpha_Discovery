"""
Model Manager - 2025 State-of-the-Art Edition

Manages different free LLM models with intelligent fallback strategies,
rate limiting, and optimal model selection based on task requirements.
Updated for 2025 with latest models and improved performance.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import time
import random
import json
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types - 2025 Edition"""
    GROQ = "groq"
    GEMINI = "gemini"
    CLAUDE = "claude"
    TOGETHER = "together"
    MISTRAL = "mistral"
    OLLAMA = "ollama"


class TaskType(Enum):
    """Task types for model selection - Enhanced for 2025"""
    REALTIME = "realtime"          # Ultra-fast responses needed
    MULTIMODAL = "multimodal"      # Text + images/data
    REASONING = "reasoning"        # Complex logical reasoning
    COMPLEX = "complex"            # Multi-step complex tasks
    SPECIALIZED = "specialized"    # Domain-specific tasks
    EFFICIENT = "efficient"        # Resource-efficient tasks
    CREATIVE = "creative"          # Creative content generation
    ANALYTICAL = "analytical"      # Data analysis and insights
    FINANCIAL = "financial"        # Financial domain tasks


@dataclass
class ModelConfig:
    """Configuration for each model"""
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 30.0
    supports_streaming: bool = True
    supports_functions: bool = True
    cost_per_token: float = 0.0  # For free models


@dataclass
class ModelResponse:
    """Enhanced model response with metadata"""
    content: str
    model_used: str
    response_time: float
    timestamp: datetime
    tokens_used: int = 0
    cost: float = 0.0
    confidence: float = 1.0
    cached: bool = False
    error: Optional[str] = None


class ModelManager:
    """
    Enhanced Model Manager for 2025 - State-of-the-Art Features:
    
    - Latest free LLM models with optimal configurations
    - Intelligent model routing based on task characteristics
    - Advanced caching with TTL and LRU eviction
    - Performance monitoring and analytics
    - Automatic failover and circuit breaker patterns
    - Concurrent request handling
    - Smart rate limiting with backoff strategies
    """
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.rate_limits = {}
        self.usage_tracking = {}
        self.response_cache = {}
        self.cache_lock = threading.Lock()
        self.performance_metrics = {}
        self.circuit_breakers = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        self._initialize_models()
        self._setup_rate_limits()
        self._setup_circuit_breakers()
        
    def _initialize_models(self):
        """Initialize all available models with 2025 configurations"""
        try:
            # Groq - Fastest for real-time tasks (2025 models)
            if os.getenv('GROQ_API_KEY'):
                self.model_configs[ModelType.GROQ] = ModelConfig(
                    model_name="llama-3.3-70b-versatile",  # Latest Llama 3.3
                    max_tokens=8192,
                    temperature=0.1,
                    supports_streaming=True,
                    supports_functions=True
                )
                self.models[ModelType.GROQ] = ChatGroq(
                    model=self.model_configs[ModelType.GROQ].model_name,
                    temperature=self.model_configs[ModelType.GROQ].temperature,
                    max_tokens=self.model_configs[ModelType.GROQ].max_tokens,
                    groq_api_key=os.getenv('GROQ_API_KEY')
                )
                logger.info("Groq Llama 3.3 70B initialized")
            
            # Google Gemini - Best for multimodal tasks (2025 models)
            if os.getenv('GOOGLE_AI_API_KEY'):
                self.model_configs[ModelType.GEMINI] = ModelConfig(
                    model_name="gemini-2.0-flash-exp",  # Latest Gemini 2.0
                    max_tokens=8192,
                    temperature=0.1,
                    supports_streaming=True,
                    supports_functions=True
                )
                self.models[ModelType.GEMINI] = ChatGoogleGenerativeAI(
                    model=self.model_configs[ModelType.GEMINI].model_name,
                    temperature=self.model_configs[ModelType.GEMINI].temperature,
                    max_output_tokens=self.model_configs[ModelType.GEMINI].max_tokens,
                    google_api_key=os.getenv('GOOGLE_AI_API_KEY')
                )
                logger.info("Google Gemini 2.0 Flash initialized")
            
            # Anthropic Claude - Best for reasoning (2025 models)
            if os.getenv('ANTHROPIC_API_KEY'):
                self.model_configs[ModelType.CLAUDE] = ModelConfig(
                    model_name="claude-3-5-sonnet-20241022",  # Latest Claude 3.5
                    max_tokens=8192,
                    temperature=0.1,
                    supports_streaming=True,
                    supports_functions=True
                )
                self.models[ModelType.CLAUDE] = ChatAnthropic(
                    model=self.model_configs[ModelType.CLAUDE].model_name,
                    temperature=self.model_configs[ModelType.CLAUDE].temperature,
                    max_tokens=self.model_configs[ModelType.CLAUDE].max_tokens,
                    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
                )
                logger.info("Anthropic Claude 3.5 Sonnet initialized")
            
            # Together AI - Largest models for complex tasks (2025 models)
            if os.getenv('TOGETHER_API_KEY'):
                self.model_configs[ModelType.TOGETHER] = ModelConfig(
                    model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",  # Latest 405B
                    max_tokens=8192,
                    temperature=0.1,
                    supports_streaming=True,
                    supports_functions=True
                )
                self.models[ModelType.TOGETHER] = ChatTogether(
                    model=self.model_configs[ModelType.TOGETHER].model_name,
                    temperature=self.model_configs[ModelType.TOGETHER].temperature,
                    max_tokens=self.model_configs[ModelType.TOGETHER].max_tokens,
                    together_api_key=os.getenv('TOGETHER_API_KEY')
                )
                logger.info("Together AI Llama 3.1 405B initialized")
            
            # Mistral AI - Efficient multilingual (2025 models)
            if os.getenv('MISTRAL_API_KEY'):
                self.model_configs[ModelType.MISTRAL] = ModelConfig(
                    model_name="mistral-large-2411",  # Latest Mistral Large
                    max_tokens=8192,
                    temperature=0.1,
                    supports_streaming=True,
                    supports_functions=True
                )
                self.models[ModelType.MISTRAL] = ChatMistralAI(
                    model=self.model_configs[ModelType.MISTRAL].model_name,
                    temperature=self.model_configs[ModelType.MISTRAL].temperature,
                    max_tokens=self.model_configs[ModelType.MISTRAL].max_tokens,
                    mistral_api_key=os.getenv('MISTRAL_API_KEY')
                )
                logger.info("Mistral Large 2411 initialized")
            
            # Ollama - Local models (2025 models)
            try:
                self.model_configs[ModelType.OLLAMA] = ModelConfig(
                    model_name="llama3.3:70b",  # Latest Llama 3.3 70B local
                    max_tokens=8192,
                    temperature=0.1,
                    supports_streaming=True,
                    supports_functions=False,  # Limited function support
                    timeout=60.0  # Longer timeout for local models
                )
                self.models[ModelType.OLLAMA] = ChatOllama(
                    model=self.model_configs[ModelType.OLLAMA].model_name,
                    temperature=self.model_configs[ModelType.OLLAMA].temperature,
                    num_predict=self.model_configs[ModelType.OLLAMA].max_tokens
                )
                logger.info("Ollama Llama 3.3 70B initialized")
            except Exception as e:
                logger.warning(f"Ollama not available: {e}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _setup_rate_limits(self):
        """Setup enhanced rate limits for 2025 models"""
        self.rate_limits = {
            ModelType.GROQ: {
                "requests_per_day": 14400,  # Updated Groq limits
                "requests_per_minute": 30,
                "tokens_per_minute": 30000,
                "concurrent_requests": 5
            },
            ModelType.GEMINI: {
                "requests_per_day": 1500,  # Updated Gemini limits
                "requests_per_minute": 15,
                "tokens_per_minute": 32000,
                "concurrent_requests": 3
            },
            ModelType.CLAUDE: {
                "requests_per_day": 1000,  # Updated Claude limits
                "requests_per_minute": 5,
                "tokens_per_minute": 40000,
                "concurrent_requests": 2
            },
            ModelType.TOGETHER: {
                "requests_per_day": 200,  # Updated Together limits
                "requests_per_minute": 10,
                "tokens_per_minute": 20000,
                "concurrent_requests": 3
            },
            ModelType.MISTRAL: {
                "requests_per_day": 1000,  # Updated Mistral limits
                "requests_per_minute": 10,
                "tokens_per_minute": 25000,
                "concurrent_requests": 3
            },
            ModelType.OLLAMA: {
                "requests_per_day": 100000,  # Local model - high limits
                "requests_per_minute": 100,
                "tokens_per_minute": 100000,
                "concurrent_requests": 10
            }
        }
        
        # Initialize enhanced usage tracking
        for model_type in self.rate_limits:
            self.usage_tracking[model_type] = {
                "daily_requests": 0,
                "minute_requests": 0,
                "tokens_used": 0,
                "concurrent_requests": 0,
                "last_request": None,
                "last_minute_reset": datetime.now(),
                "last_day_reset": datetime.now()
            }
    
    def _setup_circuit_breakers(self):
        """Setup circuit breakers for fault tolerance"""
        for model_type in ModelType:
            self.circuit_breakers[model_type] = {
                "state": "closed",  # closed, open, half-open
                "failure_count": 0,
                "failure_threshold": 5,
                "recovery_timeout": 60.0,
                "last_failure": None
            }
    
    def get_optimal_model(self, task_type: TaskType, context_length: int = 0) -> Optional[ModelType]:
        """
        Get the optimal model for a given task type with enhanced 2025 logic.
        
        Args:
            task_type: Type of task to perform
            context_length: Length of context (for model selection)
            
        Returns:
            Optimal model type or None if no models available
        """
        try:
            # Enhanced model selection strategies for 2025
            model_priorities = {
                TaskType.REALTIME: [
                    ModelType.GROQ,      # Fastest with Llama 3.3 70B
                    ModelType.MISTRAL,   # Fast and efficient
                    ModelType.OLLAMA     # Local fallback
                ],
                TaskType.MULTIMODAL: [
                    ModelType.GEMINI,    # Best multimodal with Gemini 2.0
                    ModelType.CLAUDE,    # Good vision capabilities
                    ModelType.TOGETHER   # Large model for complex multimodal
                ],
                TaskType.REASONING: [
                    ModelType.CLAUDE,    # Best reasoning with Claude 3.5
                    ModelType.TOGETHER,  # 405B model for complex reasoning
                    ModelType.GEMINI     # Strong reasoning capabilities
                ],
                TaskType.COMPLEX: [
                    ModelType.TOGETHER,  # 405B model for complexity
                    ModelType.CLAUDE,    # Strong reasoning
                    ModelType.GEMINI     # Good general capabilities
                ],
                TaskType.SPECIALIZED: [
                    ModelType.CLAUDE,    # Domain expertise
                    ModelType.GROQ,      # Fast specialized tasks
                    ModelType.MISTRAL    # Efficient specialized
                ],
                TaskType.EFFICIENT: [
                    ModelType.MISTRAL,   # Most efficient
                    ModelType.GROQ,      # Fast and efficient
                    ModelType.OLLAMA     # Local efficiency
                ],
                TaskType.CREATIVE: [
                    ModelType.CLAUDE,    # Creative capabilities
                    ModelType.GEMINI,    # Creative multimodal
                    ModelType.TOGETHER   # Large model creativity
                ],
                TaskType.ANALYTICAL: [
                    ModelType.CLAUDE,    # Strong analysis
                    ModelType.TOGETHER,  # Complex analysis
                    ModelType.GEMINI     # Data analysis
                ],
                TaskType.FINANCIAL: [
                    ModelType.CLAUDE,    # Financial reasoning
                    ModelType.GROQ,      # Fast financial data
                    ModelType.TOGETHER   # Complex financial models
                ]
            }
            
            priorities = model_priorities.get(task_type, list(self.models.keys()))
            
            # Check availability, rate limits, and circuit breakers
            for model_type in priorities:
                if (self._is_model_available(model_type) and 
                    self._check_circuit_breaker(model_type) and
                    self._check_context_length(model_type, context_length)):
                    return model_type
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting optimal model: {e}")
            return None
    
    def _check_circuit_breaker(self, model_type: ModelType) -> bool:
        """Check if circuit breaker allows requests"""
        try:
            cb = self.circuit_breakers[model_type]
            
            if cb["state"] == "closed":
                return True
            elif cb["state"] == "open":
                # Check if recovery timeout has passed
                if (datetime.now() - cb["last_failure"]).seconds > cb["recovery_timeout"]:
                    cb["state"] = "half-open"
                    cb["failure_count"] = 0
                    return True
                return False
            elif cb["state"] == "half-open":
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return True
    
    def _check_context_length(self, model_type: ModelType, context_length: int) -> bool:
        """Check if model can handle the context length"""
        try:
            if model_type not in self.model_configs:
                return True
            
            config = self.model_configs[model_type]
            # Reserve some tokens for response
            max_context = config.max_tokens - 1000
            
            return context_length <= max_context
            
        except Exception as e:
            logger.error(f"Error checking context length: {e}")
            return True
    
    def _is_model_available(self, model_type: ModelType) -> bool:
        """Enhanced availability check with concurrent request limits"""
        try:
            if model_type not in self.models:
                return False
            
            if model_type not in self.usage_tracking:
                return True
            
            usage = self.usage_tracking[model_type]
            limits = self.rate_limits[model_type]
            now = datetime.now()
            
            # Reset counters if needed
            if (now - usage["last_minute_reset"]).seconds >= 60:
                usage["minute_requests"] = 0
                usage["tokens_used"] = 0
                usage["last_minute_reset"] = now
            
            if (now - usage["last_day_reset"]).days >= 1:
                usage["daily_requests"] = 0
                usage["last_day_reset"] = now
            
            # Check all limits
            if usage["daily_requests"] >= limits["requests_per_day"]:
                return False
            
            if usage["minute_requests"] >= limits["requests_per_minute"]:
                return False
            
            if usage["concurrent_requests"] >= limits["concurrent_requests"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    async def get_response(
        self, 
        prompt: str, 
        task_type: TaskType = TaskType.REALTIME,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        max_retries: int = 3,
        stream: bool = False,
        **kwargs
    ) -> ModelResponse:
        """
        Enhanced response generation with 2025 features.
        
        Args:
            prompt: Input prompt
            task_type: Type of task
            system_prompt: Optional system prompt
            use_cache: Whether to use response caching
            max_retries: Maximum retry attempts
            stream: Whether to stream response
            **kwargs: Additional model parameters
            
        Returns:
            ModelResponse with enhanced metadata
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(prompt, task_type, system_prompt, kwargs)
            
            # Check cache first
            if use_cache and cache_key in self.response_cache:
                with self.cache_lock:
                    cached = self.response_cache[cache_key]
                    if (datetime.now() - cached["timestamp"]).seconds < 3600:  # 1 hour cache
                        logger.info(f"Using cached response for {task_type.value}")
                        cached["cached"] = True
                        return cached
            
            # Get optimal model
            context_length = len(prompt) + (len(system_prompt) if system_prompt else 0)
            model_type = self.get_optimal_model(task_type, context_length)
            
            if not model_type:
                raise ValueError("No available models for task")
            
            # Try with enhanced fallback
            for attempt in range(max_retries):
                try:
                    response = await self._call_model_enhanced(
                        model_type, prompt, system_prompt, stream, **kwargs
                    )
                    
                    # Update circuit breaker on success
                    self._update_circuit_breaker(model_type, success=True)
                    
                    # Cache response
                    if use_cache:
                        with self.cache_lock:
                            self.response_cache[cache_key] = response
                            # Implement LRU eviction if cache gets too large
                            if len(self.response_cache) > 1000:
                                self._evict_old_cache_entries()
                    
                    return response
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {model_type.value}: {e}")
                    
                    # Update circuit breaker on failure
                    self._update_circuit_breaker(model_type, success=False)
                    
                    if attempt < max_retries - 1:
                        # Try next best model
                        model_type = self._get_fallback_model(model_type, task_type)
                        if not model_type:
                            break
                        
                        # Exponential backoff with jitter
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(delay)
            
            raise Exception("All model attempts failed")
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return ModelResponse(
                content="Error: Unable to get response from any model",
                model_used="none",
                response_time=0.0,
                timestamp=datetime.now(),
                error=str(e)
            )
    
    def _generate_cache_key(self, prompt: str, task_type: TaskType, system_prompt: Optional[str], kwargs: dict) -> str:
        """Generate cache key for request"""
        key_data = {
            "prompt": prompt,
            "task_type": task_type.value,
            "system_prompt": system_prompt,
            "kwargs": kwargs
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _evict_old_cache_entries(self):
        """Evict old cache entries (LRU-like behavior)"""
        try:
            # Remove entries older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            keys_to_remove = []
            
            for key, entry in self.response_cache.items():
                if entry["timestamp"] < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.response_cache[key]
            
            # If still too large, remove oldest entries
            if len(self.response_cache) > 800:
                sorted_entries = sorted(
                    self.response_cache.items(),
                    key=lambda x: x[1]["timestamp"]
                )
                for key, _ in sorted_entries[:200]:  # Remove oldest 200
                    del self.response_cache[key]
            
        except Exception as e:
            logger.error(f"Error evicting cache entries: {e}")
    
    async def _call_model_enhanced(
        self, 
        model_type: ModelType, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> ModelResponse:
        """Enhanced model calling with better error handling and metrics"""
        try:
            start_time = time.time()
            
            # Update usage tracking
            self._update_usage_tracking(model_type, increment_concurrent=True)
            
            try:
                # Prepare messages
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=prompt))
                
                # Call model with timeout
                model = self.models[model_type]
                config = self.model_configs.get(model_type, ModelConfig(model_name="unknown"))
                
                # Apply custom parameters
                model_kwargs = {
                    "temperature": kwargs.get("temperature", config.temperature),
                    "max_tokens": kwargs.get("max_tokens", config.max_tokens),
                    "top_p": kwargs.get("top_p", config.top_p),
                }
                
                # Call model with timeout
                response = await asyncio.wait_for(
                    model.ainvoke(messages, **model_kwargs),
                    timeout=config.timeout
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Extract token usage if available
                tokens_used = getattr(response, 'usage', {}).get('total_tokens', 0)
                
                # Update performance metrics
                self._update_performance_metrics(model_type, response_time, tokens_used)
                
                return ModelResponse(
                    content=response.content,
                    model_used=model_type.value,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    tokens_used=tokens_used,
                    confidence=1.0,
                    cached=False
                )
                
            finally:
                # Always decrement concurrent requests
                self._update_usage_tracking(model_type, increment_concurrent=False)
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling model {model_type.value}")
            raise Exception(f"Model {model_type.value} timed out")
        except Exception as e:
            logger.error(f"Error calling model {model_type.value}: {e}")
            raise
    
    def _update_usage_tracking(self, model_type: ModelType, increment_concurrent: bool = True):
        """Update usage tracking with concurrent request management"""
        try:
            if model_type in self.usage_tracking:
                usage = self.usage_tracking[model_type]
                
                if increment_concurrent:
                    usage["daily_requests"] += 1
                    usage["minute_requests"] += 1
                    usage["concurrent_requests"] += 1
                    usage["last_request"] = datetime.now()
                else:
                    usage["concurrent_requests"] = max(0, usage["concurrent_requests"] - 1)
                    
        except Exception as e:
            logger.error(f"Error updating usage tracking: {e}")
    
    def _update_performance_metrics(self, model_type: ModelType, response_time: float, tokens_used: int):
        """Update performance metrics for monitoring"""
        try:
            if model_type not in self.performance_metrics:
                self.performance_metrics[model_type] = {
                    "total_requests": 0,
                    "total_response_time": 0.0,
                    "total_tokens": 0,
                    "avg_response_time": 0.0,
                    "avg_tokens_per_request": 0.0,
                    "last_updated": datetime.now()
                }
            
            metrics = self.performance_metrics[model_type]
            metrics["total_requests"] += 1
            metrics["total_response_time"] += response_time
            metrics["total_tokens"] += tokens_used
            metrics["avg_response_time"] = metrics["total_response_time"] / metrics["total_requests"]
            metrics["avg_tokens_per_request"] = metrics["total_tokens"] / metrics["total_requests"]
            metrics["last_updated"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _update_circuit_breaker(self, model_type: ModelType, success: bool):
        """Update circuit breaker state"""
        try:
            cb = self.circuit_breakers[model_type]
            
            if success:
                if cb["state"] == "half-open":
                    cb["state"] = "closed"
                cb["failure_count"] = 0
            else:
                cb["failure_count"] += 1
                cb["last_failure"] = datetime.now()
                
                if cb["failure_count"] >= cb["failure_threshold"]:
                    cb["state"] = "open"
                    logger.warning(f"Circuit breaker opened for {model_type.value}")
                    
        except Exception as e:
            logger.error(f"Error updating circuit breaker: {e}")
    
    def _get_fallback_model(self, current_model: ModelType, task_type: TaskType) -> Optional[ModelType]:
        """Get fallback model with enhanced 2025 logic"""
        try:
            # Enhanced fallback priorities
            fallback_priorities = {
                TaskType.REALTIME: [ModelType.OLLAMA, ModelType.MISTRAL, ModelType.GROQ],
                TaskType.MULTIMODAL: [ModelType.CLAUDE, ModelType.TOGETHER, ModelType.GROQ],
                TaskType.REASONING: [ModelType.TOGETHER, ModelType.GEMINI, ModelType.GROQ],
                TaskType.COMPLEX: [ModelType.CLAUDE, ModelType.GEMINI, ModelType.GROQ],
                TaskType.SPECIALIZED: [ModelType.GROQ, ModelType.MISTRAL, ModelType.OLLAMA],
                TaskType.EFFICIENT: [ModelType.OLLAMA, ModelType.GROQ, ModelType.MISTRAL],
                TaskType.CREATIVE: [ModelType.GEMINI, ModelType.TOGETHER, ModelType.GROQ],
                TaskType.ANALYTICAL: [ModelType.TOGETHER, ModelType.GEMINI, ModelType.GROQ],
                TaskType.FINANCIAL: [ModelType.GROQ, ModelType.TOGETHER, ModelType.MISTRAL]
            }
            
            priorities = fallback_priorities.get(task_type, list(self.models.keys()))
            
            # Find next available model after current
            current_index = -1
            for i, model in enumerate(priorities):
                if model == current_model:
                    current_index = i
                    break
            
            # Try models after current
            for i in range(current_index + 1, len(priorities)):
                model_type = priorities[i]
                if (self._is_model_available(model_type) and 
                    self._check_circuit_breaker(model_type)):
                    return model_type
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting fallback model: {e}")
            return None
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for 2025"""
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(self.models),
                "available_models": [],
                "usage_by_model": {},
                "performance_metrics": self.performance_metrics,
                "circuit_breakers": {},
                "cache_stats": {
                    "total_entries": len(self.response_cache),
                    "cache_hit_ratio": 0.0  # Could be calculated if we track hits/misses
                }
            }
            
            for model_type in self.models:
                stats["available_models"].append(model_type.value)
                
                # Usage stats
                if model_type in self.usage_tracking:
                    usage = self.usage_tracking[model_type]
                    limits = self.rate_limits[model_type]
                    
                    stats["usage_by_model"][model_type.value] = {
                        "daily_requests": usage["daily_requests"],
                        "daily_limit": limits["requests_per_day"],
                        "minute_requests": usage["minute_requests"],
                        "minute_limit": limits["requests_per_minute"],
                        "concurrent_requests": usage["concurrent_requests"],
                        "concurrent_limit": limits["concurrent_requests"],
                        "last_request": usage["last_request"].isoformat() if usage["last_request"] else None,
                        "available": self._is_model_available(model_type)
                    }
                
                # Circuit breaker stats
                if model_type in self.circuit_breakers:
                    cb = self.circuit_breakers[model_type]
                    stats["circuit_breakers"][model_type.value] = {
                        "state": cb["state"],
                        "failure_count": cb["failure_count"],
                        "last_failure": cb["last_failure"].isoformat() if cb["last_failure"] else None
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting enhanced stats: {e}")
            return {"error": str(e)}
    
    def reset_all_tracking(self):
        """Reset all tracking data"""
        try:
            # Reset usage tracking
            for model_type in self.usage_tracking:
                self.usage_tracking[model_type] = {
                    "daily_requests": 0,
                    "minute_requests": 0,
                    "tokens_used": 0,
                    "concurrent_requests": 0,
                    "last_request": None,
                    "last_minute_reset": datetime.now(),
                    "last_day_reset": datetime.now()
                }
            
            # Reset circuit breakers
            for model_type in self.circuit_breakers:
                self.circuit_breakers[model_type] = {
                    "state": "closed",
                    "failure_count": 0,
                    "failure_threshold": 5,
                    "recovery_timeout": 60.0,
                    "last_failure": None
                }
            
            # Clear cache
            with self.cache_lock:
                self.response_cache.clear()
            
            # Reset performance metrics
            self.performance_metrics.clear()
            
            logger.info("All tracking data reset")
            
        except Exception as e:
            logger.error(f"Error resetting tracking: {e}")


# Global model manager instance - 2025 Edition
model_manager = ModelManager() 