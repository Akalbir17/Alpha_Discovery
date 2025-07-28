#!/usr/bin/env python3
"""
Model Cache Manager for Alpha Discovery Platform

Optimizes model loading and caching for faster startup and better performance.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from functools import lru_cache
import gc
import psutil

logger = logging.getLogger(__name__)

class ModelCache:
    """Global model cache for optimizing model loading"""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._loading: Dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()
        self._initialized = False
        
    def get(self, key: str) -> Optional[Any]:
        """Get a model from cache"""
        return self._cache.get(key)
    
    def set(self, key: str, model: Any):
        """Set a model in cache"""
        with self._lock:
            self._cache[key] = model
            logger.info(f"Model cached: {key}")
    
    def is_loading(self, key: str) -> bool:
        """Check if a model is currently loading"""
        return key in self._loading
    
    def set_loading(self, key: str, task: asyncio.Task):
        """Mark a model as loading"""
        with self._lock:
            self._loading[key] = task
    
    def clear_loading(self, key: str):
        """Clear loading status for a model"""
        with self._lock:
            self._loading.pop(key, None)
    
    async def get_or_load(self, key: str, loader_func: Callable, timeout: float = 60.0) -> Any:
        """Get model from cache or load it if not present"""
        
        # Check if already cached
        cached = self.get(key)
        if cached is not None:
            logger.info(f"Model retrieved from cache: {key}")
            return cached
        
        # Check if already loading
        if self.is_loading(key):
            logger.info(f"Model already loading, waiting: {key}")
            task = self._loading[key]
            try:
                result = await asyncio.wait_for(task, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                logger.error(f"Model loading timed out: {key}")
                self.clear_loading(key)
                raise
        
        # Start loading
        logger.info(f"Starting model load: {key}")
        task = asyncio.create_task(self._load_model(key, loader_func))
        self.set_loading(key, task)
        
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            self.set(key, result)
            return result
        except Exception as e:
            logger.error(f"Failed to load model {key}: {e}")
            raise
        finally:
            self.clear_loading(key)
    
    async def _load_model(self, key: str, loader_func: Callable) -> Any:
        """Load a model with memory optimization"""
        try:
            start_time = time.time()
            
            # Run loader in thread to avoid blocking
            model = await asyncio.to_thread(loader_func)
            
            elapsed = time.time() - start_time
            logger.info(f"Model loaded successfully: {key} in {elapsed:.2f}s")
            
            # Optimize memory after loading
            await self._optimize_memory()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {key}: {e}")
            raise
    
    async def _optimize_memory(self):
        """Optimize memory usage after model loading"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            memory = psutil.virtual_memory()
            logger.info(f"Memory usage after model load: {memory.percent:.1f}%")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
    
    def preload_models(self, models: Dict[str, Callable]):
        """Preload multiple models in parallel"""
        async def preload():
            tasks = []
            for key, loader in models.items():
                if self.get(key) is None and not self.is_loading(key):
                    task = asyncio.create_task(self.get_or_load(key, loader))
                    tasks.append(task)
            
            if tasks:
                logger.info(f"Preloading {len(tasks)} models...")
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info("Model preloading completed")
        
        return asyncio.create_task(preload())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "cached_models": len(self._cache),
                "loading_models": len(self._loading),
                "cache_keys": list(self._cache.keys()),
                "loading_keys": list(self._loading.keys())
            }

# Global model cache instance
model_cache = ModelCache()

# Decorator for caching model initialization
def cached_model(model_key: str):
    """Decorator to cache model initialization"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await model_cache.get_or_load(model_key, lambda: func(*args, **kwargs))
        return wrapper
    return decorator

# Pre-warming functions for common models
async def prewarm_common_models():
    """Pre-warm commonly used models"""
    common_models = {
        "finbert_sentiment": lambda: None,  # Will be replaced with actual loader
        "sentence_transformer": lambda: None,  # Will be replaced with actual loader
        "emotion_model": lambda: None,  # Will be replaced with actual loader
    }
    
    logger.info("Pre-warming common models...")
    await model_cache.preload_models(common_models) 