"""
Alpha Discovery Configuration Loader

This module provides comprehensive configuration management for the Alpha Discovery platform,
including loading, validation, environment-specific overrides, and configuration watching.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from pydantic import BaseModel, ValidationError, validator
from pydantic.fields import Field
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConfigMetadata:
    """Configuration metadata for tracking and validation."""
    loaded_at: datetime = field(default_factory=datetime.now)
    file_path: str = ""
    file_hash: str = ""
    environment: str = "development"
    version: str = "1.0.0"
    last_modified: Optional[datetime] = None


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration file changes."""
    
    def __init__(self, config_loader):
        self.config_loader = config_loader
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.yaml'):
            logger.info(f"Configuration file changed: {event.src_path}")
            self.config_loader.reload_config()


class BaseConfig(BaseModel):
    """Base configuration model with common validation."""
    
    class Config:
        extra = "allow"  # Allow extra fields
        validate_assignment = True
        
    @validator('*', pre=True)
    def validate_environment_variables(cls, v):
        """Replace environment variable placeholders."""
        if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
            env_var = v[2:-1]
            return os.getenv(env_var, v)
        return v


class StrategiesConfig(BaseConfig):
    """Configuration model for strategies.yaml."""
    global_: Dict[str, Any] = Field(alias='global')
    agents: Dict[str, Any]
    strategies: Dict[str, Any]
    portfolio: Dict[str, Any]
    risk_management: Dict[str, Any]
    execution: Dict[str, Any]
    backtesting: Dict[str, Any]
    performance: Dict[str, Any]
    limits: Dict[str, Any]


class MarketDataConfig(BaseConfig):
    """Configuration model for market_data.yaml."""
    global_: Dict[str, Any] = Field(alias='global')
    data_sources: Dict[str, Any]
    symbols: Dict[str, Any]
    data_processing: Dict[str, Any]
    storage: Dict[str, Any]
    feeds: Dict[str, Any]
    monitoring: Dict[str, Any]
    failover: Dict[str, Any]


class RiskConfig(BaseConfig):
    """Configuration model for risk.yaml."""
    global_: Dict[str, Any] = Field(alias='global')
    var_config: Dict[str, Any]
    expected_shortfall: Dict[str, Any]
    position_limits: Dict[str, Any]
    portfolio_limits: Dict[str, Any]
    liquidity_risk: Dict[str, Any]
    credit_risk: Dict[str, Any]
    operational_risk: Dict[str, Any]
    model_risk: Dict[str, Any]
    stress_testing: Dict[str, Any]
    circuit_breakers: Dict[str, Any]
    risk_reporting: Dict[str, Any]
    risk_governance: Dict[str, Any]


class MonitoringConfig(BaseConfig):
    """Configuration model for monitoring.yaml."""
    global_: Dict[str, Any] = Field(alias='global')
    reddit: Dict[str, Any]
    alerts: Dict[str, Any]
    system_monitoring: Dict[str, Any]
    notifications: Dict[str, Any]
    dashboards: Dict[str, Any]


class APIConfig(BaseConfig):
    """Configuration model for api.yaml."""
    global_: Dict[str, Any] = Field(alias='global')
    authentication: Dict[str, Any]
    rate_limiting: Dict[str, Any]
    endpoints: Dict[str, Any]
    external_apis: Dict[str, Any]
    websocket: Dict[str, Any]
    api_gateway: Dict[str, Any]
    monitoring: Dict[str, Any]
    documentation: Dict[str, Any]
    security: Dict[str, Any]


class ModelsConfig(BaseConfig):
    """Configuration model for models.yaml."""
    global_: Dict[str, Any] = Field(alias='global')
    technical_models: Dict[str, Any]
    fundamental_models: Dict[str, Any]
    sentiment_models: Dict[str, Any]
    regime_models: Dict[str, Any]
    options_models: Dict[str, Any]
    risk_models: Dict[str, Any]
    reinforcement_learning: Dict[str, Any]
    ensemble_models: Dict[str, Any]
    evaluation: Dict[str, Any]
    deployment: Dict[str, Any]


class ConfigLoader:
    """
    Comprehensive configuration loader for Alpha Discovery platform.
    
    Features:
    - Load and validate YAML configuration files
    - Environment-specific overrides
    - Configuration watching and hot-reloading
    - Configuration merging and inheritance
    - Validation and error handling
    - Configuration caching and performance optimization
    """
    
    def __init__(self, config_dir: str = "configs", environment: str = None):
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.configs: Dict[str, Any] = {}
        self.metadata: Dict[str, ConfigMetadata] = {}
        self.config_models = {
            'strategies': StrategiesConfig,
            'market_data': MarketDataConfig,
            'risk': RiskConfig,
            'monitoring': MonitoringConfig,
            'api': APIConfig,
            'models': ModelsConfig
        }
        self._observers = []
        self._lock = threading.RLock()
        self._callbacks = []
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Load initial configuration
        self.load_all_configs()
        
        # Start file watching if in development
        if self.environment == "development":
            self.start_watching()
    
    def load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load and parse a YAML configuration file."""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Replace environment variables
            content = self._replace_environment_variables(content)
            
            # Parse YAML
            config_data = yaml.safe_load(content)
            
            # Calculate file hash for change detection
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Create metadata
            metadata = ConfigMetadata(
                file_path=str(file_path),
                file_hash=file_hash,
                environment=self.environment,
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
            )
            
            return config_data, metadata
            
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"YAML parsing error in {filename}: {e}")
        except Exception as e:
            raise ConfigValidationError(f"Error loading {filename}: {e}")
    
    def _replace_environment_variables(self, content: str) -> str:
        """Replace environment variable placeholders in configuration content."""
        import re
        
        def replace_env_var(match):
            env_var = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(env_var, default_value)
        
        # Pattern for ${ENV_VAR} or ${ENV_VAR:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        return re.sub(pattern, replace_env_var, content)
    
    def validate_config(self, config_name: str, config_data: Dict[str, Any]) -> Any:
        """Validate configuration data using Pydantic models."""
        if config_name not in self.config_models:
            logger.warning(f"No validation model found for {config_name}")
            return config_data
        
        try:
            model_class = self.config_models[config_name]
            validated_config = model_class(**config_data)
            return validated_config.dict()
        except ValidationError as e:
            raise ConfigValidationError(f"Validation error in {config_name}: {e}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a specific configuration file."""
        filename = f"{config_name}.yaml"
        
        with self._lock:
            try:
                # Load base configuration
                config_data, metadata = self.load_config_file(filename)
                
                # Apply environment-specific overrides
                config_data = self._apply_environment_overrides(config_name, config_data)
                
                # Validate configuration
                config_data = self.validate_config(config_name, config_data)
                
                # Store configuration and metadata
                self.configs[config_name] = config_data
                self.metadata[config_name] = metadata
                
                logger.info(f"Loaded configuration: {config_name}")
                return config_data
                
            except Exception as e:
                logger.error(f"Failed to load configuration {config_name}: {e}")
                raise
    
    def _apply_environment_overrides(self, config_name: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        override_file = self.config_dir / f"{config_name}.{self.environment}.yaml"
        
        if override_file.exists():
            try:
                override_data, _ = self.load_config_file(override_file.name)
                config_data = self._deep_merge(config_data, override_data)
                logger.info(f"Applied {self.environment} overrides for {config_name}")
            except Exception as e:
                logger.warning(f"Failed to apply environment overrides for {config_name}: {e}")
        
        return config_data
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_all_configs(self) -> None:
        """Load all configuration files."""
        config_files = [
            'strategies',
            'market_data',
            'risk',
            'monitoring',
            'api',
            'models'
        ]
        
        for config_name in config_files:
            try:
                self.load_config(config_name)
            except Exception as e:
                logger.error(f"Failed to load {config_name} configuration: {e}")
                # Continue loading other configs
    
    def reload_config(self, config_name: str = None) -> None:
        """Reload one or all configuration files."""
        if config_name:
            self.load_config(config_name)
            self._notify_callbacks(config_name)
        else:
            self.load_all_configs()
            self._notify_callbacks()
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a loaded configuration."""
        if config_name not in self.configs:
            raise KeyError(f"Configuration {config_name} not loaded")
        
        return self.configs[config_name]
    
    def get_config_value(self, config_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation.
        
        Example: get_config_value('strategies.global.initial_capital')
        """
        path_parts = config_path.split('.')
        config_name = path_parts[0]
        
        if config_name not in self.configs:
            return default
        
        current = self.configs[config_name]
        
        for part in path_parts[1:]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set_config_value(self, config_path: str, value: Any) -> None:
        """
        Set a specific configuration value using dot notation.
        
        Example: set_config_value('strategies.global.initial_capital', 2000000)
        """
        path_parts = config_path.split('.')
        config_name = path_parts[0]
        
        if config_name not in self.configs:
            raise KeyError(f"Configuration {config_name} not loaded")
        
        current = self.configs[config_name]
        
        for part in path_parts[1:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[path_parts[-1]] = value
        
        # Notify callbacks of change
        self._notify_callbacks(config_name)
    
    def start_watching(self) -> None:
        """Start watching configuration files for changes."""
        try:
            observer = Observer()
            event_handler = ConfigFileHandler(self)
            observer.schedule(event_handler, str(self.config_dir), recursive=False)
            observer.start()
            self._observers.append(observer)
            logger.info("Started configuration file watching")
        except Exception as e:
            logger.error(f"Failed to start configuration watching: {e}")
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        for observer in self._observers:
            observer.stop()
            observer.join()
        self._observers.clear()
        logger.info("Stopped configuration file watching")
    
    def add_callback(self, callback) -> None:
        """Add a callback function to be called when configuration changes."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback) -> None:
        """Remove a callback function."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_callbacks(self, config_name: str = None) -> None:
        """Notify all registered callbacks of configuration changes."""
        for callback in self._callbacks:
            try:
                if config_name:
                    callback(config_name, self.configs[config_name])
                else:
                    callback(self.configs)
            except Exception as e:
                logger.error(f"Error in configuration callback: {e}")
    
    def export_config(self, config_name: str, format_type: str = "yaml") -> str:
        """Export configuration in specified format."""
        if config_name not in self.configs:
            raise KeyError(f"Configuration {config_name} not loaded")
        
        config_data = self.configs[config_name]
        
        if format_type.lower() == "yaml":
            return yaml.dump(config_data, default_flow_style=False, indent=2)
        elif format_type.lower() == "json":
            return json.dumps(config_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def validate_all_configs(self) -> Dict[str, List[str]]:
        """Validate all loaded configurations and return validation results."""
        validation_results = {}
        
        for config_name, config_data in self.configs.items():
            try:
                self.validate_config(config_name, config_data)
                validation_results[config_name] = []
            except ConfigValidationError as e:
                validation_results[config_name] = [str(e)]
        
        return validation_results
    
    def get_metadata(self, config_name: str = None) -> Union[ConfigMetadata, Dict[str, ConfigMetadata]]:
        """Get metadata for one or all configurations."""
        if config_name:
            return self.metadata.get(config_name)
        return self.metadata
    
    def backup_configs(self, backup_dir: str = None) -> str:
        """Create a backup of all configuration files."""
        if backup_dir is None:
            backup_dir = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        for config_file in self.config_dir.glob("*.yaml"):
            backup_file = backup_path / config_file.name
            backup_file.write_text(config_file.read_text())
        
        logger.info(f"Configuration backup created: {backup_path}")
        return str(backup_path)
    
    def compare_configs(self, config_name: str, other_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current configuration with another configuration."""
        if config_name not in self.configs:
            raise KeyError(f"Configuration {config_name} not loaded")
        
        current_config = self.configs[config_name]
        differences = {}
        
        # Find differences
        self._find_differences(current_config, other_config, differences, "")
        
        return differences
    
    def _find_differences(self, current: Any, other: Any, differences: Dict[str, Any], path: str) -> None:
        """Recursively find differences between two configurations."""
        if isinstance(current, dict) and isinstance(other, dict):
            all_keys = set(current.keys()) | set(other.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in current:
                    differences[new_path] = {"action": "added", "value": other[key]}
                elif key not in other:
                    differences[new_path] = {"action": "removed", "value": current[key]}
                else:
                    self._find_differences(current[key], other[key], differences, new_path)
        elif current != other:
            differences[path] = {
                "action": "changed",
                "old_value": current,
                "new_value": other
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_watching()


# Singleton instance for global access
_config_loader = None
_config_loader_lock = threading.Lock()


def get_config_loader() -> ConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    
    if _config_loader is None:
        with _config_loader_lock:
            if _config_loader is None:
                _config_loader = ConfigLoader()
    
    return _config_loader


def get_config(config_name: str) -> Dict[str, Any]:
    """Convenience function to get configuration."""
    return get_config_loader().get_config(config_name)


def get_config_value(config_path: str, default: Any = None) -> Any:
    """Convenience function to get configuration value."""
    return get_config_loader().get_config_value(config_path, default)


def reload_configs():
    """Convenience function to reload all configurations."""
    get_config_loader().reload_config()


# Configuration validation utilities
def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format."""
    import re
    return bool(re.match(r'^[A-Z]{1,5}$', symbol))


def validate_date_format(date_string: str) -> bool:
    """Validate date format (YYYY-MM-DD)."""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def validate_percentage(value: float) -> bool:
    """Validate percentage value (0-1)."""
    return 0 <= value <= 1


def validate_positive_number(value: Union[int, float]) -> bool:
    """Validate positive number."""
    return value > 0


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        # Initialize configuration loader
        config_loader = ConfigLoader()
        
        # Get specific configurations
        strategies_config = config_loader.get_config('strategies')
        market_data_config = config_loader.get_config('market_data')
        
        # Get specific values
        initial_capital = config_loader.get_config_value('strategies.global.initial_capital')
        print(f"Initial Capital: ${initial_capital:,}")
        
        # Validate all configurations
        validation_results = config_loader.validate_all_configs()
        print("Validation Results:", validation_results)
        
        # Export configuration
        yaml_export = config_loader.export_config('strategies', 'yaml')
        print("Strategies YAML Export Length:", len(yaml_export))
        
    except Exception as e:
        logger.error(f"Configuration loader example failed: {e}") 