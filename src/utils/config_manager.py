"""
Configuration Manager
Handles loading and accessing configuration
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration for the Alpha Discovery platform"""
    
    def __init__(self, config_dir: str = 'configs'):
        self.config_dir = config_dir
        self.config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML files"""
        try:
            # Load main config
            with open(os.path.join(self.config_dir, 'config.yaml'), 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Load environment specific config
            env = os.environ.get('ALPHA_DISCOVERY_ENV', 'development')
            env_config_path = os.path.join(self.config_dir, f'config.{env}.yaml')
            
            if os.path.exists(env_config_path):
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f)
                    self._deep_merge(self.config, env_config)
            
            logger.info(f"Configuration loaded for environment: {env}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """Get configuration section"""
        return self.config.get(section)
        
    def _deep_merge(self, source, destination):
        """Deep merge two dictionaries"""
        for key, value in source.items():
            if isinstance(value, dict) and key in destination:
                destination[key] = self._deep_merge(value, destination[key])
            else:
                destination[key] = value
        return destination

# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global config manager instance"""
    global _config_manager
    if _config_manager is None:
        # Use relative path for local development, absolute path for Docker
        config_dir = os.environ.get('CONFIG_DIR', 'configs')
        _config_manager = ConfigManager(config_dir)
    return _config_manager

def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific configuration value"""
    return get_config_manager().get(key, default)

def get_config_section(section: str) -> Optional[Dict[str, Any]]:
    """Get a configuration section"""
    return get_config_manager().get_section(section) 