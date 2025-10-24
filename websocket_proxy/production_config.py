"""
Production Configuration for WebSocket Proxy

This module provides production-ready configuration settings
for optimal performance and reliability.
"""

import os
from typing import Dict, Any


class ProductionConfig:
    """Production configuration settings"""
    
    # WebSocket Server Settings
    WEBSOCKET_HOST = os.getenv('WEBSOCKET_HOST', '0.0.0.0')
    WEBSOCKET_PORT = int(os.getenv('WEBSOCKET_PORT', '8765'))
    
    # Performance Settings
    HIGH_PERFORMANCE_MODE = os.getenv('DISRUPTOR_HIGH_PERFORMANCE', 'false').lower() == 'true'
    BUFFER_SIZE = int(os.getenv('DISRUPTOR_BUFFER_SIZE', '65536'))  # 64K for balanced performance
    WAIT_STRATEGY = os.getenv('DISRUPTOR_WAIT_STRATEGY', 'yielding')  # yielding/busy_spin
    NUM_PROCESSORS = int(os.getenv('DISRUPTOR_PROCESSORS', '4'))
    
    # Client Management
    MAX_CLIENTS = int(os.getenv('MAX_WEBSOCKET_CLIENTS', '10000'))
    CLIENT_TIMEOUT = int(os.getenv('CLIENT_TIMEOUT_SECONDS', '300'))  # 5 minutes
    HEARTBEAT_INTERVAL = int(os.getenv('HEARTBEAT_INTERVAL_SECONDS', '30'))
    
    # Memory Management
    ENABLE_MEMORY_MONITORING = os.getenv('ENABLE_MEMORY_MONITORING', 'true').lower() == 'true'
    MAX_MEMORY_MB = int(os.getenv('MAX_MEMORY_MB', '2048'))  # 2GB limit
    GC_THRESHOLD = int(os.getenv('GC_THRESHOLD_MB', '1024'))  # Trigger cleanup at 1GB
    
    # Logging
    LOG_LEVEL = os.getenv('WEBSOCKET_LOG_LEVEL', 'INFO')
    ENABLE_PERFORMANCE_LOGGING = os.getenv('ENABLE_PERFORMANCE_LOGGING', 'true').lower() == 'true'
    LOG_STATS_INTERVAL = int(os.getenv('LOG_STATS_INTERVAL_SECONDS', '60'))
    
    # Security
    ENABLE_RATE_LIMITING = os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true'
    MAX_MESSAGES_PER_SECOND = int(os.getenv('MAX_MESSAGES_PER_SECOND', '1000'))
    MAX_SUBSCRIPTIONS_PER_CLIENT = int(os.getenv('MAX_SUBSCRIPTIONS_PER_CLIENT', '500'))
    
    # Broker Settings
    BROKER_RECONNECT_ATTEMPTS = int(os.getenv('BROKER_RECONNECT_ATTEMPTS', '10'))
    BROKER_RECONNECT_DELAY = int(os.getenv('BROKER_RECONNECT_DELAY_SECONDS', '5'))
    BROKER_TIMEOUT = int(os.getenv('BROKER_TIMEOUT_SECONDS', '30'))
    
    @classmethod
    def get_disruptor_config(cls) -> Dict[str, Any]:
        """Get disruptor-specific configuration"""
        return {
            'ring_buffer_size': cls.BUFFER_SIZE,
            'wait_strategy': cls.WAIT_STRATEGY,
            'num_processors': cls.NUM_PROCESSORS,
            'high_performance': cls.HIGH_PERFORMANCE_MODE
        }
    
    @classmethod
    def get_server_config(cls) -> Dict[str, Any]:
        """Get server configuration"""
        return {
            'host': cls.WEBSOCKET_HOST,
            'port': cls.WEBSOCKET_PORT,
            'max_clients': cls.MAX_CLIENTS,
            'client_timeout': cls.CLIENT_TIMEOUT,
            'heartbeat_interval': cls.HEARTBEAT_INTERVAL
        }
    
    @classmethod
    def get_monitoring_config(cls) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            'enable_memory_monitoring': cls.ENABLE_MEMORY_MONITORING,
            'max_memory_mb': cls.MAX_MEMORY_MB,
            'gc_threshold': cls.GC_THRESHOLD,
            'log_level': cls.LOG_LEVEL,
            'enable_performance_logging': cls.ENABLE_PERFORMANCE_LOGGING,
            'log_stats_interval': cls.LOG_STATS_INTERVAL
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings"""
        try:
            # Validate buffer size is power of 2
            if cls.BUFFER_SIZE & (cls.BUFFER_SIZE - 1) != 0:
                raise ValueError(f"Buffer size {cls.BUFFER_SIZE} must be power of 2")
            
            # Validate port range
            if not (1024 <= cls.WEBSOCKET_PORT <= 65535):
                raise ValueError(f"Port {cls.WEBSOCKET_PORT} must be between 1024-65535")
            
            # Validate wait strategy
            if cls.WAIT_STRATEGY not in ['yielding', 'busy_spin']:
                raise ValueError(f"Invalid wait strategy: {cls.WAIT_STRATEGY}")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Development configuration for testing
class DevelopmentConfig(ProductionConfig):
    """Development configuration with relaxed settings"""
    
    WEBSOCKET_HOST = '127.0.0.1'
    BUFFER_SIZE = 16384  # Smaller buffer for development
    MAX_CLIENTS = 100
    LOG_LEVEL = 'DEBUG'
    ENABLE_PERFORMANCE_LOGGING = True
    HIGH_PERFORMANCE_MODE = False


# Export the appropriate config based on environment
config = DevelopmentConfig if os.getenv('FLASK_ENV') == 'development' else ProductionConfig