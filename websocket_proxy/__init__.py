"""
WebSocket Proxy Package

This package provides high-performance WebSocket market data distribution
using a lightweight, efficient architecture for real-time streaming.

Architecture:
- core/: Lightweight WebSocket proxy
- adapters/: Broker-specific adapters and factories  
- data/: Market data structures and binary formats
- utils/: Utility functions
"""

import logging

# Core components
from .core import LightweightWebSocketProxy

# Integration functions
from .integration import (
    start_disruptor_proxy,
    cleanup_disruptor_server,
    get_disruptor_stats,
    publish_market_data,
    get_proxy_instance
)

# Adapters
from .adapters import BaseBrokerWebSocketAdapter, create_broker_adapter, register_adapter

# Data structures
from .data import BinaryMarketData, MarketDataMessage

# Configuration
from .production_config import ProductionConfig, DevelopmentConfig

# Utilities
from .utils import find_available_port, is_port_in_use

# Set up logger
logger = logging.getLogger(__name__)

# Register broker adapters
try:
    from broker.angel.streaming.angel_adapter import AngelWebSocketAdapter
    register_adapter("angel", AngelWebSocketAdapter)
    logger.info("Registered Angel adapter")
except ImportError:
    logger.debug("Angel adapter not available")

try:
    from broker.flattrade.streaming.flattrade_adapter import FlattradeWebSocketAdapter
    register_adapter("flattrade", FlattradeWebSocketAdapter)
    logger.info("Registered Flattrade adapter")
except ImportError:
    logger.debug("Flattrade adapter not available")

try:
    from broker.fyers.streaming.fyers_adapter import FyersWebSocketAdapter
    register_adapter("fyers", FyersWebSocketAdapter)
    logger.info("Registered Fyers adapter")
except ImportError:
    logger.debug("Fyers adapter not available")

__version__ = "2.1.0"
__all__ = [
    # Core components
    'LightweightWebSocketProxy',
    
    # Integration functions
    'start_disruptor_proxy',
    'cleanup_disruptor_server',
    'get_disruptor_stats',
    'publish_market_data',
    'get_proxy_instance',
    
    # Adapters
    'BaseBrokerWebSocketAdapter',
    'create_broker_adapter',
    'register_adapter',
    
    # Data structures
    'BinaryMarketData',
    'MarketDataMessage',
    
    # Configuration
    'ProductionConfig',
    'DevelopmentConfig',
    
    # Utilities
    'find_available_port',
    'is_port_in_use'
]