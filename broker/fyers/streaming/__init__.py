"""
Fyers streaming adapter initialization
Registers the adapter with the websocket proxy system
"""

from .fyers_adapter import FyersWebSocketAdapter

# Register the adapter with the broker factory
try:
    from websocket_proxy.adapters.broker_factory import register_adapter
    register_adapter('fyers', FyersWebSocketAdapter)
except ImportError:
    # If websocket_proxy is not available, skip registration
    pass

__all__ = ['FyersWebSocketAdapter']