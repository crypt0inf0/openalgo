import json
import threading
import random
import socket
import os
from abc import ABC, abstractmethod
from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

def is_port_available(port):
    """
    Check if a port is available for use
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(1.0)
            s.bind(("127.0.0.1", port))
            return True
    except socket.error:
        return False

class BaseBrokerWebSocketAdapter(ABC):
    """
    Base class for all broker-specific WebSocket adapters that implements
    common functionality and defines the interface for broker-specific implementations.
    """
    
    def __init__(self):
        self.logger = get_logger("broker_adapter")
        self.logger.info("BaseBrokerWebSocketAdapter initializing")
        
        try:
            # Initialize instance variables
            self.subscriptions = {}
            self.connected = False
            
            # Get reference to the market data proxy for publishing
            # Removed lockfree_architecture dependency - using lightweight proxy
            self.market_proxy = None  # Will be set during initialization
            
            self.logger.info("BaseBrokerWebSocketAdapter initialized")
            
        except Exception as e:
            self.logger.error(f"Error in BaseBrokerWebSocketAdapter init: {e}")
            raise
        
    @abstractmethod
    def initialize(self, broker_name, user_id, auth_data=None):
        """
        Initialize connection with broker WebSocket API
        
        Args:
            broker_name: The name of the broker (e.g., 'angel', 'zerodha')
            user_id: The user's ID or client code
            auth_data: Dict containing authentication data, if not provided will fetch from DB
        """
        pass
        
    @abstractmethod
    def subscribe(self, symbol, exchange, mode=2, depth_level=5):
        """
        Subscribe to market data with the specified mode and depth level
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            exchange: Exchange code (e.g., 'NSE', 'BSE')
            mode: Subscription mode - 1:LTP, 2:Quote, 3:Depth
            depth_level: Market depth level (5, 20, or 30 depending on broker support)
            
        Returns:
            dict: Response with status and capability information
        """
        pass
        
    @abstractmethod
    def unsubscribe(self, symbol, exchange, mode=2):
        """
        Unsubscribe from market data
        
        Args:
            symbol: Trading symbol
            exchange: Exchange code
            mode: Subscription mode
            
        Returns:
            dict: Response with status
        """
        pass
        
    @abstractmethod
    def connect(self):
        """
        Establish connection to the broker's WebSocket
        """
        pass
        
    @abstractmethod
    def disconnect(self):
        """
        Disconnect from the broker's WebSocket
        """
        pass
    
    def publish_market_data(self, symbol, exchange, data):
        """
        Publish market data to the zero-backpressure proxy
        
        Args:
            symbol: Trading symbol
            exchange: Exchange code
            data: Market data dictionary
        """
        try:
            # Format the symbol as "EXCHANGE:SYMBOL" for the proxy
            formatted_symbol = f"{exchange}:{symbol}"
            
            # Create the market data message in the expected format
            market_message = {
                "type": "market_data",
                "symbol": symbol,
                "exchange": exchange,
                "broker": getattr(self, 'broker_name', 'unknown'),
                "data": data
            }
            
            # Publish to the zero-backpressure proxy if available
            if self.market_proxy:
                success = self.market_proxy.publish_market_data(formatted_symbol, market_message)
                if not success:
                    self.logger.warning(f"Failed to publish market data for {formatted_symbol}")
            else:
                self.logger.warning("Market proxy not available for publishing data")
                
        except Exception as e:
            self.logger.exception(f"Error publishing market data: {e}")
    
    def _create_success_response(self, message, **kwargs):
        """
        Create a standard success response
        """
        response = {
            'status': 'success',
            'message': message
        }
        response.update(kwargs)
        return response
    
    def _create_error_response(self, code, message):
        """
        Create a standard error response
        """
        return {
            'status': 'error',
            'code': code,
            'message': message
        }