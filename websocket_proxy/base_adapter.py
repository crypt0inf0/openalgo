import json
import threading
import zmq
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

def find_free_zmq_port(start_port=5556, max_attempts=50):
    """
    Find an available port starting from start_port that's not already bound
    
    Args:
        start_port (int): Port number to start the search from
        max_attempts (int): Maximum number of attempts to find a free port
        
    Returns:
        int: Available port number, or None if no port is available
    """
    # Create logger here instead of using self.logger because this is a standalone function
    logger = get_logger("zmq_port_finder")
    
    # First check if any ports in the bound_ports set are actually free now
    # This handles cases where the process that had the port died without cleanup
    with BaseBrokerWebSocketAdapter._port_lock:
        ports_to_remove = [port for port in BaseBrokerWebSocketAdapter._bound_ports 
                          if is_port_available(port)]
        
        # Remove ports that are actually available now
        for port in ports_to_remove:
            BaseBrokerWebSocketAdapter._bound_ports.remove(port)
            logger.info(f"Port {port} removed from bound ports registry")
    
    # Now find a new free port
    for _ in range(max_attempts):
        # Try a sequential port first, then random if that fails
        if (start_port not in BaseBrokerWebSocketAdapter._bound_ports and 
            is_port_available(start_port)):
            return start_port
            
        # Try a random port between start_port and 65535
        random_port = random.randint(start_port, 65535)
        if (random_port not in BaseBrokerWebSocketAdapter._bound_ports and 
            is_port_available(random_port)):
            return random_port
            
        start_port = min(start_port + 1, 65000)
    
    # If we get here, we couldn't find an available port 
    logger.error("Failed to find an available port after maximum attempts")
    return None

class BaseBrokerWebSocketAdapter(ABC):
    """
    Base class for all broker-specific WebSocket adapters that implements
    common functionality and defines the interface for broker-specific implementations.
    """
    # Class variable to track bound ports across instances
    _bound_ports = set()
    _port_lock = threading.Lock()
    _shared_context = None
    _context_lock = threading.Lock()
    
    def __init__(self):
        self.logger = get_logger("broker_adapter")
        self.logger.info("BaseBrokerWebSocketAdapter initializing")
        
        try:
            # Initialize shared ZeroMQ context
            self._initialize_shared_context()
            
            # Create socket and bind to port
            self.socket = self._create_socket()
            self.zmq_port = self._bind_to_available_port()
            os.environ["ZMQ_PORT"] = str(self.zmq_port)
            
            # Initialize instance variables
            self.subscriptions = {}
            self.connected = False
            
            # Reference counting for WebSocket subscriptions
            # Prevents duplicate subscriptions when multiple clients request same symbol
            self._ws_subscription_refs = {}  # {scrip: {'touchline': 0, 'depth': 0}}
            self._ref_lock = threading.Lock()
            
            self.logger.info(f"BaseBrokerWebSocketAdapter initialized on port {self.zmq_port}")
            
        except Exception as e:
            self.logger.error(f"Error in BaseBrokerWebSocketAdapter init: {e}")
            raise
    
    def _increment_ref(self, scrip: str, sub_type: str = 'touchline') -> bool:
        """
        Increment reference count for a subscription.
        
        Args:
            scrip: Subscription key (e.g., 'NSE|1594' or 'RELIANCE_NSE')
            sub_type: Type of subscription - 'touchline' or 'depth'
            
        Returns:
            bool: True if this is the first subscription (should send to broker)
        """
        with self._ref_lock:
            if scrip not in self._ws_subscription_refs:
                self._ws_subscription_refs[scrip] = {'touchline': 0, 'depth': 0}
            
            is_first = self._ws_subscription_refs[scrip][sub_type] == 0
            self._ws_subscription_refs[scrip][sub_type] += 1
            
            if is_first:
                self.logger.info(f"First {sub_type} subscription for {scrip}")
            else:
                self.logger.debug(f"Additional {sub_type} subscription for {scrip} (count: {self._ws_subscription_refs[scrip][sub_type]})")
            
            return is_first
    
    def _decrement_ref(self, scrip: str, sub_type: str = 'touchline') -> bool:
        """
        Decrement reference count for a subscription.
        
        Args:
            scrip: Subscription key
            sub_type: Type of subscription - 'touchline' or 'depth'
            
        Returns:
            bool: True if this was the last subscription (should unsubscribe from broker)
        """
        with self._ref_lock:
            if scrip not in self._ws_subscription_refs:
                self.logger.warning(f"Attempted to decrement ref for unknown scrip: {scrip}")
                return False
            
            if self._ws_subscription_refs[scrip][sub_type] > 0:
                self._ws_subscription_refs[scrip][sub_type] -= 1
            
            is_last = self._ws_subscription_refs[scrip][sub_type] == 0
            
            if is_last:
                self.logger.info(f"Last {sub_type} subscription for {scrip}")
                
                # Clean up entry if no subscriptions remain
                if all(v == 0 for v in self._ws_subscription_refs[scrip].values()):
                    del self._ws_subscription_refs[scrip]
            
            return is_last
    
    def _get_ref_count(self, scrip: str, sub_type: str = 'touchline') -> int:
        """Get current reference count for a subscription."""
        with self._ref_lock:
            if scrip not in self._ws_subscription_refs:
                return 0
            return self._ws_subscription_refs[scrip].get(sub_type, 0)
    
    def _clear_all_refs(self) -> int:
        """Clear all subscription references. Returns count of cleared entries."""
        with self._ref_lock:
            count = len(self._ws_subscription_refs)
            self._ws_subscription_refs.clear()
            if count > 0:
                self.logger.info(f"Cleared all subscription references ({count} entries)")
            return count
    
    def _initialize_shared_context(self):
        """
        Initialize shared ZeroMQ context if not already created
        """
        with self._context_lock:
            if not BaseBrokerWebSocketAdapter._shared_context:
                self.logger.info("Creating shared ZMQ context")
                BaseBrokerWebSocketAdapter._shared_context = zmq.Context()
        
        self.context = BaseBrokerWebSocketAdapter._shared_context
    
    def _create_socket(self):
        """
        Create and configure ZeroMQ socket
        """
        with self._context_lock:
            socket = self.context.socket(zmq.PUB)
            socket.setsockopt(zmq.LINGER, 1000)  # 1 second linger
            socket.setsockopt(zmq.SNDHWM, 1000)  # High water mark
            return socket
        
    def _bind_to_available_port(self):
        """
        Find an available port and bind the socket to it
        """
        with self._port_lock:
            # Try default port from environment first
            default_port = int(os.getenv('ZMQ_PORT', '5555'))
            
            if (default_port not in self._bound_ports and 
                is_port_available(default_port)):
                try:
                    self.socket.bind(f"tcp://*:{default_port}")
                    self._bound_ports.add(default_port)
                    self.logger.info(f"Bound to default port {default_port}")
                    return default_port
                except zmq.ZMQError as e:
                    self.logger.warning(f"Failed to bind to default port {default_port}: {e}")
            
            # Find random available port
            for attempt in range(5):
                port = find_free_zmq_port(start_port=5556 + random.randint(0, 1000))
                
                if not port:
                    self.logger.warning(f"Failed to find free port on attempt {attempt+1}")
                    continue
                    
                try:
                    self.socket.bind(f"tcp://*:{port}")
                    self._bound_ports.add(port)
                    self.logger.info(f"Successfully bound to port {port}")
                    return port
                except zmq.ZMQError as e:
                    self.logger.warning(f"Failed to bind to port {port}: {e}")
                    continue
            
            raise RuntimeError("Could not bind to any available ZMQ port after multiple attempts")
        
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
            mode: Subscription mode - 1:LTP, 2:Quote, 4:Depth
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
        
    def cleanup_zmq(self):
        """
        Properly clean up ZeroMQ resources and release bound ports
        """
        try:
            # Release the port from the bound ports set
            if hasattr(self, 'zmq_port'):
                with self._port_lock:
                    self._bound_ports.discard(self.zmq_port)
                    self.logger.info(f"Released port {self.zmq_port}")
            
            # Close the socket
            if hasattr(self, 'socket') and self.socket:
                self.socket.close(linger=0)  # Don't linger on close
                self.logger.info("ZeroMQ socket closed")
                
        except Exception as e:
            self.logger.exception(f"Error cleaning up ZeroMQ resources: {e}")
            
    def __del__(self):
        """
        Destructor to ensure ZeroMQ resources are properly cleaned up
        """
        try:
            self.cleanup_zmq()
        except Exception as e:
            # Can't use self.logger here as it might be gone during destruction
            logger.exception(f"Error in __del__ cleaning up ZMQ resources: {e}")
            pass
    
    def publish_market_data(self, topic, data):
        """
        Publish market data to ZeroMQ subscribers
        
        Args:
            topic: Topic string for subscriber filtering (e.g., 'NSE_RELIANCE_LTP')
            data: Market data dictionary
        """
        try:
            self.socket.send_multipart([
                topic.encode('utf-8'),
                json.dumps(data).encode('utf-8')
            ])
        except Exception as e:
            self.logger.exception(f"Error publishing market data: {e}")
    
    def _generate_topic(self, exchange: str, symbol: str, mode_str: str) -> str:
        """
        Generate standardized topic for ZMQ publishing.
        
        All adapters should use this method to ensure consistent topic format
        that matches what the server expects: BROKER_EXCHANGE_SYMBOL_MODE
        
        Args:
            exchange: Exchange code (e.g., 'NSE', 'BSE', 'NFO')
            symbol: Trading symbol (e.g., 'RELIANCE', 'NIFTY')
            mode_str: Mode string (e.g., 'LTP', 'QUOTE', 'DEPTH')
            
        Returns:
            str: Topic in format BROKER_EXCHANGE_SYMBOL_MODE
        """
        broker = getattr(self, 'broker_name', 'unknown')
        return f"{broker}_{exchange}_{symbol}_{mode_str}"
    
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
