"""
AngelOne WebSocket Adapter for OpenAlgo - Clean Implementation

This adapter provides a clean, robust implementation for AngelOne WebSocket streaming
with proper authentication flow, data normalization, and zero-backpressure publishing.
"""

import threading
import json
import logging
import time
from typing import Dict, Any, Optional, List
import os

from .smartWebSocketV2 import SmartWebSocketV2
from database.auth_db import get_auth_token, get_feed_token
from database.token_db import get_token
from websocket_proxy import get_proxy_instance
from websocket_proxy.adapters.mapping import SymbolMapper
from websocket_proxy.data.binary_market_data import BinaryMarketData
# Removed BaseBrokerWebSocketAdapter import since we're using zero-backpressure architecture
from .angel_mapping import AngelExchangeMapper, AngelCapabilityRegistry


class Config:
    """Configuration constants for AngelOne adapter"""
    MODE_LTP = 1
    MODE_QUOTE = 2
    MODE_DEPTH = 3
    
    # Angel exchange types
    NSE_CM = 1
    NSE_FO = 2
    BSE_CM = 3
    BSE_FO = 4
    MCX_FO = 5
    NCX_FO = 7
    CDE_FO = 13


class MarketDataCache:
    """Thread-safe cache for market data to handle partial updates with memory limits"""
    
    def __init__(self, max_size: int = 10000):
        self._cache = {}
        self._access_times = {}  # For LRU eviction
        self._lock = threading.Lock()
        self.max_size = max_size
        self.eviction_batch_size = max(100, max_size // 100)  # Evict 1% or min 100 items
        self.logger = logging.getLogger("angel_market_cache")
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def update(self, token: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update cache with new data and return merged result"""
        with self._lock:
            import time
            current_time = time.time()
            
            # Check if we need to evict before adding
            if len(self._cache) >= self.max_size and token not in self._cache:
                self._evict_lru_items()
            
            cached_data = self._cache.get(token, {})
            if cached_data:
                self.hits += 1
            else:
                self.misses += 1
            
            merged = cached_data.copy()
            
            # Update with non-empty values
            merged.update({k: v for k, v in data.items() if v not in [None, '', '-']})
            
            # Preserve existing fields not in new data
            for k, v in cached_data.items():
                if k not in merged:
                    merged[k] = v
                    
            self._cache[token] = merged
            self._access_times[token] = current_time
            
            return merged.copy()

    def _evict_lru_items(self):
        """Evict least recently used items to free memory"""
        if not self._access_times:
            return
            
        # Sort by access time and remove oldest items
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        items_to_remove = sorted_items[:self.eviction_batch_size]
        
        for token, _ in items_to_remove:
            self._cache.pop(token, None)
            self._access_times.pop(token, None)
            self.evictions += 1
        
        self.logger.info(f"Evicted {len(items_to_remove)} items from cache. "
                        f"Cache size: {len(self._cache)}/{self.max_size}")

    def clear_token(self, token: str):
        """Clear cache for a specific token"""
        with self._lock:
            self._cache.pop(token, None)
            self._access_times.pop(token, None)

    def clear_all(self):
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'utilization': (len(self._cache) / self.max_size * 100) if self.max_size > 0 else 0,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions
            }


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    if value is None or value == '' or value == '-':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    if value is None or value == '' or value == '-':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


class AngelWebSocketAdapter:
    """
    AngelOne WebSocket Adapter with clean architecture
    
    Features:
    - Proper authentication flow with database integration
    - Robust connection management with auto-reconnection
    - Data normalization and caching
    - Zero-backpressure publishing
    - Comprehensive error handling
    """

    def __init__(self):
        self.logger = logging.getLogger("angel_adapter")
        
        # Connection state
        self.ws_client: Optional[SmartWebSocketV2] = None
        self.connected = False
        self.running = False
        
        # Authentication
        self.auth_token: Optional[str] = None
        self.feed_token: Optional[str] = None
        self.api_key: Optional[str] = None
        self.client_code: Optional[str] = None
        
        # Data management with memory limits
        cache_size = int(os.getenv('ANGEL_CACHE_SIZE', '10000'))
        self.market_cache = MarketDataCache(max_size=cache_size)
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.token_to_symbol: Dict[str, tuple] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Reconnection management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
        
        # Memory monitoring
        self.max_memory_mb = int(os.getenv('MAX_MEMORY_MB', '2048'))
        self.last_memory_check = time.time()
        self.memory_check_interval = 60  # Check every minute
        
        # Zero-backpressure proxy
        self.zbp_proxy = None
        self.logger.info("Angel adapter constructor: zbp_proxy initialized to None")

    def initialize(self, broker_name: str, user_id: str, auth_data: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Initialize the adapter with authentication credentials
        
        Args:
            broker_name: Name of the broker (should be 'angel')
            user_id: Client ID/user ID
            auth_data: Optional auth data, if not provided will fetch from database
            
        Raises:
            ValueError: If required authentication tokens are not found
        """
        self.user_id = user_id
        self.broker_name = broker_name
        
        # Get authentication tokens
        if auth_data:
            self.auth_token = auth_data.get('auth_token')
            self.feed_token = auth_data.get('feed_token')
            self.api_key = auth_data.get('api_key')
            self.client_code = user_id
        else:
            # Fetch from database
            self.auth_token = get_auth_token(user_id)
            self.feed_token = get_feed_token(user_id)
            self.client_code = user_id
            
            # Extract API key from auth token
            self.api_key = self._extract_api_key_from_token(self.auth_token)
        
        # Debug: Log token status
        self.logger.info(f"Token validation - auth_token: {'✓' if self.auth_token else '✗'}, feed_token: {'✓' if self.feed_token else '✗'}, api_key: {'✓' if self.api_key else '✗'}, client_code: {'✓' if self.client_code else '✗'}")
        
        # Validate required tokens
        if not all([self.auth_token, self.feed_token, self.api_key, self.client_code]):
            missing = []
            if not self.auth_token: missing.append('auth_token')
            if not self.feed_token: missing.append('feed_token')
            if not self.api_key: missing.append('api_key')
            if not self.client_code: missing.append('client_code')
            
            error_msg = f"Missing required authentication data: {', '.join(missing)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create SmartWebSocketV2 instance now that tokens are available
        self.ws_client = SmartWebSocketV2(
            auth_token=self.auth_token,
            api_key=self.api_key,
            client_code=self.client_code,
            feed_token=self.feed_token,
            max_retry_attempt=5
        )
        
        # Set up callbacks with proper signatures
        self.ws_client.on_open = self._on_open
        self.ws_client.on_data = self._on_data
        self.ws_client.on_error = self._on_error
        self.ws_client.on_close = self._on_close
        self.ws_client.on_message = self._on_message
        
        self.running = True
        self.logger.info(f"AngelOne adapter initialized for user {user_id}")
        
        return {"success": True, "message": f"AngelOne adapter initialized for user {user_id}"}
        
    def _extract_api_key_from_token(self, auth_token: str) -> str:
        """Extract API key from JWT auth token"""
        try:
            if not auth_token:
                return ''
            
            # JWT tokens have 3 parts separated by dots
            parts = auth_token.split('.')
            if len(parts) != 3:
                self.logger.warning("Invalid JWT token format")
                return ''
            
            # Decode the payload (second part)
            import json
            import base64
            
            # Add padding if needed
            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += '=' * padding
            
            payload_json = base64.b64decode(payload_b64).decode('utf-8')
            payload = json.loads(payload_json)
            
            # Extract API key
            api_key = payload.get('API-KEY', '')
            if api_key:
                self.logger.info(f"Extracted API key from token: {api_key[:6]}...")
                return api_key
            else:
                self.logger.warning("No API-KEY found in token payload")
                return ''
                
        except Exception as e:
            self.logger.error(f"Error extracting API key from token: {e}")
            return ''

    def set_proxy(self, proxy):
        """Set the zero-backpressure proxy reference"""
        self.zbp_proxy = proxy
        self.logger.info(f"Angel adapter: zbp_proxy set to {type(proxy)}, has_publish_method={hasattr(proxy, 'publish_market_data')}")

    def connect(self) -> Dict[str, Any]:
        """Establish connection to AngelOne WebSocket"""
        if not self.ws_client:
            self.logger.error("WebSocket client not initialized. Call initialize() first.")
            return {"success": False, "error": "WebSocket client not initialized"}
        
        self.logger.info("Connecting to AngelOne WebSocket...")
        threading.Thread(target=self._connect_with_retry, daemon=True).start()
        
        return {"success": True, "message": "AngelOne WebSocket connection initiated"}

    def disconnect(self) -> None:
        """Disconnect from AngelOne WebSocket"""
        self.running = False
        self.connected = False
        
        if self.ws_client:
            try:
                self.ws_client.close_connection()
                self.logger.info("Disconnected from AngelOne WebSocket")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
        
        # Clear cache and subscriptions
        self.market_cache.clear_all()
        with self.lock:
            self.subscriptions.clear()
            self.token_to_symbol.clear()

    def subscribe(self, symbol: str, exchange: str, mode: int = 2, depth_level: int = 5) -> Dict[str, Any]:
        """
        Subscribe to market data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            exchange: Exchange code (e.g., 'NSE', 'BSE', 'NFO')
            mode: Subscription mode (1=LTP, 2=Quote, 3=Depth)
            depth_level: Market depth level (5, 20, 30)
            
        Returns:
            Dict: Response with status and details
        """
        # Validate parameters
        if not symbol or not exchange:
            return self._create_error_response("INVALID_PARAMS", "Symbol and exchange are required")
        
        if mode not in [Config.MODE_LTP, Config.MODE_QUOTE, Config.MODE_DEPTH]:
            return self._create_error_response("INVALID_MODE", f"Invalid mode: {mode}")
        
        # Get token information
        token_info = SymbolMapper.get_token_from_symbol(symbol, exchange)
        if not token_info:
            return self._create_error_response("SYMBOL_NOT_FOUND", 
                                             f"Symbol {symbol} not found for exchange {exchange}")
        
        token = token_info['token']
        brexchange = token_info['brexchange']
        
        # Check depth level support for depth mode
        actual_depth = depth_level
        is_fallback = False
        
        if mode == Config.MODE_DEPTH:
            if not AngelCapabilityRegistry.is_depth_level_supported(exchange, depth_level):
                actual_depth = AngelCapabilityRegistry.get_fallback_depth_level(exchange, depth_level)
                is_fallback = True
                self.logger.info(f"Using depth level {actual_depth} instead of {depth_level} for {exchange}")
        
        # Create subscription
        correlation_id = f"{symbol}_{exchange}_{mode}"
        token_list = [{
            "exchangeType": AngelExchangeMapper.get_exchange_type(brexchange),
            "tokens": [token]
        }]
        
        subscription = {
            'symbol': symbol,
            'exchange': exchange,
            'brexchange': brexchange,
            'token': token,
            'mode': mode,
            'depth_level': depth_level,
            'actual_depth': actual_depth,
            'token_list': token_list,
            'is_fallback': is_fallback,
            'correlation_id': correlation_id
        }
        
        # Store subscription
        with self.lock:
            self.subscriptions[correlation_id] = subscription
            self.token_to_symbol[token] = (symbol, exchange)
        
        # Subscribe if connected
        if self.connected and self.ws_client:
            try:
                self.ws_client.subscribe(correlation_id, mode, token_list)
                self.logger.debug(f"Subscribed to {symbol}.{exchange} mode={mode}")
            except Exception as e:
                self.logger.error(f"Error subscribing to {symbol}.{exchange}: {e}")
                return self._create_error_response("SUBSCRIPTION_ERROR", str(e))
        else:
            self.logger.warning(f"Cannot subscribe to {symbol}.{exchange} - not connected (connected={self.connected}, ws_client={self.ws_client is not None})")
        
        # Return success response
        message = f"Subscribed to {symbol}.{exchange}"
        if is_fallback:
            message += f" (using depth level {actual_depth})"
        
        return self._create_success_response(
            message,
            symbol=symbol,
            exchange=exchange,
            mode=mode,
            requested_depth=depth_level,
            actual_depth=actual_depth,
            is_fallback=is_fallback
        )

    def unsubscribe(self, symbol: str, exchange: str, mode: int = 2) -> Dict[str, Any]:
        """
        Unsubscribe from market data for a symbol
        
        Args:
            symbol: Trading symbol
            exchange: Exchange code
            mode: Subscription mode
            
        Returns:
            Dict: Response with status
        """
        if not symbol or not exchange:
            return self._create_error_response("INVALID_PARAMS", "Symbol and exchange are required")
        
        if mode not in [Config.MODE_LTP, Config.MODE_QUOTE, Config.MODE_DEPTH]:
            return self._create_error_response("INVALID_MODE", f"Invalid mode: {mode}")
        
        correlation_id = f"{symbol}_{exchange}_{mode}"
        
        with self.lock:
            if correlation_id not in self.subscriptions:
                return self._create_success_response(f"Already unsubscribed from {symbol}.{exchange}")
            
            subscription = self.subscriptions[correlation_id]
            token = subscription['token']
            
            # Unsubscribe from WebSocket
            if self.connected and self.ws_client:
                try:
                    self.ws_client.unsubscribe(correlation_id, mode, subscription['token_list'])
                    self.logger.debug(f"Unsubscribed from {symbol}.{exchange} mode={mode}")
                except Exception as e:
                    self.logger.warning(f"Error during WebSocket unsubscribe: {e}")
            
            # Remove subscription
            del self.subscriptions[correlation_id]
            
            # Clean up token mapping if no other subscriptions use this token
            if not any(sub.get('token') == token for sub in self.subscriptions.values()):
                self.token_to_symbol.pop(token, None)
                self.market_cache.clear_token(token)
        
        return self._create_success_response(f"Unsubscribed from {symbol}.{exchange}")

    # WebSocket callback methods with proper signatures
    def _on_open(self, wsapp=None) -> None:
        """Callback when WebSocket connection is established"""
        self.logger.info("Connected to AngelOne WebSocket")
        self.connected = True
        self.reconnect_attempts = 0
        
        # Resubscribe to existing subscriptions
        # Resubscribe to existing subscriptions
        self._resubscribe_all()

    def _on_close(self, wsapp=None, close_status_code=None) -> None:
        """Callback when WebSocket connection is closed"""
        self.logger.info("AngelOne WebSocket connection closed")
        self.connected = False
        
        # Attempt reconnection if still running
        if self.running:
            self._schedule_reconnection()

    def _on_error(self, wsapp=None, error=None) -> None:
        """Callback for WebSocket errors"""
        self.logger.error(f"AngelOne WebSocket error: {error}")

    def _on_message(self, wsapp=None, message=None) -> None:
        """Callback for text messages"""
        self.logger.debug(f"Received text message: {message}")

    def _on_data(self, wsapp=None, message=None) -> None:
        """
        Callback for market data messages
        
        This method handles both callback signature variations:
        - _on_data(wsapp, message) - from SmartWebSocketV2
        - _on_data(message) - direct call
        """
        try:
            # Handle different callback signatures
            if wsapp is not None and message is None:
                # If wsapp is provided but message is None, wsapp might be the message
                actual_message = wsapp
            else:
                actual_message = message
            
            if actual_message is None:
                self.logger.warning("Received None message in _on_data")
                return
            
            self.logger.debug(f"Processing market data: {type(actual_message)}")
            
            # Handle binary data
            if isinstance(actual_message, (bytes, bytearray)):
                self.logger.debug(f"Received binary data of length: {len(actual_message)}")
                return
            
            # Process dictionary message (parsed data from SmartWebSocketV2)
            if not isinstance(actual_message, dict):
                self.logger.warning(f"Received non-dict message: {type(actual_message)}")
                return
            
            # Extract token and exchange_type
            token = actual_message.get('token')
            exchange_type = actual_message.get('exchange_type')
            
            if not token:
                self.logger.warning("Received message without token")
                return
            
            # Find matching subscription
            subscription = self._find_subscription_by_token(token, exchange_type)
            if not subscription:
                self.logger.warning(f"Received data for unsubscribed token: {token}")
                return
            
            # Process the message
            self._process_market_message(actual_message, subscription)
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}", exc_info=True)

    def _connect_with_retry(self) -> None:
        """Connect with retry logic"""
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.logger.info(f"Connecting to AngelOne WebSocket (attempt {self.reconnect_attempts + 1})")
                self.ws_client.connect()
                break
            except Exception as e:
                self.reconnect_attempts += 1
                delay = min(self.reconnect_delay * (2 ** self.reconnect_attempts), self.max_reconnect_delay)
                self.logger.error(f"Connection failed: {e}. Retrying in {delay} seconds...")
                if self.running:
                    time.sleep(delay)
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")

    def _schedule_reconnection(self) -> None:
        """Schedule reconnection with exponential backoff"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            delay = min(self.reconnect_delay * (2 ** self.reconnect_attempts), self.max_reconnect_delay)
            self.reconnect_attempts += 1
            self.logger.info(f"Scheduling reconnection in {delay} seconds (attempt {self.reconnect_attempts})")
            threading.Timer(delay, self._attempt_reconnection).start()

    def _attempt_reconnection(self) -> None:
        """Attempt to reconnect"""
        if self.running:
            threading.Thread(target=self._connect_with_retry, daemon=True).start()

    def _resubscribe_all(self) -> None:
        """Resubscribe to all existing subscriptions after reconnection"""
        with self.lock:
            if not self.subscriptions:
                self.logger.debug("No subscriptions to resubscribe")
                return
                
            for correlation_id, subscription in self.subscriptions.items():
                try:
                    self.ws_client.subscribe(
                        correlation_id, 
                        subscription["mode"], 
                        subscription["token_list"]
                    )
                    self.logger.info(f"Resubscribed to {subscription['symbol']}.{subscription['exchange']}")
                except Exception as e:
                    self.logger.error(f"Error resubscribing to {subscription['symbol']}.{subscription['exchange']}: {e}")

    def _find_subscription_by_token(self, token: str, exchange_type: int) -> Optional[Dict[str, Any]]:
        """Find subscription matching token and exchange type"""
        with self.lock:
            for subscription in self.subscriptions.values():
                if (subscription['token'] == token and 
                    AngelExchangeMapper.get_exchange_type(subscription['brexchange']) == exchange_type):
                    return subscription
        return None

    def _process_market_message(self, data: Dict[str, Any], subscription: Dict[str, Any]) -> None:
        """Process market message and publish to zero-backpressure proxy"""
        try:
            symbol = subscription['symbol']
            exchange = subscription['exchange']
            mode = subscription['mode']
            token = subscription['token']
            
            # Periodic memory check
            self._check_memory_usage()
            
            # Update cache with new data
            cached_data = self.market_cache.update(token, data)
            
            # Normalize the data
            normalized = self._normalize_market_data(cached_data, mode)
            normalized.update({
                'symbol': symbol,
                'exchange': exchange,
                'timestamp': int(time.time() * 1000)
            })
            
            # Add broker timestamp for latency measurement
            broker_timestamp_ms = time.time() * 1000  # Current time in milliseconds
            normalized['broker_timestamp'] = broker_timestamp_ms
            
            # Publish to zero-backpressure proxy
            self.logger.info(f"[ANGEL] About to publish {symbol}, zbp_proxy={self.zbp_proxy is not None}, zbp_proxy_type={type(self.zbp_proxy)}, has_method={hasattr(self.zbp_proxy, 'publish_market_data') if self.zbp_proxy else False}")
            
            if self.zbp_proxy:
                if hasattr(self.zbp_proxy, 'publish_market_data'):
                    self.logger.info(f"[ANGEL] Calling publish_market_data for {symbol}")
                    try:
                        success = self.zbp_proxy.publish_market_data(symbol, normalized)
                        if success:
                            self.logger.info(f"Published {symbol} data: LTP={normalized.get('ltp', 'N/A')}, Mode={normalized.get('mode', 'N/A')}, Broker_TS={broker_timestamp_ms}")
                        else:
                            self.logger.warning(f"Failed to publish market data for {symbol}")
                    except Exception as e:
                        self.logger.error(f"Exception calling publish_market_data for {symbol}: {e}", exc_info=True)
                else:
                    self.logger.error(f"zbp_proxy does not have publish_market_data method! Type: {type(self.zbp_proxy)}")
            else:
                self.logger.warning("Zero-backpressure proxy not available")
                
        except Exception as e:
            self.logger.error(f"Error processing market message: {e}", exc_info=True)

    def _normalize_market_data(self, message: Dict[str, Any], mode: int) -> Dict[str, Any]:
        """
        Normalize AngelOne data format to standard format
        
        Args:
            message: Raw message from AngelOne
            mode: Subscription mode
            
        Returns:
            Dict: Normalized market data
        """
        # AngelOne sends prices in paise (1/100 rupee), so divide by 100
        
        if mode == Config.MODE_LTP:
            return {
                'mode': Config.MODE_LTP,
                'ltp': safe_float(message.get('last_traded_price', 0)) / 100,
                'ltt': safe_int(message.get('exchange_timestamp', 0)),
                'angel_timestamp': safe_int(message.get('exchange_timestamp', 0))
            }
            
        elif mode == Config.MODE_QUOTE:
            return {
                'mode': Config.MODE_QUOTE,
                'ltp': safe_float(message.get('last_traded_price', 0)) / 100,
                'volume': safe_int(message.get('volume_trade_for_the_day', 0)),
                'open': safe_float(message.get('open_price_of_the_day', 0)) / 100,
                'high': safe_float(message.get('high_price_of_the_day', 0)) / 100,
                'low': safe_float(message.get('low_price_of_the_day', 0)) / 100,
                'close': safe_float(message.get('closed_price', 0)) / 100,
                'last_quantity': safe_int(message.get('last_traded_quantity', 0)),
                'average_price': safe_float(message.get('average_traded_price', 0)) / 100,
                'total_buy_quantity': safe_int(message.get('total_buy_quantity', 0)),
                'total_sell_quantity': safe_int(message.get('total_sell_quantity', 0)),
                'ltt': safe_int(message.get('exchange_timestamp', 0)),
                'angel_timestamp': safe_int(message.get('exchange_timestamp', 0))
            }
            
        elif mode == Config.MODE_DEPTH:
            result = {
                'mode': Config.MODE_DEPTH,
                'ltp': safe_float(message.get('last_traded_price', 0)) / 100,
                'volume': safe_int(message.get('volume_trade_for_the_day', 0)),
                'open': safe_float(message.get('open_price', 0)) / 100,
                'high': safe_float(message.get('high_price', 0)) / 100,
                'low': safe_float(message.get('low_price', 0)) / 100,
                'close': safe_float(message.get('close_price', 0)) / 100,
                'oi': safe_int(message.get('open_interest', 0)),
                'upper_circuit': safe_float(message.get('upper_circuit_limit', 0)) / 100,
                'lower_circuit': safe_float(message.get('lower_circuit_limit', 0)) / 100,
                'ltt': safe_int(message.get('exchange_timestamp', 0)),
                'angel_timestamp': safe_int(message.get('exchange_timestamp', 0))
            }
            
            # Add depth data
            result['depth'] = {
                'buy': self._extract_depth_data(message, is_buy=True),
                'sell': self._extract_depth_data(message, is_buy=False)
            }
            result['depth_level'] = 5
            
            return result
        
        return {}

    def _extract_depth_data(self, message: Dict[str, Any], is_buy: bool) -> List[Dict[str, Any]]:
        """Extract depth data from AngelOne message format"""
        depth = []
        side_label = 'Buy' if is_buy else 'Sell'
        
        # Try different possible depth data keys
        possible_keys = [
            f'best_5_{"buy" if is_buy else "sell"}_data',
            f'depth_20_{"buy" if is_buy else "sell"}_data',
            f'best_five_{"buy" if is_buy else "sell"}_market_data'
        ]
        
        depth_data = None
        for key in possible_keys:
            if key in message and isinstance(message[key], list):
                depth_data = message[key]
                self.logger.debug(f"Found {side_label} depth data using {key}: {len(depth_data)} levels")
                break
        
        if depth_data:
            for level in depth_data:
                if isinstance(level, dict):
                    price = level.get('price', 0)
                    if price > 0:
                        price = price / 100  # Convert from paise to rupees
                    depth.append({
                        'price': price,
                        'quantity': level.get('quantity', 0),
                        'orders': level.get('no of orders', 0)
                    })
        else:
            # Return empty levels if no depth data found
            self.logger.debug(f"No {side_label} depth data found")
            for i in range(5):
                depth.append({'price': 0.0, 'quantity': 0, 'orders': 0})
        
        return depth

    def _check_memory_usage(self) -> None:
        """Check memory usage and perform cleanup if needed"""
        try:
            current_time = time.time()
            if current_time - self.last_memory_check < self.memory_check_interval:
                return
                
            self.last_memory_check = current_time
            
            # Try to get memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > self.max_memory_mb:
                    self.logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_mb}MB. "
                                      f"Performing cleanup...")
                    self._force_memory_cleanup()
                elif memory_mb > self.max_memory_mb * 0.8:  # 80% threshold warning
                    self.logger.info(f"Memory usage {memory_mb:.1f}MB approaching limit {self.max_memory_mb}MB")
                    
            except ImportError:
                # psutil not available, use basic cleanup based on cache size
                cache_stats = self.market_cache.get_stats()
                if cache_stats['utilization'] > 90:  # 90% cache utilization
                    self.logger.warning("Cache utilization high, performing cleanup...")
                    self._force_memory_cleanup()
                    
        except Exception as e:
            self.logger.error(f"Error checking memory usage: {e}")

    def _force_memory_cleanup(self) -> None:
        """Force memory cleanup when limits are exceeded"""
        try:
            # Clear half the cache
            with self.lock:
                cache_size = len(self.market_cache._cache)
                if cache_size > 100:
                    # Reduce cache size by 50%
                    self.market_cache.max_size = max(1000, cache_size // 2)
                    self.market_cache._evict_lru_items()
                    
            # Clear unused subscriptions (those not in active subscriptions)
            active_tokens = set(sub.get('token') for sub in self.subscriptions.values())
            with self.lock:
                cached_tokens = set(self.market_cache._cache.keys())
                unused_tokens = cached_tokens - active_tokens
                
                for token in list(unused_tokens)[:1000]:  # Limit cleanup batch
                    self.market_cache.clear_token(token)
                    
            self.logger.info(f"Memory cleanup completed. Removed {len(unused_tokens)} unused cache entries")
            
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")

    def cleanup(self) -> None:
        """Clean up adapter resources"""
        try:
            self.disconnect()
            self.market_cache.clear_all()
            self.logger.info("Angel adapter cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure resources are properly cleaned up"""
        try:
            self.cleanup()
        except Exception:
            pass

    def _create_success_response(self, message: str, **kwargs) -> Dict[str, Any]:
        """Create standardized success response"""
        response = {"status": "success", "message": message}
        response.update(kwargs)
        return response

    def _create_error_response(self, code: str, message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {"status": "error", "code": code, "message": message}


# For backward compatibility
AngelSHMWebSocketAdapter = AngelWebSocketAdapter
AngelZBPWebSocketAdapter = AngelWebSocketAdapter