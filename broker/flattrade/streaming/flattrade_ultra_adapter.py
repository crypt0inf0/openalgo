"""
Flattrade Ultra-Low Latency WebSocket Adapter
Optimized for minimal latency market data processing
"""

import time
import threading
from typing import Dict, Any, Optional
from websocket_proxy.ultra_low_latency_adapter import UltraLowLatencyAdapter
from websocket_proxy.cross_platform_structures import MarketTick, MessageType
from .flattrade_websocket import FlattradeWebSocket
from .flattrade_mapping import FlattradeExchangeMapper
from websocket_proxy.mapping import SymbolMapper
from database.auth_db import get_auth_token
import os
import logging
import pickle

class FlattradeUltraAdapter(UltraLowLatencyAdapter):
    """Ultra-low latency Flattrade WebSocket adapter"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("flattrade_ultra_adapter")
        self.ws_client = None
        self.subscriptions = {}
        self.token_to_symbol_mappings = {}  # Changed from token_to_symbol_id to support multiple mappings
        self._symbol_id_counter = 1000  # Start from 1000 for user symbols
        self.ws_subscription_refs = {}  # Reference counting for WebSocket subscriptions
        
    def initialize(self, broker_name: str, user_id: str, auth_data: Optional[Dict[str, str]] = None) -> None:
        """Initialize ultra adapter"""
        super().initialize(broker_name, user_id, auth_data)
        
        # Get Flattrade credentials
        api_key = os.getenv('BROKER_API_KEY', '')
        if ':::' in api_key:
            self.actid = api_key.split(':::')[0]
        else:
            self.actid = user_id
            
        self.susertoken = get_auth_token(user_id)
        
        if not self.actid or not self.susertoken:
            raise ValueError(f"Missing Flattrade credentials for user {user_id}")
            
        self.logger.info(f"Initialized Flattrade ultra adapter for user {user_id}")
        
    def connect(self) -> None:
        """Connect to Flattrade WebSocket"""
        self.ws_client = FlattradeWebSocket(
            user_id=self.actid,
            actid=self.actid,
            susertoken=self.susertoken,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        if self.ws_client.connect():
            self.connected = True
            self.logger.info("Connected to Flattrade WebSocket")
        else:
            raise ConnectionError("Failed to connect to Flattrade WebSocket")
            
    def disconnect(self) -> None:
        """Disconnect from Flattrade WebSocket"""
        if self.ws_client:
            self.ws_client.stop()
        super().disconnect()
        self.logger.info("Disconnected from Flattrade WebSocket")
        
    def subscribe(self, symbol: str, exchange: str, mode: int = 1, depth_level: int = 5) -> Dict[str, Any]:
        """Subscribe to market data"""
        try:
            # Get token info
            token_info = SymbolMapper.get_token_from_symbol(symbol, exchange)
            if not token_info:
                return self._create_error_response("SYMBOL_NOT_FOUND", f"Symbol {symbol} not found")
                
            token = token_info['token']
            brexchange = token_info['brexchange']
            flattrade_exchange = FlattradeExchangeMapper.to_flattrade_exchange(brexchange)
            scrip = f"{flattrade_exchange}|{token}"
            
            correlation_id = f"{symbol}_{exchange}_{mode}"
            
            # Check if already subscribed to this exact subscription
            if correlation_id in self.subscriptions:
                return self._create_error_response("ALREADY_SUBSCRIBED", 
                    f"Already subscribed to {symbol}.{exchange} mode {mode}")
            
            # Generate unique symbol ID for this subscription
            symbol_id = self._get_symbol_id(symbol, exchange, mode)
            
            # Store token to symbol mapping for this specific subscription
            if token not in self.token_to_symbol_mappings:
                self.token_to_symbol_mappings[token] = []
            
            # Always add the mapping for this subscription
            self.token_to_symbol_mappings[token].append({
                'symbol_id': symbol_id,
                'symbol': symbol,
                'exchange': exchange,
                'mode': mode,
                'correlation_id': correlation_id
            })
            
            # Store subscription
            subscription = {
                'symbol': symbol,
                'exchange': exchange,
                'mode': mode,
                'token': token,
                'scrip': scrip,
                'symbol_id': symbol_id
            }
            
            self.subscriptions[correlation_id] = subscription
            
            self.logger.info(f"Adding subscription: {correlation_id} -> symbol_id: {symbol_id}")
            self.logger.info(f"Token {token} now has mappings: {self.token_to_symbol_mappings[token]}")
            
            # Handle WebSocket subscription with reference counting
            self._websocket_subscribe(subscription)
                
            return self._create_success_response(
                f'Subscribed to {symbol}.{exchange}',
                symbol=symbol, exchange=exchange, mode=mode, symbol_id=symbol_id
            )
            
        except Exception as e:
            self.logger.error(f"Subscription error: {e}")
            return self._create_error_response("SUBSCRIPTION_ERROR", str(e))

    def unsubscribe(self, symbol: str, exchange: str, mode: int = 1) -> Dict[str, Any]:
        """Unsubscribe from market data"""
        try:
            correlation_id = f"{symbol}_{exchange}_{mode}"
            
            if correlation_id not in self.subscriptions:
                return self._create_error_response("NOT_SUBSCRIBED", 
                    f"Not subscribed to {symbol}.{exchange}")
            
            subscription = self.subscriptions[correlation_id]
            
            # Handle WebSocket unsubscription with reference counting
            self._websocket_unsubscribe(subscription)
            
            # Remove subscription
            self._remove_subscription(correlation_id, subscription)
            
            return self._create_success_response(
                f"Unsubscribed from {symbol}.{exchange}",
                symbol=symbol, exchange=exchange, mode=mode
            )
            
        except Exception as e:
            self.logger.error(f"Unsubscription error: {e}")
            return self._create_error_response("UNSUBSCRIPTION_ERROR", str(e))

    def _websocket_subscribe(self, subscription: Dict) -> None:
        """Handle WebSocket subscription with reference counting"""
        scrip = subscription['scrip']
        mode = subscription['mode']
        
        # Initialize reference count for this scrip if not exists
        if scrip not in self.ws_subscription_refs:
            self.ws_subscription_refs[scrip] = {'touchline_count': 0, 'depth_count': 0}
        
        if mode in [1, 2]:  # LTP, Quote
            if self.ws_subscription_refs[scrip]['touchline_count'] == 0:
                self.logger.info(f"First touchline subscription for {scrip}")
                self.ws_client.subscribe_touchline(scrip)
            self.ws_subscription_refs[scrip]['touchline_count'] += 1
        elif mode == 3:  # Depth
            if self.ws_subscription_refs[scrip]['depth_count'] == 0:
                self.logger.info(f"First depth subscription for {scrip}")
                self.ws_client.subscribe_depth(scrip)
            self.ws_subscription_refs[scrip]['depth_count'] += 1

    def _websocket_unsubscribe(self, subscription: Dict) -> None:
        """Handle WebSocket unsubscription with reference counting"""
        scrip = subscription['scrip']
        mode = subscription['mode']
        
        if scrip not in self.ws_subscription_refs:
            return
        
        if mode in [1, 2]:  # LTP, Quote
            self.ws_subscription_refs[scrip]['touchline_count'] -= 1
            if self.ws_subscription_refs[scrip]['touchline_count'] <= 0:
                self.logger.info(f"Last touchline subscription for {scrip}")
                self.ws_client.unsubscribe_touchline(scrip)
                self.ws_subscription_refs[scrip]['touchline_count'] = 0
        elif mode == 3:  # Depth
            self.ws_subscription_refs[scrip]['depth_count'] -= 1
            if self.ws_subscription_refs[scrip]['depth_count'] <= 0:
                self.logger.info(f"Last depth subscription for {scrip}")
                self.ws_client.unsubscribe_depth(scrip)
                self.ws_subscription_refs[scrip]['depth_count'] = 0

    def _remove_subscription(self, correlation_id: str, subscription: Dict) -> None:
        """Remove subscription and clean up mappings"""
        token = subscription['token']
        scrip = subscription['scrip']
        symbol = subscription['symbol']
        exchange = subscription['exchange']
        mode = subscription['mode']
        symbol_id = subscription['symbol_id']
        
        # Remove subscription
        del self.subscriptions[correlation_id]
        
        # Remove from token mappings
        if token in self.token_to_symbol_mappings:
            self.token_to_symbol_mappings[token] = [
                mapping for mapping in self.token_to_symbol_mappings[token]
                if mapping.get('correlation_id') != correlation_id
            ]
            
            # Clean up empty mappings
            if not self.token_to_symbol_mappings[token]:
                del self.token_to_symbol_mappings[token]
        
        # Clean up reference count if both counts are 0
        if scrip in self.ws_subscription_refs:
            if (self.ws_subscription_refs[scrip]['touchline_count'] <= 0 and
                self.ws_subscription_refs[scrip]['depth_count'] <= 0):
                del self.ws_subscription_refs[scrip]
            
    def direct_publish(self, tick: MarketTick):
        """Publish market data directly to the ring buffer"""
        try:
            self.logger.debug(f"Original MarketTick: {tick}")
            
            # Publish the tick directly to the shared ring buffer
            if UltraLowLatencyAdapter._shared_ring_buffer is not None:
                # Convert MarketTick to bytes using the correct serialization method
                tick_data = tick.to_bytes()
                if UltraLowLatencyAdapter._shared_ring_buffer.publish(tick_data):
                    self.logger.debug("Successfully published market data to shared ring buffer")
                else:
                    self.logger.error("Failed to publish to ring buffer - buffer may be full")
            else:
                self.logger.error("Shared ring buffer not available")
                
        except Exception as e:
            self.logger.error(f"Error in direct_publish: {str(e)}", exc_info=True)
            raise
    
    def _get_symbol_id(self, symbol: str, exchange: str, mode: int) -> int:
        """Generate unique symbol ID for each subscription (including mode)"""
        # Generate new symbol ID for each unique subscription
        symbol_id = self._symbol_id_counter
        self._symbol_id_counter += 1
        return symbol_id
        
    def _on_open(self, ws):
        """Handle WebSocket open"""
        self.logger.info("Flattrade WebSocket opened")
        
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.logger.info(f"Flattrade WebSocket closed: {close_status_code}")
        
    def _on_error(self, ws, error):
        """Handle WebSocket error"""
        self.logger.error(f"Flattrade WebSocket error: {error}")
        
    def _on_message(self, ws, message):
        """Process incoming market data with ultra-low latency"""
        try:
            import json
            data = json.loads(message)
            msg_type = data.get('t')
            
            if msg_type in ['tf', 'tk', 'df', 'dk']:  # Market data messages
                self._process_market_data_ultra(data)
                
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            
    def _process_market_data_ultra(self, data: Dict[str, Any]) -> None:
        """Ultra-fast market data processing - Fixed to handle multiple subscriptions"""
        token = data.get('tk')
        if not token:
            return
            
        # Get all symbol mappings for this token
        if token not in self.token_to_symbol_mappings:
            self.logger.debug(f"No mappings found for token {token}")
            return
        
        self.logger.debug(f"Processing market data for token {token} with {len(self.token_to_symbol_mappings[token])} mappings")
        
        # Process market data for ALL subscriptions with this token
        for mapping in self.token_to_symbol_mappings[token]:
            symbol_id = mapping['symbol_id']
            symbol = mapping['symbol']
            exchange = mapping['exchange']
            mode = mapping['mode']
            
            self.logger.debug(f"Creating tick for {symbol}.{exchange} (mode:{mode}, symbol_id:{symbol_id})")
            
            # Fast data normalization
            normalized_data = self._normalize_data_ultra(data, mode)
            
            # Create MarketTick with proper data types
            tick = MarketTick(
                symbol_id=int(symbol_id),  # Ensure integer
                timestamp=int(time.time() * 1_000_000_000),  # Ensure integer
                sequence=0,  # Will be set by ring buffer
                message_type=MessageType.MARKET_DATA,  # This is an enum
                ltp=float(normalized_data.get('ltp', 0.0)),  # Ensure float
                open=float(normalized_data.get('open', 0.0)),  # Ensure float
                high=float(normalized_data.get('high', 0.0)),  # Ensure float
                low=float(normalized_data.get('low', 0.0)),  # Ensure float
                close=float(normalized_data.get('close', 0.0)),  # Ensure float
                bid=float(normalized_data.get('bid', 0.0)),  # Ensure float
                ask=float(normalized_data.get('ask', 0.0)),  # Ensure float
                volume=int(normalized_data.get('volume', 0)),  # Ensure integer
                bid_qty=int(normalized_data.get('bid_qty', 0)),  # Ensure integer
                ask_qty=int(normalized_data.get('ask_qty', 0)),  # Ensure integer
                changed_fields=int(normalized_data.get('changed_fields', 0)),  # Ensure integer and within range
                exchange=str(exchange),  # Ensure string
                symbol=str(symbol),  # Ensure string
                mode=int(mode)  # Ensure integer
            )
            
            # Direct publish to ring buffer
            self.direct_publish(tick)
        
    def _normalize_data_ultra(self, data: Dict[str, Any], mode: int) -> Dict[str, Any]:
        """Fast data normalization - Fixed to provide all required fields with safe values"""
        return {
            'mode': mode,
            'ltp': self._safe_float(data.get('lp')),
            'volume': self._safe_int(data.get('v')),
            'open': self._safe_float(data.get('o')),
            'high': self._safe_float(data.get('h')),
            'low': self._safe_float(data.get('l')),
            'close': self._safe_float(data.get('c')),
            'bid': self._safe_float(data.get('bp1')),
            'ask': self._safe_float(data.get('sp1')),
            'bid_qty': self._safe_int(data.get('bq1')),
            'ask_qty': self._safe_int(data.get('sq1')),
            'changed_fields': 0x3FFFFFFF  # Use a safer value that's definitely within 32-bit signed int range
        }
            
    def _safe_float(self, value) -> float:
        """Safe float conversion"""
        if value is None or value == '' or value == '-':
            return 0.0
        try:
            f = float(value)
            # Check for NaN, infinity, or out of range values
            if not (-3.4e38 <= f <= 3.4e38) or f != f:  # f != f checks for NaN
                return 0.0
            return f
        except (ValueError, TypeError, OverflowError):
            return 0.0
            
    def _safe_int(self, value) -> int:
        """Safe int conversion"""
        if value is None or value == '' or value == '-':
            return 0
        try:
            i = int(float(value))
            # Ensure it's within 32-bit unsigned integer range
            return i & 0xFFFFFFFF
        except (ValueError, TypeError, OverflowError):
            return 0

    def _create_success_response(self, message, **kwargs):
        """Create a standard success response"""
        response = {
            'status': 'success',
            'message': message
        }
        response.update(kwargs)
        return response

    def _create_error_response(self, code, message):
        """Create a standard error response"""
        return {
            'status': 'error',
            'code': code,
            'message': message
        }