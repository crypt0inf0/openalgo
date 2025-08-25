"""
Lock-free Flattrade Ultra Adapter with atomic value retention
Zero locks, zero cache, minimal latency
FIXED: Race condition handling for initialization
"""

import time
import threading
import ctypes
import os
import logging
from typing import Dict, Any, Optional
from websocket_proxy.ultra_low_latency_adapter import UltraLowLatencyAdapter
from websocket_proxy.cross_platform_structures import MarketTick, MessageType
from .flattrade_websocket import FlattradeWebSocket
from .flattrade_mapping import FlattradeExchangeMapper
from websocket_proxy.mapping import SymbolMapper
from database.auth_db import get_auth_token

class AtomicFloat:
    """Lock-free atomic float using ctypes with initialization state"""
    def __init__(self, initial_value: float = 0.0):
        self._value = ctypes.c_double(initial_value)
        self._initialized = ctypes.c_int(0)  # 0 = uninitialized, 1 = initialized
    
    def load(self) -> float:
        return self._value.value
    
    def store(self, value: float):
        self._value.value = value
        self._initialized.value = 1  # Mark as initialized
    
    def is_initialized(self) -> bool:
        return self._initialized.value == 1
    
    def load_or_default(self, default: float = 0.0) -> float:
        """Load value if initialized, otherwise return default"""
        if self._initialized.value == 1:
            return self._value.value
        return default
    
    def store_if_valid(self, value: float) -> float:
        """Store if value is valid (non-zero), otherwise return current or zero"""
        if value != 0.0:
            # Valid new value - store it
            self._value.value = value
            self._initialized.value = 1
            return value
        else:
            # Invalid/empty value - return current if initialized, else 0
            if self._initialized.value == 1:
                return self._value.value
            return 0.0

class AtomicInt:
    """Lock-free atomic integer with initialization state"""
    def __init__(self, initial_value: int = 0):
        self._value = ctypes.c_int64(initial_value)
        self._initialized = ctypes.c_int(0)
    
    def load(self) -> int:
        return self._value.value
    
    def store(self, value: int):
        self._value.value = value
        self._initialized.value = 1
    
    def is_initialized(self) -> bool:
        return self._initialized.value == 1
    
    def store_if_valid(self, value: int) -> int:
        """Store if value is valid, otherwise return current or zero"""
        # For volumes/quantities, we always store (zero can be legitimate)
        self._value.value = value
        self._initialized.value = 1
        return value

class SymbolState:
    """Lock-free symbol state storage with proper initialization"""
    def __init__(self):
        # Price fields with retention logic
        self.ltp = AtomicFloat(0.0)
        self.open = AtomicFloat(0.0)
        self.high = AtomicFloat(0.0)
        self.low = AtomicFloat(0.0)
        self.close = AtomicFloat(0.0)
        self.bid = AtomicFloat(0.0)
        self.ask = AtomicFloat(0.0)
        
        # Volume/quantity fields (always updated)
        self.volume = AtomicInt(0)
        self.bid_qty = AtomicInt(0)
        self.ask_qty = AtomicInt(0)
        
        # Global initialization flag
        self.first_update_done = ctypes.c_int(0)
    
    def mark_first_update(self):
        """Mark that first update has been processed"""
        self.first_update_done.value = 1
    
    def is_first_update_done(self) -> bool:
        return self.first_update_done.value == 1

class FlattradeUltraAdapterLockFree(UltraLowLatencyAdapter):
    """Ultra-low latency lock-free Flattrade adapter"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("flattrade_ultra_adapter_lockfree")
        self.ws_client = None
        self.subscriptions = {}
        self.token_to_symbol_mappings = {}
        self._symbol_id_counter = 1000
        self.ws_subscription_refs = {}
        
        # Lock-free symbol states - pre-allocated for performance
        self.symbol_states: Dict[int, SymbolState] = {}
        
    def initialize(self, broker_name: str, user_id: str, auth_data: Optional[Dict[str, str]] = None) -> None:
        """Initialize adapter"""
        super().initialize(broker_name, user_id, auth_data)
        
        # Get credentials
        api_key = os.getenv('BROKER_API_KEY', '')
        if ':::' in api_key:
            self.actid = api_key.split(':::')[0]
        else:
            self.actid = user_id
            
        self.susertoken = get_auth_token(user_id)
        
        if not self.actid or not self.susertoken:
            raise ValueError(f"Missing Flattrade credentials for user {user_id}")
        
        self.logger.info(f"Initialized lock-free Flattrade ultra adapter for user {user_id}")
    
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
        """Subscribe with pre-allocated state"""
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
            
            if correlation_id in self.subscriptions:
                return self._create_error_response("ALREADY_SUBSCRIBED", 
                    f"Already subscribed to {symbol}.{exchange} mode {mode}")
            
            symbol_id = self._get_symbol_id(symbol, exchange, mode)
            
            # Pre-allocate symbol state (lock-free)
            self.symbol_states[symbol_id] = SymbolState()
            
            # Store token mapping
            if token not in self.token_to_symbol_mappings:
                self.token_to_symbol_mappings[token] = []
            
            self.token_to_symbol_mappings[token].append({
                'symbol_id': symbol_id,
                'symbol': symbol,
                'exchange': exchange,
                'mode': mode,
                'correlation_id': correlation_id
            })
            
            subscription = {
                'symbol': symbol,
                'exchange': exchange,
                'mode': mode,
                'token': token,
                'scrip': scrip,
                'symbol_id': symbol_id
            }
            
            self.subscriptions[correlation_id] = subscription
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
        symbol_id = subscription['symbol_id']
        
        # Remove subscription
        del self.subscriptions[correlation_id]
        
        # Clean up symbol state
        if symbol_id in self.symbol_states:
            del self.symbol_states[symbol_id]
        
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
            if UltraLowLatencyAdapter._shared_ring_buffer is not None:
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
        """Generate unique symbol ID for each subscription"""
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
        """Ultra-fast lock-free market data processing with race condition fix"""
        token = data.get('tk')
        if not token or token not in self.token_to_symbol_mappings:
            return
        
        # Process all mappings for this token
        for mapping in self.token_to_symbol_mappings[token]:
            symbol_id = mapping['symbol_id']
            symbol = mapping['symbol']
            exchange = mapping['exchange']
            mode = mapping['mode']
            
            # Get symbol state (pre-allocated, no locks)
            state = self.symbol_states.get(symbol_id)
            if not state:
                continue
            
            # Extract and validate raw values
            raw_ltp = self._safe_float(data.get('lp'))
            raw_volume = self._safe_int(data.get('v'))
            raw_open = self._safe_float(data.get('o'))
            raw_high = self._safe_float(data.get('h'))
            raw_low = self._safe_float(data.get('l'))
            raw_close = self._safe_float(data.get('c'))
            raw_bid = self._safe_float(data.get('bp1'))
            raw_ask = self._safe_float(data.get('sp1'))
            raw_bid_qty = self._safe_int(data.get('bq1'))
            raw_ask_qty = self._safe_int(data.get('sq1'))
            
            # FIXED: Handle first update case properly
            is_first_update = not state.is_first_update_done()
            
            # Lock-free atomic updates with retention logic
            if is_first_update:
                # First update - store all valid values, use zero for invalid ones
                final_ltp = raw_ltp if raw_ltp != 0.0 else 0.0
                final_open = raw_open if raw_open != 0.0 else 0.0
                final_high = raw_high if raw_high != 0.0 else 0.0
                final_low = raw_low if raw_low != 0.0 else 0.0
                final_close = raw_close if raw_close != 0.0 else 0.0
                final_bid = raw_bid if raw_bid != 0.0 else 0.0
                final_ask = raw_ask if raw_ask != 0.0 else 0.0
                
                # Store all values atomically
                state.ltp.store(final_ltp)
                state.open.store(final_open)
                state.high.store(final_high)
                state.low.store(final_low)
                state.close.store(final_close)
                state.bid.store(final_bid)
                state.ask.store(final_ask)
                
                # Mark first update done
                state.mark_first_update()
            else:
                # Subsequent updates - use retention logic
                final_ltp = state.ltp.store_if_valid(raw_ltp)
                final_open = state.open.store_if_valid(raw_open)
                final_high = state.high.store_if_valid(raw_high)
                final_low = state.low.store_if_valid(raw_low)
                final_close = state.close.store_if_valid(raw_close)
                final_bid = state.bid.store_if_valid(raw_bid)
                final_ask = state.ask.store_if_valid(raw_ask)
            
            # Volume and quantities always update (can be zero legitimately)
            final_volume = state.volume.store_if_valid(raw_volume)
            final_bid_qty = state.bid_qty.store_if_valid(raw_bid_qty)
            final_ask_qty = state.ask_qty.store_if_valid(raw_ask_qty)
            
            # Create tick with current values
            tick = MarketTick(
                symbol_id=symbol_id,
                timestamp=int(time.time() * 1_000_000_000),
                sequence=0,
                message_type=MessageType.MARKET_DATA,
                ltp=final_ltp,
                open=final_open,
                high=final_high,
                low=final_low,
                close=final_close,
                bid=final_bid,
                ask=final_ask,
                volume=final_volume,
                bid_qty=final_bid_qty,
                ask_qty=final_ask_qty,
                changed_fields=0x3FFFFFFF,
                exchange=exchange,
                symbol=symbol,
                mode=mode
            )
            
            # Direct publish (no additional overhead)
            self.direct_publish(tick)
            
            # Debug log for first few updates
            if is_first_update:
                self.logger.debug(f"First update for {symbol}.{exchange}: "
                               f"LTP={final_ltp}, Volume={final_volume}")
    
    def _safe_float(self, value) -> float:
        """Safe float conversion - returns 0.0 for invalid values"""
        if value is None or value == '' or value == '-' or value == 0:
            return 0.0
        try:
            f = float(value)
            if f != f or not (-1e30 <= f <= 1e30):  # Check for NaN or extreme values
                return 0.0
            return f
        except (ValueError, TypeError, OverflowError):
            return 0.0
    
    def _safe_int(self, value) -> int:
        """Safe int conversion - returns 0 for invalid values"""
        if value is None or value == '' or value == '-':
            return 0
        try:
            return int(float(value)) & 0xFFFFFFFF  # Ensure 32-bit range
        except (ValueError, TypeError, OverflowError):
            return 0
    
    def _create_success_response(self, message, **kwargs):
        response = {'status': 'success', 'message': message}
        response.update(kwargs)
        return response
    
    def _create_error_response(self, code, message):
        return {'status': 'error', 'code': code, 'message': message}