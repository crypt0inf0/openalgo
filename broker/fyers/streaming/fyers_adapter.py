"""
Updated Fyers SHM WebSocket Adapter - Using brsymbol instead of token
"""
import threading
import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
import os
from concurrent.futures import ThreadPoolExecutor

from database.auth_db import get_auth_token
from websocket_proxy.adapters.base_adapter import BaseBrokerWebSocketAdapter
from websocket_proxy.adapters.mapping import SymbolMapper
from .fyers_mapping import FyersExchangeMapper, FyersSymbolConverter
from .fyers_websocket import FyersWebSocketClient


class Config:
    # Fyers subscription modes
    MODE_LTP = 1
    MODE_QUOTE = 2
    MODE_DEPTH = 3
    
    # Fyers data types (from SDK)
    DATA_TYPE_SYMBOL = 'SymbolUpdate'
    DATA_TYPE_DEPTH = 'DepthUpdate'


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == '' or value == '-':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == '' or value == '-':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


class FyersWebSocketAdapter(BaseBrokerWebSocketAdapter):
    """Fyers adapter with lightweight proxy integration using brsymbol format."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("fyers_adapter")
        self.user_id: Optional[str] = None
        self.broker_name = "fyers"
        self.ws_client: Optional[FyersWebSocketClient] = None
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.hsm_symbol_to_symbol: Dict[str, tuple] = {}
        self.running = False
        self.connected = False
        self.lock = threading.Lock()
        self.reconnect_attempts = 0
        
        # Event loop and thread management
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Market data proxy reference (from base adapter)
        self.market_proxy = None
        
        # Performance monitoring
        self.message_count = 0
        self.last_perf_log = time.time()

    def initialize(self, broker_name: str, user_id: str, auth_data: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Initialize the adapter with broker credentials"""
        self.user_id = user_id
        self.broker_name = broker_name
        
        # Get auth token from database or auth_data
        if auth_data:
            auth_token = auth_data.get('auth_token')
        else:
            auth_token = get_auth_token(user_id)
        
        if not auth_token:
            error_msg = f"Missing Fyers auth token for user {user_id}"
            self.logger.error(error_msg)
            return self._create_error_response("MISSING_CREDENTIALS", error_msg)
        
        # Initialize WebSocket client
        self.ws_client = FyersWebSocketClient(
            access_token=auth_token,
            lite_mode=False,  # Use full mode for comprehensive data
            reconnect=True,
            reconnect_retry=5
        )
        
        # Set adapter loop reference for callback scheduling
        self.ws_client.adapter_loop = None  # Will be set when event loop starts
        
        # Setup callbacks - Direct dictionary access like working version
        self.ws_client.callbacks['on_connect'] = self._on_connect_direct
        self.ws_client.callbacks['on_message'] = self._on_message_direct
        self.ws_client.callbacks['on_error'] = self._on_error_direct
        self.ws_client.callbacks['on_close'] = self._on_close_direct
        
        self.running = True
        
        return self._create_success_response(f"Fyers adapter initialized for user {user_id}")

    def set_proxy(self, proxy):
        """Set the market data proxy reference"""
        self.market_proxy = proxy
        self.logger.info(f"Fyers adapter: market_proxy set to {type(proxy)}, has_publish_method={hasattr(proxy, 'publish_market_data')}")

    def connect(self) -> Dict[str, Any]:
        """Connect to Fyers WebSocket - synchronous interface"""
        if not self.ws_client:
            error_msg = "WebSocket client not initialized. Call initialize() first."
            self.logger.error(error_msg)
            return self._create_error_response("NOT_INITIALIZED", error_msg)
        
        self.logger.info("Connecting to Fyers WebSocket...")
        
        try:
            # Start event loop in separate thread
            self._start_event_loop()
            
            # Set the adapter loop reference in WebSocket client
            if self.ws_client:
                self.ws_client.adapter_loop = self.loop
            
            # Connect to WebSocket in event loop
            future = asyncio.run_coroutine_threadsafe(self.ws_client.connect(), self.loop)
            
            success = future.result(timeout=15.0)  # 15 second timeout
            if success:
                self.connected = True
                self.reconnect_attempts = 0
                self.logger.info("Connected to Fyers WebSocket successfully")
                return self._create_success_response("Connected to Fyers WebSocket")
            else:
                error_msg = "Failed to connect to Fyers WebSocket"
                self.logger.error(error_msg)
                return self._create_error_response("CONNECTION_FAILED", error_msg)
        except Exception as e:
            error_msg = f"Connection error: {e}"
            self.logger.error(error_msg)
            return self._create_error_response("CONNECTION_ERROR", error_msg)

    def disconnect(self) -> None:
        """Disconnect from Fyers WebSocket - synchronous interface"""
        self.running = False
        
        if self.ws_client and self.loop:
            future = asyncio.run_coroutine_threadsafe(self.ws_client.disconnect(), self.loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
        
        self._stop_event_loop()
        self.connected = False
        self.logger.info("Disconnected from Fyers WebSocket (SHM mode)")

    def _start_event_loop(self) -> None:
        """Start asyncio event loop in separate thread"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_forever()
            finally:
                self.loop.close()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # Wait for loop to start
        timeout = 5.0
        start_time = time.time()
        while self.loop is None and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if self.loop is None:
            raise RuntimeError("Failed to start event loop")

    def _stop_event_loop(self) -> None:
        """Stop the event loop thread"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5.0)

    def subscribe(self, symbol: str, exchange: str, mode: int = Config.MODE_QUOTE, depth_level: int = 5) -> Dict[str, Any]:
        """Subscribe to market data for a symbol using brsymbol"""
        if not (symbol and exchange and mode in [Config.MODE_LTP, Config.MODE_QUOTE, Config.MODE_DEPTH]):
            return {"status": "error", "code": "INVALID_PARAMS", "message": "Invalid subscription parameters"}
        
        # Get token info from symbol mapper to extract brsymbol and brexchange
        token_info = SymbolMapper.get_token_from_symbol(symbol, exchange)
        if not token_info:
            return {"status": "error", "code": "SYMBOL_NOT_FOUND", "message": f"Symbol {symbol} not found"}
        
        # Extract brsymbol and brexchange from token_info
        brsymbol = token_info.get('brsymbol')
        brexchange = token_info.get('brexchange')
        
        if not brsymbol or not brexchange:
            return {"status": "error", "code": "INVALID_TOKEN_INFO", "message": f"Missing brsymbol or brexchange for {symbol}"}
        
        # Convert to Fyers exchange format
        fyers_exchange = FyersExchangeMapper.to_fyers_exchange(brexchange)
        if not fyers_exchange:
            return {"status": "error", "code": "UNSUPPORTED_EXCHANGE", "message": f"Exchange {exchange} not supported"}
        
        # Determine data type based on mode
        data_type = Config.DATA_TYPE_DEPTH if mode == Config.MODE_DEPTH else Config.DATA_TYPE_SYMBOL
        
        # Create HSM symbol using database token (more efficient than API calls)
        database_token = token_info.get('token')
        if not database_token:
            return {"status": "error", "code": "NO_DATABASE_TOKEN", "message": f"No database token available for {symbol}"}
        
        hsm_symbol = FyersSymbolConverter.create_hsm_symbol_with_database_token(brsymbol, database_token, data_type)
        if not hsm_symbol:
            return {"status": "error", "code": "INVALID_SYMBOL_FORMAT", "message": f"Cannot create HSM symbol for {symbol}"}
        
        subscription = {
            'symbol': symbol,
            'exchange': exchange,
            'mode': mode,
            'depth_level': depth_level,
            'brsymbol': brsymbol,
            'brexchange': brexchange,
            'fyers_exchange': fyers_exchange,
            'hsm_symbol': hsm_symbol,
            'data_type': data_type
        }
        
        correlation_id = f"{symbol}_{exchange}_{mode}"
        
        with self.lock:
            self.subscriptions[correlation_id] = subscription
            self.hsm_symbol_to_symbol[hsm_symbol] = (symbol, exchange)
        
        if self.connected and self.loop:
            asyncio.run_coroutine_threadsafe(
                self._websocket_subscribe(subscription), 
                self.loop
            )
        
        self.logger.info(f"Subscribing to {symbol}.{exchange} with HSM symbol: {hsm_symbol}")
        
        return {
            "status": "success", 
            "message": f"Subscribed to {symbol}.{exchange}", 
            "symbol": symbol, 
            "exchange": exchange, 
            "mode": mode,
            "hsm_symbol": hsm_symbol
        }

    def unsubscribe(self, symbol: str, exchange: str, mode: int = Config.MODE_QUOTE) -> Dict[str, Any]:
        """Unsubscribe from symbol with enhanced error handling and state validation."""
        if not symbol or not exchange:
            self.logger.warning(f"Invalid unsubscribe parameters: symbol='{symbol}', exchange='{exchange}'")
            return {"status": "error", "code": "INVALID_PARAMS", "message": "Symbol and exchange are required"}
        
        if mode not in [Config.MODE_LTP, Config.MODE_QUOTE, Config.MODE_DEPTH]:
            self.logger.warning(f"Invalid mode {mode} for unsubscribe {symbol}.{exchange}")
            return {"status": "error", "code": "INVALID_MODE", "message": f"Invalid mode: {mode}"}
        
        correlation_id = f"{symbol}_{exchange}_{mode}"
        
        with self.lock:
            if correlation_id not in self.subscriptions:
                self.logger.warning(f"Unsubscribe requested for {symbol}.{exchange} mode={mode} but not found in subscriptions.")
                return {
                    "status": "success",
                    "message": f"Already unsubscribed from {symbol}.{exchange}",
                    "symbol": symbol,
                    "exchange": exchange,
                    "mode": mode
                }
            
            subscription = self.subscriptions[correlation_id]
            hsm_symbol = subscription.get('hsm_symbol')
            
            # Remove this specific subscription
            del self.subscriptions[correlation_id]
            
            # Check if there are still other subscriptions for the same HSM symbol
            remaining_subscriptions = [
                sub for sub in self.subscriptions.values() 
                if sub.get('hsm_symbol') == hsm_symbol
            ]
            
            try:
                # Only unsubscribe from WebSocket if no more subscriptions for this HSM symbol
                if not remaining_subscriptions:
                    self.logger.debug(f"No more subscriptions for {hsm_symbol}, unsubscribing from WebSocket")
                    if self.connected and self.loop:
                        asyncio.run_coroutine_threadsafe(
                            self._websocket_unsubscribe(subscription), 
                            self.loop
                        )
                    
                    # Remove HSM symbol mapping only if no more subscriptions
                    self.hsm_symbol_to_symbol.pop(hsm_symbol, None)
                    self.logger.debug(f"Cleaned up HSM symbol mapping for {hsm_symbol}")
                else:
                    self.logger.debug(f"Keeping HSM symbol {hsm_symbol} - {len(remaining_subscriptions)} subscriptions remain")
                
            except Exception as e:
                self.logger.warning(f"Error during websocket unsubscribe for {symbol}.{exchange}: {e}")
        
        self.logger.info(f"Successfully unsubscribed from {symbol}.{exchange} mode={mode}")
        return {
            "status": "success",
            "message": f"Unsubscribed from {symbol}.{exchange}",
            "symbol": symbol,
            "exchange": exchange,
            "mode": mode
        }

    # Direct async callbacks - no thread pool queuing
    async def _on_connect_direct(self) -> None:
        """Direct async connect handler - no queuing"""
        self.logger.info("Connected to Fyers WebSocket")
        self.connected = True
        await self._resubscribe_all()

    async def _on_error_direct(self, error_message: str) -> None:
        """Direct async error handler"""
        self.logger.error(f"Fyers WebSocket error: {error_message}")

    async def _on_close_direct(self) -> None:
        """Direct async close handler"""
        self.logger.info("Fyers WebSocket closed")
        self.connected = False
        if self.running:
            await self._schedule_reconnection()

    async def _on_message_direct(self, data: Dict[str, Any]) -> None:
        """Direct async message processing - optimized for high throughput"""
        # Performance monitoring - reduced frequency
        self.message_count += 1
        current_time = time.time()
        if current_time - self.last_perf_log > 30:  # Reduced from 10 to 30 seconds
            msg_per_sec = self.message_count / (current_time - self.last_perf_log)
            self.logger.info(f"Processing {msg_per_sec:.1f} messages/sec")
            self.message_count = 0
            self.last_perf_log = current_time

        try:
            # Process market data message directly without debug logging
            await self._process_market_message(data)
                
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")

    async def _process_market_message(self, data: Dict[str, Any]) -> None:
        """Process market data message from Fyers WebSocket - optimized for speed"""
        try:
            # Extract symbol information from the response
            symbol_key = data.get('symbol', '')  # This would be the HSM symbol
            
            if not symbol_key or symbol_key not in self.hsm_symbol_to_symbol:
                return  # Skip unknown symbols silently for performance
            
            symbol, exchange = self.hsm_symbol_to_symbol[symbol_key]
            matching_subscriptions = self._find_matching_subscriptions(symbol_key)
            
            # Process all matching subscriptions in parallel for better performance
            if matching_subscriptions:
                tasks = [
                    self._process_subscription_message(data, subscription, symbol, exchange)
                    for subscription in matching_subscriptions
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            self.logger.error(f"Market message processing error: {e}")

    async def _process_subscription_message(self, data: Dict[str, Any], subscription: Dict[str, Any], 
                                          symbol: str, exchange: str) -> None:
        """Process single subscription message - optimized for speed"""
        mode = subscription['mode']
        
        # Normalize market data based on Fyers response format
        normalized = self._normalize_fyers_data(data, mode)
        normalized.update({
            'symbol': symbol,
            'exchange': exchange,
            'timestamp': int(time.time() * 1000)
        })
        
        # Publish using the base adapter method for consistency
        self.publish_market_data(symbol, exchange, normalized)

    def _normalize_fyers_data(self, data: Dict[str, Any], mode: int) -> Dict[str, Any]:
        """Normalize Fyers market data to standard format"""
        msg_type = data.get('type', '')
        
        # Base data structure
        normalized = {
            'mode': mode,
            'ltp': safe_float(data.get('ltp')),
            'volume': safe_int(data.get('vol_traded_today')),
            'last_trade_time': data.get('last_traded_time'),
            'last_trade_quantity': safe_int(data.get('last_traded_qty')),
        }
        
        if mode == Config.MODE_LTP:
            return normalized
        
        # Add OHLC and additional data for QUOTE mode
        if mode in [Config.MODE_QUOTE, Config.MODE_DEPTH]:
            normalized.update({
                'open': safe_float(data.get('open_price')),
                'high': safe_float(data.get('high_price')),
                'low': safe_float(data.get('low_price')),
                'close': safe_float(data.get('prev_close_price')),
                'bid_price': safe_float(data.get('bid_price')),
                'ask_price': safe_float(data.get('ask_price')),
                'bid_size': safe_int(data.get('bid_size')),
                'ask_size': safe_int(data.get('ask_size')),
                'average_price': safe_float(data.get('avg_trade_price')),
                'total_buy_quantity': safe_int(data.get('tot_buy_qty')),
                'total_sell_quantity': safe_int(data.get('tot_sell_qty')),
                'open_interest': safe_int(data.get('OI')),
            })
            
            # Calculate change and percentage change
            ltp = normalized['ltp']
            prev_close = normalized['close']
            if ltp and prev_close and prev_close != 0:
                change = ltp - prev_close
                change_percent = (change / prev_close) * 100
                normalized['change'] = round(change, 4)
                normalized['change_percent'] = round(change_percent, 4)
        
        # Add depth data for DEPTH mode
        if mode == Config.MODE_DEPTH and msg_type == 'dp':
            depth_data = {
                'buy': [
                    {'price': safe_float(data.get('bid_price1')), 'quantity': safe_int(data.get('bid_size1')), 'orders': safe_int(data.get('bid_order1'))},
                    {'price': safe_float(data.get('bid_price2')), 'quantity': safe_int(data.get('bid_size2')), 'orders': safe_int(data.get('bid_order2'))},
                    {'price': safe_float(data.get('bid_price3')), 'quantity': safe_int(data.get('bid_size3')), 'orders': safe_int(data.get('bid_order3'))},
                    {'price': safe_float(data.get('bid_price4')), 'quantity': safe_int(data.get('bid_size4')), 'orders': safe_int(data.get('bid_order4'))},
                    {'price': safe_float(data.get('bid_price5')), 'quantity': safe_int(data.get('bid_size5')), 'orders': safe_int(data.get('bid_order5'))},
                ],
                'sell': [
                    {'price': safe_float(data.get('ask_price1')), 'quantity': safe_int(data.get('ask_size1')), 'orders': safe_int(data.get('ask_order1'))},
                    {'price': safe_float(data.get('ask_price2')), 'quantity': safe_int(data.get('ask_size2')), 'orders': safe_int(data.get('ask_order2'))},
                    {'price': safe_float(data.get('ask_price3')), 'quantity': safe_int(data.get('ask_size3')), 'orders': safe_int(data.get('ask_order3'))},
                    {'price': safe_float(data.get('ask_price4')), 'quantity': safe_int(data.get('ask_size4')), 'orders': safe_int(data.get('ask_order4'))},
                    {'price': safe_float(data.get('ask_price5')), 'quantity': safe_int(data.get('ask_size5')), 'orders': safe_int(data.get('ask_order5'))},
                ],
            }
            
            normalized['depth'] = depth_data
            normalized['depth_level'] = 5
        
        return normalized

    def publish_market_data(self, symbol: str, exchange: str, data: Dict[str, Any]) -> None:
        """
        Publish market data using the base adapter method
        Overrides the base method to add Fyers-specific enhancements
        """
        try:
            # Add broker timestamp for latency measurement
            broker_timestamp_ms = time.time() * 1000
            data['broker_timestamp'] = broker_timestamp_ms
            data['broker'] = self.broker_name
            
            # Call the base adapter's publish method
            super().publish_market_data(symbol, exchange, data)
            
            # Log for debugging (reduced frequency for performance)
            if self.message_count % 100 == 0:  # Log every 100th message
                mode = data.get('mode', 1)
                mode_str = {1: 'LTP', 2: 'QUOTE', 3: 'DEPTH'}[mode]
                self.logger.debug(f"Published {symbol} data: LTP={data.get('ltp', 'N/A')}, Mode={mode_str}, Broker_TS={broker_timestamp_ms}")
                    
        except Exception as e:
            self.logger.error(f"Error publishing market data for {symbol}.{exchange}: {e}")

    async def _websocket_subscribe(self, subscription: Dict) -> None:
        """Subscribe via WebSocket using brsymbol"""
        hsm_symbol = subscription['hsm_symbol']
        data_type = subscription['data_type']
        
        self.logger.info(f"Subscribing to WebSocket with HSM symbol: {hsm_symbol}, data_type: {data_type}")
        
        if self.ws_client and self.connected:
            success = await self.ws_client.subscribe([hsm_symbol], data_type)
            if success:
                self.logger.info(f"Successfully subscribed to {hsm_symbol}")
            else:
                self.logger.error(f"Failed to subscribe to {hsm_symbol}")

    async def _websocket_unsubscribe(self, subscription: Dict) -> bool:
        """Unsubscribe from websocket"""
        hsm_symbol = subscription.get('hsm_symbol')
        
        if not hsm_symbol:
            self.logger.warning(f"Invalid subscription data for websocket unsubscribe: {subscription}")
            return False
        
        self.logger.debug(f"[BROKER<-APP] unsubscribe: hsm_symbol={hsm_symbol}")
        
        try:
            if self.ws_client and self.connected:
                success = await self.ws_client.unsubscribe([hsm_symbol])
                if not success:
                    self.logger.warning(f"Failed to send unsubscribe for {hsm_symbol}")
                    return False
                return True
            else:
                self.logger.warning(f"WebSocket client not available for unsubscribe: {hsm_symbol}")
                return False
        except Exception as e:
            self.logger.error(f"Error in websocket unsubscribe for {hsm_symbol}: {e}")
            return False

    async def _resubscribe_all(self):
        """Resubscribe to all symbols after connection"""
        with self.lock:
            # Group subscriptions by data type
            symbol_subscriptions = {'SymbolUpdate': [], 'DepthUpdate': []}
            
            for sub in self.subscriptions.values():
                hsm_symbol = sub['hsm_symbol']
                data_type = sub['data_type']
                if hsm_symbol not in symbol_subscriptions[data_type]:
                    symbol_subscriptions[data_type].append(hsm_symbol)
            
            # Subscribe for each data type
            for data_type, symbols in symbol_subscriptions.items():
                if symbols:
                    self.logger.info(f"Resubscribing to {len(symbols)} symbols for {data_type}: {symbols}")
                    success = await self.ws_client.subscribe(symbols, data_type)
                    if success:
                        self.logger.info(f"Resubscribed to {len(symbols)} symbols for {data_type}")
                    else:
                        self.logger.error(f"Failed to resubscribe to {data_type} symbols")

    async def _schedule_reconnection(self):
        """Schedule reconnection with exponential backoff"""
        delay = min(5 * (2 ** self.reconnect_attempts), 60)
        self.reconnect_attempts += 1
        
        self.logger.info(f"Scheduling reconnection in {delay} seconds (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
        await self._attempt_reconnection()

    async def _attempt_reconnection(self):
        """Attempt to reconnect to WebSocket"""
        try:
            # Reinitialize WebSocket client
            auth_token = get_auth_token(self.user_id)
            if auth_token:
                self.ws_client = FyersWebSocketClient(
                    access_token=auth_token,
                    lite_mode=False,
                    reconnect=True,
                    reconnect_retry=5
                )
                
                # Setup callbacks
                self.ws_client.callbacks['on_connect'] = self._on_connect_direct
                self.ws_client.callbacks['on_message'] = self._on_message_direct
                self.ws_client.callbacks['on_error'] = self._on_error_direct
                self.ws_client.callbacks['on_close'] = self._on_close_direct
                
                success = await self.ws_client.connect()
                if success:
                    self.connected = True
                    self.reconnect_attempts = 0
                    self.logger.info("Reconnected successfully")
                else:
                    self.logger.error("Reconnection failed")
            else:
                self.logger.error("No auth token available for reconnection")
                
        except Exception as e:
            self.logger.error(f"Reconnection error: {e}")

    def _find_matching_subscriptions(self, hsm_symbol: str) -> List[Dict]:
        """Find subscriptions matching the HSM symbol"""
        return [s for s in self.subscriptions.values() if s['hsm_symbol'] == hsm_symbol]

    def get_subscriptions(self) -> Dict[str, Any]:
        """Get current subscriptions status"""
        with self.lock:
            return {
                "count": len(self.subscriptions),
                "subscriptions": list(self.subscriptions.keys()),
                "connected": self.connected,
                "running": self.running
            }

    def is_connected(self) -> bool:
        """Check if adapter is connected"""
        return self.connected

    def is_running(self) -> bool:
        """Check if adapter is running"""
        return self.running
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            'broker': self.broker_name,
            'connected': self.connected,
            'running': self.running,
            'subscriptions': len(self.subscriptions),
            'reconnect_attempts': self.reconnect_attempts,
            'message_count': self.message_count,
            'event_loop_running': self.loop is not None and not self.loop.is_closed() if self.loop else False,
            'ws_client_connected': self.ws_client.is_connected() if self.ws_client else False
        }

    def cleanup(self) -> None:
        """Clean up adapter resources"""
        try:
            self.disconnect()
            self.logger.info("Fyers adapter cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during Fyers adapter cleanup: {e}")

    def _create_success_response(self, message: str, **kwargs) -> Dict[str, Any]:
        """Create standardized success response"""
        response = {"status": "success", "message": message}
        response.update(kwargs)
        return response

    def _create_error_response(self, code: str, message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {"status": "error", "code": code, "message": message}