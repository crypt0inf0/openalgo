"""
Enhanced Fyers WebSocket Client Implementation
Updated with proper message formats from enhanced version and comprehensive debug logging
"""
import asyncio
import json
import logging
import struct
import threading
import time
import websocket
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from .fyers_mapping import FyersSymbolConverter, FyersExchangeMapper
from utils.logging import get_logger


class FyersWebSocketClient:
    """
    Enhanced Fyers WebSocket client with proper message formatting and debug logging
    """
    
    # WebSocket URL
    WS_URL = "wss://socket.fyers.in/hsm/v1-5/prod"
    
    # Connection constants
    CONNECTION_TIMEOUT = 15
    RECONNECT_MAX_ATTEMPTS = 5
    RECONNECT_DELAY_BASE = 2
    RECONNECT_DELAY_MAX = 30
    
    # Message types (from SDK)
    MSG_AUTH_RESPONSE = 1
    MSG_SUBSCRIBE_RESPONSE = 4
    MSG_UNSUBSCRIBE_RESPONSE = 5
    MSG_DATA_FEED = 6
    MSG_CHANNEL_RESUME = 8
    MSG_CHANNEL_PAUSE = 7
    MSG_MODE_RESPONSE = 12
    
    # Data feed types (from SDK)
    DATA_SNAPSHOT = 83  # 'S'
    DATA_UPDATE = 85    # 'U' 
    DATA_LITE = 76      # 'L'
    
    def __init__(self, access_token: str, lite_mode: bool = False, 
                 reconnect: bool = True, reconnect_retry: int = 5):
        """
        Initialize Fyers WebSocket client
        
        Args:
            access_token: Fyers access token
            lite_mode: Whether to use lite mode (LTP only)
            reconnect: Whether to auto-reconnect on disconnection
            reconnect_retry: Maximum reconnection attempts
        """
        self.logger = get_logger(__name__)
        self.access_token = access_token
        self.lite_mode = lite_mode
        self.reconnect_enabled = reconnect
        self.max_reconnect_attempts = min(reconnect_retry, 50)
        
        # Extract HSM token from access token
        self.hsm_token = self._extract_hsm_token()
        if not self.hsm_token:
            raise ValueError("Invalid access token - cannot extract HSM token")
        
        # Connection state
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.connected = False
        self.running = False
        self.authenticated = False
        
        # Reconnection state
        self.reconnect_attempts = 0
        self.reconnect_delay = 0
        
        # Message handling
        self.message_lock = threading.Lock()
        self.message_condition = threading.Condition(lock=self.message_lock)
        self.message_queue = []
        self.message_thread: Optional[threading.Thread] = None
        self.message_thread_stop_event = threading.Event()
        
        # Ping thread for keepalive
        self.ping_thread: Optional[threading.Thread] = None
        
        # Subscription tracking
        self.subscriptions: Set[str] = set()
        self.symbol_token_map: Dict[str, str] = {}
        
        # Data processing maps (from SDK)
        self.scrips_sym: Dict[int, str] = {}
        self.index_sym: Dict[int, str] = {}
        self.dp_sym: Dict[int, str] = {}
        self.response_data: Dict[str, Dict] = {}
        
        # Acknowledgment tracking
        self.ack_count = 0
        self.update_count = 0
        
        # Enhanced configuration from enhanced version
        self.source = "PythonSDK-3.0.9"
        self.channel_num = 11
        
        # Callbacks
        self.callbacks: Dict[str, Optional[Callable]] = {
            "on_connect": None,
            "on_message": None,
            "on_error": None,
            "on_close": None
        }
        
        # Constants (from SDK map.json)
        self.data_val = [
            "ltp", "vol_traded_today", "last_traded_time", "exch_feed_time",
            "bid_size", "ask_size", "bid_price", "ask_price", "last_traded_qty",
            "tot_buy_qty", "tot_sell_qty", "avg_trade_price", "OI", "low_price",
            "high_price", "Yhigh", "Ylow", "lower_ckt", "upper_ckt", "open_price",
            "prev_close_price", "type", "symbol"
        ]
        
        self.index_val = [
            "ltp", "prev_close_price", "exch_feed_time", "high_price", 
            "low_price", "open_price", "type", "symbol"
        ]
        
        self.lite_val = ["ltp", "symbol", "type"]
        
        self.depth_val = [
            "bid_price1", "bid_price2", "bid_price3", "bid_price4", "bid_price5",
            "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5",
            "bid_size1", "bid_size2", "bid_size3", "bid_size4", "bid_size5",
            "ask_size1", "ask_size2", "ask_size3", "ask_size4", "ask_size5",
            "bid_order1", "bid_order2", "bid_order3", "bid_order4", "bid_order5",
            "ask_order1", "ask_order2", "ask_order3", "ask_order4", "ask_order5",
            "type", "symbol"
        ]
        
        # Logging
        self.logger = logging.getLogger("fyers_websocket")
        # Set debug level for detailed message logging
        self.logger.setLevel(logging.DEBUG)

    def _extract_hsm_token(self) -> Optional[str]:
        """Extract HSM token from access token (from SDK logic)"""
        try:
            import base64
            if ":" in self.access_token:
                self.access_token = self.access_token.split(":")[1]
            
            header_token, payload_b64, _ = self.access_token.split(".")
            decoded_payload = base64.urlsafe_b64decode(payload_b64 + "===")
            decode_token = json.loads(decoded_payload.decode())
            
            # Check token expiry
            current_time = int(time.time())
            if decode_token["exp"] - current_time < 0:
                self.logger.error("Access token has expired")
                return None
            
            hsm_key = decode_token["hsm_key"]
            self.logger.info(f"Successfully extracted HSM token (length: {len(hsm_key)})")
            return hsm_key
            
        except Exception as e:
            self.logger.error(f"Failed to extract HSM token: {e}")
            return None

    async def connect(self) -> bool:
        """Establish WebSocket connection with authentication"""
        for attempt in range(1, self.max_reconnect_attempts + 1):
            try:
                self.running = True
                self.message_thread_stop_event.clear()
                
                self._create_websocket()
                
                if await self._wait_for_connection():
                    self.logger.info("Connected to Fyers WebSocket successfully")
                    await self._trigger_callback("on_connect")
                    return True
                else:
                    self.logger.warning(f"Connection attempt {attempt} failed")
                    
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt} error: {e}")
                
            if attempt < self.max_reconnect_attempts:
                delay = self._calculate_backoff_delay(attempt)
                self.logger.info(f"Reconnecting in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                await self._trigger_error("Max reconnection attempts reached")
                return False
        
        return False

    def _create_websocket(self) -> None:
        """Create WebSocket connection"""
        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()

    def _run_websocket(self) -> None:
        """Run WebSocket connection"""
        try:
            self.ws.run_forever(ping_interval=None, ping_timeout=None)
        except Exception as e:
            self.logger.error(f"WebSocket run error: {e}")

    async def _wait_for_connection(self) -> bool:
        """Wait for WebSocket connection to be established"""
        start_time = time.time()
        while time.time() - start_time < self.CONNECTION_TIMEOUT:
            if self.authenticated:
                return True
            await asyncio.sleep(0.1)
        return False

    def _calculate_backoff_delay(self, attempt: int) -> int:
        """Calculate exponential backoff delay"""
        delay = self.RECONNECT_DELAY_BASE * (2 ** (attempt - 1))
        return min(delay, self.RECONNECT_DELAY_MAX)

    async def disconnect(self) -> None:
        """Disconnect from WebSocket and cleanup resources"""
        self.running = False
        self.connected = False
        
        # Stop threads
        self._stop_ping_thread()
        self._stop_message_thread()
        
        # Close WebSocket
        if self.ws:
            self.ws.close()
            
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5.0)
            
        await self._trigger_callback("on_close")

    def _stop_ping_thread(self) -> None:
        """Stop ping thread"""
        if self.ping_thread and self.ping_thread.is_alive():
            self.ping_thread.join(timeout=2.0)

    def _stop_message_thread(self) -> None:
        """Stop message processing thread"""
        self.message_thread_stop_event.set()
        with self.message_lock:
            self.message_condition.notify()
            
        if self.message_thread and self.message_thread.is_alive():
            self.message_thread.join(timeout=2.0)

    # WebSocket Event Handlers
    def _on_open(self, ws) -> None:
        """Handle WebSocket connection open - ENHANCED with proper message format"""
        self.connected = True
        self.logger.info("🔗 WebSocket connection opened")
        
        # Reset counters
        self.ack_count = 0
        self.update_count = 0
        
        # Start message processing thread
        self.message_thread = threading.Thread(target=self._process_message_queue, daemon=True)
        self.message_thread.start()
        
        # Start ping thread
        self.ping_thread = threading.Thread(target=self._ping_worker, daemon=True)
        self.ping_thread.start()
        
        # Send authentication - ENHANCED FORMAT
        auth_msg = self._create_auth_message_enhanced()
        self._add_message_to_queue(auth_msg)
        
        # Send mode selection after small delay - ENHANCED FORMAT
        time.sleep(0.5)
        if self.lite_mode:
            mode_msg = self._create_lite_mode_message_enhanced()
        else:
            mode_msg = self._create_full_mode_message_enhanced()
        self._add_message_to_queue(mode_msg)

    def _on_message(self, ws, message) -> None:
        """Handle incoming WebSocket messages"""
        try:
            if isinstance(message, bytes):
                self._process_binary_message(message)
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")

    def _on_error(self, ws, error) -> None:
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")
        asyncio.create_task(self._trigger_error(str(error)))

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle WebSocket connection close"""
        self.connected = False
        self.authenticated = False
        self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        if self.running and self.reconnect_enabled:
            asyncio.create_task(self._schedule_reconnection())

    async def _schedule_reconnection(self) -> None:
        """Schedule reconnection with exponential backoff"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            delay = min(5 * (2 ** self.reconnect_attempts), 60)
            self.reconnect_attempts += 1
            
            self.logger.info(f"Scheduling reconnection in {delay} seconds")
            await asyncio.sleep(delay)
            await self.connect()

    # Message Processing
    def _add_message_to_queue(self, message: bytearray) -> None:
        """Add message to queue for sending - optimized"""
        with self.message_lock:
            self.message_queue.append(message)
            self.message_condition.notify()

    def _process_message_queue(self) -> None:
        """Process outgoing message queue - optimized for high throughput"""
        while not self.message_thread_stop_event.is_set():
            with self.message_lock:
                while not self.message_thread_stop_event.is_set() and not self.message_queue:
                    self.message_condition.wait(timeout=1.0)  # Add timeout to prevent hanging
                
                if self.message_thread_stop_event.is_set():
                    break
                    
                # Process multiple messages at once for better performance
                messages_to_send = []
                while self.message_queue and len(messages_to_send) < 10:  # Batch up to 10 messages
                    messages_to_send.append(self.message_queue.pop(0))
            
            # Send all messages in batch
            for message in messages_to_send:
                try:
                    self.ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception as e:
                    self.logger.error(f"Failed to send message: {e}")
                    break  # Stop sending if there's an error

    def _ping_worker(self) -> None:
        """Send periodic ping messages - optimized"""
        while self.running and self.connected:
            try:
                if self.ws and self.connected:
                    ping_msg = bytes([0, 1, 11])
                    self.ws.send(ping_msg, opcode=websocket.ABNF.OPCODE_BINARY)
                time.sleep(30)  # Increased ping interval from 10 to 30 seconds for better performance
            except Exception as e:
                self.logger.error(f"Ping error: {e}")
                break

    def _process_binary_message(self, message: bytes) -> None:
        """Process incoming binary message - optimized for high throughput"""
        try:
            data = bytearray(message)
            
            _, resp_type = struct.unpack("!HB", data[:3])
            
            # Optimized message type handling - most common first
            if resp_type == self.MSG_DATA_FEED:
                self._handle_data_feed(data)
            elif resp_type == self.MSG_AUTH_RESPONSE:
                self._handle_auth_response(data)
            elif resp_type == self.MSG_SUBSCRIBE_RESPONSE:
                self._handle_subscribe_response(data)
            elif resp_type == self.MSG_UNSUBSCRIBE_RESPONSE:
                self._handle_unsubscribe_response(data)
            elif resp_type in [self.MSG_CHANNEL_RESUME, self.MSG_CHANNEL_PAUSE]:
                self._handle_channel_response(data, resp_type)
            elif resp_type == self.MSG_MODE_RESPONSE:
                self._handle_mode_response(data)
            # Remove unknown message type logging for performance
                
        except Exception as e:
            self.logger.error(f"Binary message processing error: {e}")

    def _handle_auth_response(self, data: bytearray) -> None:
        """Handle authentication response with enhanced logging"""
        try:
            self.logger.debug("🔐 Processing authentication response")
            offset = 4
            offset += 1
            field_length = struct.unpack("!H", data[offset:offset + 2])[0]
            offset += 2
            string_val = data[offset:offset + field_length].decode("utf-8")
            offset += field_length
            
            self.logger.debug(f"Auth response value: {string_val}")
            
            if string_val == "K":
                # Get acknowledgment count
                offset += 1
                field_length = struct.unpack("!H", data[offset:offset + 2])[0]
                offset += 2
                self.ack_count = struct.unpack(">I", data[offset:offset + 4])[0]
                
                self.authenticated = True
                self.logger.info(f"✅ Authentication successful, ack_count: {self.ack_count}")
            else:
                self.logger.error(f"❌ Authentication failed: {string_val}")
                
        except Exception as e:
            self.logger.error(f"Auth response processing error: {e}")

    def _handle_subscribe_response(self, data: bytearray) -> None:
        """Handle subscription response - optimized"""
        try:
            offset = 5
            field_length = struct.unpack("H", data[offset:offset + 2])[0]
            offset += 2
            string_val = data[offset:offset + 1].decode("latin-1")
            
            if string_val != "K":
                self.logger.error(f"Subscription failed: {string_val}")
                
        except Exception as e:
            self.logger.error(f"Subscribe response processing error: {e}")

    def _handle_unsubscribe_response(self, data: bytearray) -> None:
        """Handle unsubscription response - optimized"""
        try:
            offset = 5
            field_length = struct.unpack("H", data[offset:offset + 2])[0]
            offset += 2
            string_val = data[offset:offset + 1].decode("latin-1")
            
            if string_val != "K":
                self.logger.error(f"Unsubscription failed: {string_val}")
                
        except Exception as e:
            self.logger.error(f"Unsubscribe response processing error: {e}")

    def _handle_mode_response(self, data: bytearray) -> None:
        """Handle mode selection response with enhanced logging"""
        try:
            self.logger.debug("⚙️ Processing mode response")
            offset = 3
            field_count = struct.unpack("!B", data[offset:offset + 1])[0]
            offset += 1
            
            if field_count >= 1:
                offset += 1
                field_length = struct.unpack("!H", data[offset:offset + 2])[0]
                offset += 2
                string_val = data[offset:offset + field_length].decode("utf-8")
                
                if string_val == "K":
                    mode = "Lite" if self.lite_mode else "Full"
                    self.logger.info(f"✅ {mode} mode activated successfully")
                else:
                    self.logger.error(f"❌ Mode selection failed: {string_val}")
            else:
                self.logger.warning("Mode response has no fields")
                
        except Exception as e:
            self.logger.error(f"Mode response processing error: {e}")

    def _handle_channel_response(self, data: bytearray, channel_type: int) -> None:
        """Handle channel pause/resume response with enhanced logging"""
        try:
            self.logger.debug(f"📡 Processing channel response type: {channel_type}")
            offset = 5
            field_length = struct.unpack("!H", data[offset:offset + 2])[0]
            offset += 2
            string_val = data[offset:offset + field_length].decode("utf-8")
            
            if string_val == "K":
                action = "paused" if channel_type == self.MSG_CHANNEL_PAUSE else "resumed"
                self.logger.info(f"✅ Channel {action}")
            else:
                self.logger.error(f"❌ Channel operation failed: {string_val}")
                
        except Exception as e:
            self.logger.error(f"Channel response processing error: {e}")

    def _handle_data_feed(self, data: bytearray) -> None:
        """Handle market data feed - optimized for high throughput"""
        try:
            if self.ack_count > 0:
                self.update_count += 1
                message_num = struct.unpack(">I", data[3:7])[0]
                
                # Optimized acknowledgment - send ack much less frequently for better performance
                # This is the key fix for delay issues with high symbol counts
                ack_threshold = max(100, self.ack_count * 2)  # Increased from 10 to 100+ 
                if self.update_count >= ack_threshold:
                    ack_msg = self._create_acknowledgment_message(message_num)
                    self._add_message_to_queue(ack_msg)
                    self.update_count = 0
            
            scrip_count = struct.unpack("!H", data[7:9])[0]
            offset = 9
            
            # Process all scrips without debug logging for performance
            for _ in range(scrip_count):
                offset = self._process_single_scrip_data(data, offset)
                
        except Exception as e:
            self.logger.error(f"Data feed processing error: {e}")

    def _process_single_scrip_data(self, data: bytearray, offset: int) -> int:
        """Process single scrip data from feed"""
        try:
            data_type = struct.unpack("B", data[offset:offset + 1])[0]
            offset += 1
            
            if data_type == self.DATA_SNAPSHOT:  # Snapshot data
                return self._process_snapshot_data(data, offset)
            elif data_type == self.DATA_UPDATE:  # Update data
                return self._process_update_data(data, offset)
            elif data_type == self.DATA_LITE:   # Lite data
                return self._process_lite_data(data, offset)
            else:
                self.logger.warning(f"Unknown data type: {data_type}")
                return offset
                
        except Exception as e:
            self.logger.error(f"Scrip data processing error: {e}")
            return offset

    # ENHANCED MESSAGE CREATION METHODS - Updated from enhanced version
    
    def _create_auth_message_enhanced(self) -> bytearray:
        """Create authentication message - optimized"""
        try:
            buffer_size = 18 + len(self.hsm_token) + len(self.source)
            
            byte_buffer = bytearray()
            byte_buffer.extend(struct.pack("!H", buffer_size - 2))
            byte_buffer.extend(bytes([1]))  # ReqType
            byte_buffer.extend(bytes([4]))  # FieldCount
            
            # Field-1: AuthToken
            byte_buffer.extend(bytes([1]))
            byte_buffer.extend(struct.pack("!H", len(self.hsm_token)))
            byte_buffer.extend(self.hsm_token.encode())
            
            # Field-2: Mode
            byte_buffer.extend(bytes([2]))
            byte_buffer.extend(struct.pack("!H", 1))
            byte_buffer.extend("P".encode('utf-8'))
            
            # Field-3: Channel
            byte_buffer.extend(bytes([3]))
            byte_buffer.extend(struct.pack("!H", 1))
            byte_buffer.extend(bytes([1]))
            
            # Field-4: Source
            byte_buffer.extend(bytes([4]))
            byte_buffer.extend(struct.pack("!H", len(self.source)))
            byte_buffer.extend(self.source.encode())
            
            return byte_buffer
            
        except Exception as e:
            self.logger.error(f"Auth message creation error: {e}")
            return bytearray()

    def _create_full_mode_message_enhanced(self) -> bytearray:
        """Create full mode message - ENHANCED FORMAT from enhanced version"""
        try:
            self.logger.debug(f"⚙️ Creating full mode message for channel {self.channel_num}")
            
            data = bytearray()
            
            # Data length (to be updated)
            data.extend(struct.pack(">H", 0))
            
            # Request type (12 for mode change)
            data.extend(struct.pack("B", 12))
            
            # Field count
            data.extend(struct.pack("B", 2))
            
            # Calculate channel bits for channel
            channel_bits = 1 << self.channel_num
            
            # Field-1: Channel bits
            data.extend(struct.pack("B", 1))
            data.extend(struct.pack(">H", 8))
            data.extend(struct.pack(">Q", channel_bits))
            
            # Field-2: Mode (70 for full mode)
            data.extend(struct.pack("B", 2))
            data.extend(struct.pack(">H", 1))
            data.extend(struct.pack("B", 70))
            
            # Update data length
            data_length = len(data) - 2
            struct.pack_into(">H", data, 0, data_length)
            
            self.logger.debug(f"📤 Full mode message created: length={len(data)}, channel_bits={channel_bits}, hex={data.hex()}")
            return data
            
        except Exception as e:
            self.logger.error(f"Full mode message creation error: {e}")
            return bytearray()

    def _create_lite_mode_message_enhanced(self) -> bytearray:
        """Create lite mode message - ENHANCED FORMAT from enhanced version"""
        try:
            self.logger.debug(f"⚙️ Creating lite mode message for channel {self.channel_num}")
            
            data = bytearray()
            data.extend(struct.pack(">H", 0))
            data.extend(struct.pack("B", 12))
            data.extend(struct.pack("B", 2))
            
            # Channel bits for channel
            channel_bits = 1 << self.channel_num
            
            # Field-1
            data.extend(struct.pack("B", 1))
            data.extend(struct.pack(">H", 8))
            data.extend(struct.pack(">Q", channel_bits))
            
            # Field-2
            data.extend(struct.pack("B", 2))
            data.extend(struct.pack(">H", 1))
            data.extend(struct.pack("B", 76))  # Lite mode
            
            # Update length
            data[:2] = struct.pack(">H", len(data) - 2)
            
            self.logger.debug(f"📤 Lite mode message created: length={len(data)}, channel_bits={channel_bits}, hex={data.hex()}")
            return data
            
        except Exception as e:
            self.logger.error(f"Lite mode message creation error: {e}")
            return bytearray()

    def _create_subscription_message_enhanced(self, symbols: List[str]) -> bytearray:
        """Create subscription message - optimized"""
        try:
            # Create scrips data
            scrips_data = bytearray()
            scrips_data.extend(struct.pack(">H", len(symbols)))
            
            for symbol in symbols:
                scrip_bytes = str(symbol).encode("ascii")
                scrips_data.append(len(scrip_bytes))
                scrips_data.extend(scrip_bytes)
            
            # Create message
            buffer_msg = bytearray()
            data_len = 6 + len(scrips_data)
            buffer_msg.extend(struct.pack(">H", data_len))
            buffer_msg.append(4)  # Request type for subscription
            buffer_msg.append(2)  # Field count
            
            # Field-1: Scrips data
            buffer_msg.append(1)
            buffer_msg.extend(struct.pack(">H", len(scrips_data)))
            buffer_msg.extend(scrips_data)
            
            # Field-2: Channel
            buffer_msg.append(2)
            buffer_msg.extend(struct.pack(">H", 1))
            buffer_msg.append(self.channel_num)
            
            return buffer_msg
            
        except Exception as e:
            self.logger.error(f"Subscription message creation error: {e}")
            return bytearray()

    def _create_unsubscription_message_enhanced(self, symbols: List[str]) -> bytearray:
        """Create unsubscription message - ENHANCED FORMAT from enhanced version"""
        try:
            self.logger.debug(f"Creating unsubscription message for {len(symbols)} symbols: {symbols}")
            
            # Create scrips data
            scrips_data = bytearray()
            scrips_data.extend(struct.pack(">H", len(symbols)))
            
            for symbol in symbols:
                scrip_bytes = str(symbol).encode("ascii")
                scrips_data.append(len(scrip_bytes))
                scrips_data.extend(scrip_bytes)
                self.logger.debug(f"   - Adding symbol: {symbol} (length: {len(scrip_bytes)})")
            
            # Create message
            buffer_msg = bytearray()
            data_len = 6 + len(scrips_data)
            buffer_msg.extend(struct.pack(">H", data_len))
            buffer_msg.append(5)  # Request type for unsubscription
            buffer_msg.append(2)  # Field count
            
            # Field-1: Scrips data
            buffer_msg.append(1)
            buffer_msg.extend(struct.pack(">H", len(scrips_data)))
            buffer_msg.extend(scrips_data)
            
            # Field-2: Channel
            buffer_msg.append(2)
            buffer_msg.extend(struct.pack(">H", 1))
            buffer_msg.append(self.channel_num)
            
            self.logger.debug(f"Unsubscription message created: length={len(buffer_msg)}, channel={self.channel_num}, hex={buffer_msg.hex()}")
            return buffer_msg
            
        except Exception as e:
            self.logger.error(f"Unsubscription message creation error: {e}")
            return bytearray()

    def _create_acknowledgment_message(self, message_number: int) -> bytearray:
        """Create acknowledgment message - optimized"""
        try:
            buffer_msg = bytearray()
            buffer_msg.extend(struct.pack(">H", 9))  # Total size - 2
            buffer_msg.extend(struct.pack("B", 3))   # Request type
            buffer_msg.extend(struct.pack("B", 1))   # Field count
            buffer_msg.extend(struct.pack("B", 1))   # Field ID
            buffer_msg.extend(struct.pack(">H", 4))  # Field size
            buffer_msg.extend(struct.pack(">I", message_number))
            
            return buffer_msg
            
        except Exception as e:
            self.logger.error(f"Acknowledgment message creation error: {e}")
            return bytearray()

    # Subscription methods - ENHANCED with proper message formats
    
    async def subscribe(self, symbols: List[str], data_type: str = "SymbolUpdate") -> bool:
        """Subscribe to market data for symbols with enhanced logging"""
        if not self.authenticated:
            self.logger.error("Not authenticated - cannot subscribe")
            return False
        
        try:
            self.logger.info(f"Subscribing to {len(symbols)} symbols: {symbols}")
            
            # Convert symbols to HSM format and create subscription message
            hsm_symbols = []
            for symbol in symbols:
                # This would need symbol to token conversion logic
                # For now, assuming symbols are already in correct format
                hsm_symbols.append(symbol)
                self.symbol_token_map[symbol] = symbol
                
            self.subscriptions.update(hsm_symbols)
            
            # Create subscription message with enhanced format
            sub_msg = self._create_subscription_message_enhanced(hsm_symbols)
            self._add_message_to_queue(sub_msg)
            
            self.logger.info(f"Subscription request sent for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Subscription error: {e}")
            return False

    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from market data for symbols with enhanced logging"""
        if not self.authenticated:
            self.logger.error("Not authenticated - cannot unsubscribe")
            return False
        
        try:
            self.logger.info(f"Unsubscribing from {len(symbols)} symbols: {symbols}")
            
            # Remove from subscriptions
            hsm_symbols = []
            for symbol in symbols:
                if symbol in self.subscriptions:
                    hsm_symbols.append(symbol)
                    self.subscriptions.remove(symbol)
                    self.symbol_token_map.pop(symbol, None)
            
            if hsm_symbols:
                # Create unsubscription message with enhanced format
                unsub_msg = self._create_unsubscription_message_enhanced(hsm_symbols)
                self._add_message_to_queue(unsub_msg)
                self.logger.info(f"Unsubscription request sent for {len(hsm_symbols)} symbols")
            else:
                self.logger.warning("No valid symbols found to unsubscribe")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Unsubscription error: {e}")
            return False

    # Data processing methods (simplified implementations)
    
    def _process_snapshot_data(self, data: bytearray, offset: int) -> int:
        """Process snapshot data feed - optimized for high throughput"""
        try:
            topic_id = struct.unpack("H", data[offset:offset + 2])[0]
            offset += 2
            
            topic_name_len = struct.unpack("B", data[offset:offset + 1])[0]
            offset += 1
            
            topic_name = data[offset:offset + topic_name_len].decode("utf-8")
            offset += topic_name_len
            
            # Create response dict
            response = {}
            
            # Process fields
            field_count = struct.unpack("B", data[offset:offset + 1])[0]
            offset += 1
            
            field_values = []
            for _ in range(field_count):
                value = struct.unpack(">i", data[offset:offset + 4])[0]
                offset += 4
                if value != -2147483648:  # Skip null values
                    field_values.append(value)
                else:
                    field_values.append(None)
            
            # Skip padding
            offset += 2
            
            # Get multiplier and precision
            multiplier = struct.unpack(">H", data[offset:offset + 2])[0]
            offset += 2
            
            precision = struct.unpack("B", data[offset:offset + 1])[0]
            offset += 1
            
            # Get strings (exchange, exchange_token, symbol)
            strings = []
            for _ in range(3):
                string_len = struct.unpack("B", data[offset:offset + 1])[0]
                offset += 1
                string_data = data[offset:offset + string_len].decode("utf-8", errors='ignore')
                offset += string_len
                strings.append(string_data)
            
            # Build response based on topic type - optimized
            if topic_name[:2] == "sf":  # Symbol feed
                self.scrips_sym[topic_id] = topic_name
                response = self._build_scrips_response(field_values, multiplier, precision, strings)
                response['symbol'] = self.symbol_token_map.get(topic_name, topic_name)
                
            elif topic_name[:2] == "if":  # Index feed
                self.index_sym[topic_id] = topic_name
                response = self._build_index_response(field_values, multiplier, precision, strings)
                response['symbol'] = self.symbol_token_map.get(topic_name, topic_name)
                
            elif topic_name[:2] == "dp":  # Depth feed
                self.dp_sym[topic_id] = topic_name
                response = self._build_depth_response(field_values, multiplier, precision, strings)
                response['symbol'] = self.symbol_token_map.get(topic_name, topic_name)
            
            # Store for future updates
            self.response_data[topic_name] = response
            
            # Optimized callback triggering - direct call when possible
            if hasattr(self, 'adapter_loop') and self.adapter_loop:
                # Use call_soon_threadsafe for better performance than run_coroutine_threadsafe
                self.adapter_loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._trigger_callback("on_message", response))
                )
            else:
                # Direct callback for maximum performance
                callback = self.callbacks.get("on_message")
                if callback:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            asyncio.create_task(callback(response))
                        else:
                            callback(response)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
            
            return offset
            
        except Exception as e:
            self.logger.error(f"Snapshot data processing error: {e}")
            return offset

    def _process_update_data(self, data: bytearray, offset: int) -> int:
        """Process update data feed - optimized for high throughput"""
        try:
            topic_id = struct.unpack("H", data[offset:offset + 2])[0]
            offset += 2
            
            field_count = struct.unpack("B", data[offset:offset + 1])[0]
            offset += 1
            
            # Get field values
            field_values = []
            for _ in range(field_count):
                value = struct.unpack(">i", data[offset:offset + 4])[0]
                offset += 4
                field_values.append(value)
            
            # Update existing data - optimized lookup
            topic_name = None
            response = {}
            
            if topic_id in self.scrips_sym:
                topic_name = self.scrips_sym[topic_id]
                response = self.response_data.get(topic_name, {})
                self._update_scrips_data(response, field_values)
                
            elif topic_id in self.index_sym:
                topic_name = self.index_sym[topic_id]
                response = self.response_data.get(topic_name, {})
                self._update_index_data(response, field_values)
                
            elif topic_id in self.dp_sym:
                topic_name = self.dp_sym[topic_id]
                response = self.response_data.get(topic_name, {})
                self._update_depth_data(response, field_values)
            
            if topic_name and response:
                self.response_data[topic_name] = response
                # Optimized callback triggering
                if hasattr(self, 'adapter_loop') and self.adapter_loop:
                    self.adapter_loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._trigger_callback("on_message", response))
                    )
                else:
                    # Direct callback for maximum performance
                    callback = self.callbacks.get("on_message")
                    if callback:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                asyncio.create_task(callback(response))
                            else:
                                callback(response)
                        except Exception as e:
                            self.logger.error(f"Callback error: {e}")
            
            return offset
            
        except Exception as e:
            self.logger.error(f"Update data processing error: {e}")
            return offset

    def _process_lite_data(self, data: bytearray, offset: int) -> int:
        """Process lite mode data feed (LTP only) - optimized for high throughput"""
        try:
            topic_id = struct.unpack("H", data[offset:offset + 2])[0]
            offset += 2
            
            value = struct.unpack(">i", data[offset:offset + 4])[0]
            offset += 4
            
            # Update LTP data - optimized
            topic_name = None
            response = {}
            
            if topic_id in self.scrips_sym:
                topic_name = self.scrips_sym[topic_id]
                response = self.response_data.get(topic_name, {})
                if value != -2147483648:
                    response['ltp'] = self._apply_precision(value, response.get('precision', 2), response.get('multiplier', 1))
                    
            elif topic_id in self.index_sym:
                topic_name = self.index_sym[topic_id]
                response = self.response_data.get(topic_name, {})
                if value != -2147483648:
                    response['ltp'] = self._apply_precision(value, response.get('precision', 2), response.get('multiplier', 1))
            
            if topic_name and response:
                response['type'] = 'ltp'
                self.response_data[topic_name] = response
                # Optimized callback triggering
                if hasattr(self, 'adapter_loop') and self.adapter_loop:
                    self.adapter_loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self._trigger_callback("on_message", response))
                    )
                else:
                    # Direct callback for maximum performance
                    callback = self.callbacks.get("on_message")
                    if callback:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                asyncio.create_task(callback(response))
                            else:
                                callback(response)
                        except Exception as e:
                            self.logger.error(f"Callback error: {e}")
            
            return offset
            
        except Exception as e:
            self.logger.error(f"Lite data processing error: {e}")
            return offset

    def _build_scrips_response(self, field_values: List, multiplier: int, precision: int, strings: List[str]) -> Dict:
        """Build scrips response data"""
        response = {
            'type': 'sf',
            'multiplier': multiplier,
            'precision': precision,
            'exchange': strings[0],
            'exchange_token': strings[1],
            'symbol': strings[2]
        }
        
        for i, value in enumerate(field_values):
            if i < len(self.data_val) and value is not None:
                field_name = self.data_val[i]
                if field_name in ['ltp', 'bid_price', 'ask_price', 'avg_trade_price', 
                                'low_price', 'high_price', 'open_price', 'prev_close_price']:
                    response[field_name] = self._apply_precision(value, precision, multiplier)
                else:
                    response[field_name] = value
        
        # Calculate change if possible
        if 'prev_close_price' in response and 'ltp' in response and response['prev_close_price'] != 0:
            change = response['ltp'] - response['prev_close_price']
            change_percent = (change / response['prev_close_price']) * 100
            response['ch'] = round(change, 4)
            response['chp'] = round(change_percent, 4)
        
        return response

    def _build_index_response(self, field_values: List, multiplier: int, precision: int, strings: List[str]) -> Dict:
        """Build index response data"""
        response = {
            'type': 'if',
            'multiplier': multiplier,
            'precision': precision,
            'exchange': strings[0],
            'exchange_token': strings[1],
            'symbol': strings[2]
        }
        
        for i, value in enumerate(field_values):
            if i < len(self.index_val) and value is not None:
                field_name = self.index_val[i]
                if field_name in ['ltp', 'prev_close_price', 'high_price', 'low_price', 'open_price']:
                    response[field_name] = self._apply_precision(value, precision, multiplier)
                else:
                    response[field_name] = value
        
        # Calculate change if possible
        if 'prev_close_price' in response and 'ltp' in response and response['prev_close_price'] != 0:
            change = response['ltp'] - response['prev_close_price']
            change_percent = (change / response['prev_close_price']) * 100
            response['ch'] = round(change, 2)
            response['chp'] = round(change_percent, 2)
        
        return response

    def _build_depth_response(self, field_values: List, multiplier: int, precision: int, strings: List[str]) -> Dict:
        """Build depth response data"""
        response = {
            'type': 'dp',
            'multiplier': multiplier,
            'precision': precision,
            'exchange': strings[0],
            'exchange_token': strings[1],
            'symbol': strings[2]
        }
        
        for i, value in enumerate(field_values):
            if i < len(self.depth_val) and value is not None:
                field_name = self.depth_val[i]
                if 'price' in field_name:
                    response[field_name] = self._apply_precision(value, precision, multiplier)
                else:
                    response[field_name] = value
        
        # Structure depth data
        response['depth'] = {
            'buy': [
                {'price': response.get('bid_price1', 0), 'quantity': response.get('bid_size1', 0), 'orders': response.get('bid_order1', 0)},
                {'price': response.get('bid_price2', 0), 'quantity': response.get('bid_size2', 0), 'orders': response.get('bid_order2', 0)},
                {'price': response.get('bid_price3', 0), 'quantity': response.get('bid_size3', 0), 'orders': response.get('bid_order3', 0)},
                {'price': response.get('bid_price4', 0), 'quantity': response.get('bid_size4', 0), 'orders': response.get('bid_order4', 0)},
                {'price': response.get('bid_price5', 0), 'quantity': response.get('bid_size5', 0), 'orders': response.get('bid_order5', 0)}
            ],
            'sell': [
                {'price': response.get('ask_price1', 0), 'quantity': response.get('ask_size1', 0), 'orders': response.get('ask_order1', 0)},
                {'price': response.get('ask_price2', 0), 'quantity': response.get('ask_size2', 0), 'orders': response.get('ask_order2', 0)},
                {'price': response.get('ask_price3', 0), 'quantity': response.get('ask_size3', 0), 'orders': response.get('ask_order3', 0)},
                {'price': response.get('ask_price4', 0), 'quantity': response.get('ask_size4', 0), 'orders': response.get('ask_order4', 0)},
                {'price': response.get('ask_price5', 0), 'quantity': response.get('ask_size5', 0), 'orders': response.get('ask_order5', 0)}
            ]
        }
        response['depth_level'] = 5
        
        return response

    def _update_scrips_data(self, response: Dict, field_values: List) -> None:
        """Update scrips data with new values"""
        for i, value in enumerate(field_values):
            if i < len(self.data_val) and value != -2147483648:
                field_name = self.data_val[i]
                if field_name in ['ltp', 'bid_price', 'ask_price', 'avg_trade_price', 
                                'low_price', 'high_price', 'open_price', 'prev_close_price']:
                    response[field_name] = self._apply_precision(value, response.get('precision', 2), response.get('multiplier', 1))
                else:
                    response[field_name] = value

    def _update_index_data(self, response: Dict, field_values: List) -> None:
        """Update index data with new values"""
        for i, value in enumerate(field_values):
            if i < len(self.index_val) and value != -2147483648:
                field_name = self.index_val[i]
                if field_name in ['ltp', 'prev_close_price', 'high_price', 'low_price', 'open_price']:
                    response[field_name] = self._apply_precision(value, response.get('precision', 2), response.get('multiplier', 1))
                else:
                    response[field_name] = value

    def _update_depth_data(self, response: Dict, field_values: List) -> None:
        """Update depth data with new values"""
        for i, value in enumerate(field_values):
            if i < len(self.depth_val) and value != -2147483648:
                field_name = self.depth_val[i]
                if 'price' in field_name:
                    response[field_name] = self._apply_precision(value, response.get('precision', 2), response.get('multiplier', 1))
                else:
                    response[field_name] = value

    def _apply_precision(self, value: int, precision: int, multiplier: int) -> float:
        """Apply precision and multiplier to convert raw value to actual price"""
        try:
            return value / ((10 ** precision) * multiplier)
        except:
            return 0.0

    # Callback management
    async def _trigger_callback(self, callback_name: str, *args) -> None:
        """Trigger callback if it exists"""
        callback = self.callbacks.get(callback_name)
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                self.logger.error(f"Callback {callback_name} error: {e}")

    async def _trigger_error(self, error_message: str) -> None:
        """Trigger error callback"""
        self.logger.error(error_message)
        await self._trigger_callback("on_error", error_message)

    # Utility methods
    def is_connected(self) -> bool:
        """Check if WebSocket is connected and authenticated"""
        return self.connected and self.authenticated

    def get_subscriptions(self) -> Set[str]:
        """Get current subscriptions"""
        return self.subscriptions.copy()

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            'connected': self.connected,
            'authenticated': self.authenticated,
            'running': self.running,
            'subscriptions_count': len(self.subscriptions),
            'lite_mode': self.lite_mode,
            'reconnect_attempts': self.reconnect_attempts,
            'ack_count': self.ack_count,
            'update_count': self.update_count
        }

    def set_callbacks(self, on_connect: Optional[Callable] = None, on_message: Optional[Callable] = None,
                     on_error: Optional[Callable] = None, on_close: Optional[Callable] = None) -> None:
        """Set callback functions"""
        if on_connect:
            self.callbacks["on_connect"] = on_connect
        if on_message:
            self.callbacks["on_message"] = on_message
        if on_error:
            self.callbacks["on_error"] = on_error
        if on_close:
            self.callbacks["on_close"] = on_close
            
        self.logger.info("Callbacks updated successfully")
