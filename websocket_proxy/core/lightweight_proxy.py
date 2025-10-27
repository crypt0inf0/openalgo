"""
Lightweight Disruptor Implementation
Balanced performance without excessive CPU usage for Flask integration
"""

import asyncio
import json
import os
import threading
import time
from typing import Dict, Any, Optional
from collections import deque
import queue

import websockets
from websockets.exceptions import ConnectionClosed

from ..production_config import config
from utils.logging import get_logger
from .subscription_manager import SubscriptionManager

logger = get_logger(__name__)


class LightweightRingBuffer:
    """
    Lightweight ring buffer that doesn't use busy-spin
    Optimized for balanced performance and CPU usage
    """
    
    def __init__(self, size: int = 65536):  # Smaller default size
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self.count = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.published_count = 0
        self.consumed_count = 0
        self.dropped_count = 0
    
    def publish(self, data: dict) -> bool:
        """Publish data to ring buffer"""
        with self.lock:
            if self.count >= self.size:
                # Buffer full - drop oldest (backpressure handling)
                self.tail = (self.tail + 1) % self.size
                self.count -= 1
                self.dropped_count += 1
            
            # Add new data
            self.buffer[self.head] = data
            self.head = (self.head + 1) % self.size
            self.count += 1
            self.published_count += 1
            return True
    
    def consume(self) -> Optional[dict]:
        """Consume data from ring buffer"""
        with self.lock:
            if self.count == 0:
                return None
            
            data = self.buffer[self.tail]
            self.buffer[self.tail] = None  # Clear reference
            self.tail = (self.tail + 1) % self.size
            self.count -= 1
            self.consumed_count += 1
            return data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                'size': self.size,
                'count': self.count,
                'utilization': (self.count / self.size) * 100,
                'published_count': self.published_count,
                'consumed_count': self.consumed_count,
                'dropped_count': self.dropped_count
            }


class LightweightEventProcessor:
    """
    Lightweight event processor using standard threading
    No busy-spin, uses blocking queue operations
    """
    
    def __init__(self, ring_buffer: LightweightRingBuffer, proxy_instance, main_loop=None):
        self.ring_buffer = ring_buffer
        self.proxy_instance = proxy_instance
        self.main_loop = main_loop  # Store reference to main event loop
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Statistics
        self.processed_count = 0
        self.last_process_time = time.time()
    
    def start(self):
        """Start the event processor"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logger.info("Started lightweight event processor")
    
    def stop(self):
        """Stop the event processor"""
        self.running = False
        if self.thread and self.thread.is_alive():
            # Give the thread a moment to finish naturally
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                logger.info("Event processor thread still running, allowing background cleanup")
            else:
                logger.info("Event processor thread finished successfully")
        logger.info("Stopped lightweight event processor")
    
    def _process_loop(self):
        """Main processing loop with yielding"""
        while self.running:
            try:
                # Consume data from ring buffer
                data = self.ring_buffer.consume()
                
                if data:
                    # Process the data
                    self._process_data(data)
                    self.processed_count += 1
                    self.last_process_time = time.time()
                else:
                    # No data - yield CPU to other threads
                    time.sleep(0.001)  # 1ms sleep instead of busy-spin
                
            except Exception as e:
                logger.error(f"Error in event processor: {e}")
                time.sleep(0.01)  # Brief pause on error
    
    def _process_data(self, data: dict):
        """Process market data"""
        try:
            symbol = data.get('symbol', '')
            exchange = data.get('exchange', 'NSE')
            mode = data.get('mode', 1)
            
            # Create topic
            mode_str = {1: 'LTP', 2: 'QUOTE', 3: 'DEPTH'}.get(mode, 'LTP')
            topic = f"{exchange}_{symbol}_{mode_str}"
            
            # Add proxy timestamp for latency measurement
            proxy_timestamp_ms = time.time() * 1000  # Current time in milliseconds
            
            # Create WebSocket message in OpenAlgo format (matching zeromq proxy)
            message = json.dumps({
                'type': 'market_data',
                'symbol': symbol,
                'exchange': exchange,
                'mode': mode,
                'broker': data.get('broker', 'unknown'),
                'broker_timestamp': data.get('broker_timestamp'),  # Pass through broker timestamp
                'proxy_timestamp': proxy_timestamp_ms,  # Add proxy timestamp
                'data': data
            })
            
            # IMPORTANT: Feed data into MarketDataService for client API compatibility
            try:
                from services.market_data_service import get_market_data_service
                market_service = get_market_data_service()
                
                # Create the data structure expected by MarketDataService
                market_data_message = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'mode': mode,
                    'data': data
                }
                
                # Process the data through the market service
                market_service.process_market_data(market_data_message)
                logger.debug(f"Fed market data to MarketDataService: {symbol} {mode_str}")
                
            except Exception as e:
                logger.warning(f"Failed to feed data to MarketDataService: {e}")
            
            # Broadcast to clients using OpenAlgo format
            if self.main_loop and not self.main_loop.is_closed():
                try:
                    # Send OpenAlgo format message
                    asyncio.run_coroutine_threadsafe(
                        self.proxy_instance._broadcast_to_topic(topic, message),
                        self.main_loop
                    )
                        
                except Exception as e:
                    logger.warning(f"Failed to schedule broadcast for {topic}: {e}")
            else:
                # Fallback: just count as processed
                logger.debug(f"Processed message for {topic} (no main loop available)")
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'processed_count': self.processed_count,
            'running': self.running,
            'last_process_time': self.last_process_time
        }


class LightweightWebSocketProxy:
    """
    Lightweight WebSocket proxy for balanced performance
    Uses standard threading instead of lock-free structures
    """
    
    def __init__(self, buffer_size: int = 65536):
        # Lightweight ring buffer
        self.ring_buffer = LightweightRingBuffer(buffer_size)
        
        # Proper subscription management with reference counting
        self.subscription_manager = SubscriptionManager()
        
        # Simple client management
        self.clients = set()
        self.client_metrics = {}  # websocket -> basic metrics
        
        # Single event processor (lightweight) - will set main_loop later
        self.event_processor = None
        
        # Server state
        self.running = False
        self.server = None
        
        # Orphaned symbol cleanup
        self.cleanup_interval = int(os.getenv('WEBSOCKET_CLEANUP_INTERVAL', '300'))  # 5 minutes default
        self.cleanup_task = None
        
        logger.info(f"Initialized LightweightWebSocketProxy with {buffer_size} buffer")
    
    async def start(self):
        """Start the lightweight WebSocket proxy"""
        logger.info("Starting Lightweight WebSocket Proxy")
        
        self.running = True
        
        # Store the main event loop for the event processor
        self._main_loop = asyncio.get_running_loop()
        
        # Create event processor with main loop reference
        self.event_processor = LightweightEventProcessor(
            self.ring_buffer, 
            self,
            self._main_loop
        )
        
        # Start event processor
        self.event_processor.start()
        
        # Start orphaned symbol cleanup task
        self.cleanup_task = asyncio.create_task(self._orphaned_symbol_cleanup_loop())
        
        # Simple monitoring (no heavy performance tracking)
        
        # Start WebSocket server
        self.server = await websockets.serve(
            self._handle_connection,
            config.WEBSOCKET_HOST,
            config.WEBSOCKET_PORT,
            ping_interval=30.0,  # 30 seconds
            ping_timeout=10.0,   # 10 seconds
            max_size=65536,      # 64KB
            compression=None     # No compression for speed
        )
        
        logger.info(f"Lightweight WebSocket server started on {config.WEBSOCKET_HOST}:{config.WEBSOCKET_PORT}")
        
        # Keep running
        try:
            # Create a future that completes when shutdown is requested
            shutdown_future = asyncio.Future()
            
            # Store the future so we can complete it during shutdown
            self._shutdown_future = shutdown_future
            
            # Wait for shutdown signal
            await shutdown_future
            
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Received shutdown signal")
        finally:
            await self.shutdown()
    
    def publish_market_data(self, symbol: str, data: dict) -> bool:
        """
        Publish market data (lightweight, no busy-spin)
        
        Args:
            symbol: Trading symbol
            data: Market data dictionary
            
        Returns:
            bool: True if published successfully
        """
        try:
            # Add symbol to data if not present
            if 'symbol' not in data:
                data['symbol'] = symbol
            
            # Publish to ring buffer (non-blocking)
            return self.ring_buffer.publish(data)
            
        except Exception as e:
            logger.error(f"Error publishing market data: {e}")
            return False
    
    async def _handle_connection(self, websocket):
        """Handle WebSocket client connections"""
        client_id = f"client-{id(websocket)}"
        logger.info(f"Client connected: {client_id}")
        
        # Add to client set
        self.clients.add(websocket)
        self.client_metrics[websocket] = {'connected_at': time.time()}
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'connected',
                'client_id': client_id,
                'server_time': int(time.time()),
                'architecture': 'lightweight-disruptor'
            }))
            
            # Handle messages
            async for message in websocket:
                try:
                    await self._handle_message(websocket, message)
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
        
        except ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Connection error with {client_id}: {e}")
        finally:
            self.clients.discard(websocket)
            self.client_metrics.pop(websocket, None)
            client_id = f"client-{id(websocket)}"
            
            # Clean up client subscriptions and handle broker unsubscriptions
            cleanup_info = self.subscription_manager.cleanup_client(client_id)
            
            # Process broker unsubscriptions for symbols with no remaining clients
            if cleanup_info['broker_unsubscriptions_needed']:
                await self._cleanup_client_subscriptions(websocket, cleanup_info)
            
            logger.info(f"Client disconnected: {client_id}")
    
    async def _handle_message(self, websocket, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'subscribe':
                await self._handle_subscription(websocket, data)
            elif action == 'unsubscribe':
                await self._handle_unsubscription(websocket, data)
            elif action == 'authenticate' or action == 'auth':
                await self._handle_authenticate(websocket, data)
            elif action == 'ping':
                await websocket.send(json.dumps({
                    'type': 'pong',
                    'timestamp': int(time.time() * 1000)
                }))
            elif action == 'get_stats':
                stats = self.get_stats()
                await websocket.send(json.dumps({
                    'type': 'stats',
                    'data': stats
                }))
            elif action == 'get_subscriptions':
                subscription_details = self.get_subscription_details()
                await websocket.send(json.dumps({
                    'type': 'subscription_details',
                    'data': subscription_details
                }))
            elif action == 'cleanup_orphaned':
                cleanup_result = await self._handle_orphaned_cleanup()
                await websocket.send(json.dumps({
                    'type': 'cleanup_result',
                    'data': cleanup_result
                }))
            elif action == 'force_cleanup_symbol':
                symbol = data.get('symbol')
                exchange = data.get('exchange', 'NSE')
                mode = data.get('mode', 1)
                if symbol:
                    success = await self._handle_force_cleanup_symbol(symbol, exchange, mode)
                    await websocket.send(json.dumps({
                        'type': 'force_cleanup_result',
                        'symbol': symbol,
                        'exchange': exchange,
                        'mode': mode,
                        'success': success
                    }))
                else:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Symbol is required for force cleanup'
                    }))
        
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'status': 'error',
                'code': 'INVALID_JSON',
                'message': 'Invalid JSON message'
            }))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_subscription(self, websocket, data):
        """Handle client subscriptions with proper reference counting"""
        try:
            symbol = data.get('symbol')
            exchange = data.get('exchange', 'NSE')
            mode = data.get('mode', 1)
            
            if not symbol:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'code': 'INVALID_PARAMETERS',
                    'message': 'Symbol is required'
                }))
                return
            
            client_id = f"client-{id(websocket)}"
            
            # Use subscription manager to handle the subscription
            success, should_subscribe_to_broker, message = self.subscription_manager.subscribe_client(
                client_id, symbol, exchange, mode
            )
            
            if not success:
                await websocket.send(json.dumps({
                    'type': 'subscribe',
                    'status': 'error',
                    'message': message
                }))
                return
            
            # Get user_id and broker info for broker subscription
            user_id = None
            broker_name = 'unknown'
            if hasattr(self, 'client_user_mapping'):
                user_id = self.client_user_mapping.get(id(websocket))
            
            # Only subscribe to broker if this is the first client for this symbol
            broker_subscription_status = 'reused'
            if should_subscribe_to_broker and user_id and hasattr(self, 'broker_adapters') and user_id in self.broker_adapters:
                try:
                    adapter = self.broker_adapters[user_id]
                    result = adapter.subscribe(symbol, exchange, mode)
                    
                    if result and result.get('status') == 'error':
                        logger.warning(f"Broker subscription failed: {result.get('message')}")
                        broker_subscription_status = 'failed'
                    elif result and result.get('status') == 'success':
                        logger.debug(f"Broker subscription successful: {result.get('message')}")
                        broker_subscription_status = 'success'
                        broker_name = result.get('broker', 'unknown')
                    
                except Exception as e:
                    logger.error(f"Error in broker subscription: {e}")
                    broker_subscription_status = 'error'
            elif should_subscribe_to_broker:
                logger.warning(f"Should subscribe to broker for {symbol} but no adapter available")
                broker_subscription_status = 'no_adapter'
            
            # Get current subscription count for this symbol
            subscriber_count = self.subscription_manager.get_symbol_subscribers_count(symbol, exchange, mode)
            
            await websocket.send(json.dumps({
                'type': 'subscribe',
                'status': 'success',
                'subscriptions': [{
                    'symbol': symbol,
                    'exchange': exchange,
                    'status': 'success',
                    'mode': {1: 'LTP', 2: 'Quote', 3: 'Depth'}.get(mode, 'LTP'),
                    'broker': broker_name,
                    'broker_subscription': broker_subscription_status,
                    'subscriber_count': subscriber_count
                }],
                'message': message,
                'broker': broker_name
            }))
            
            mode_str = {1: 'LTP', 2: 'QUOTE', 3: 'DEPTH'}.get(mode, 'LTP')
            logger.info(f"Client {client_id} subscribed to {exchange}_{symbol}_{mode_str} "
                       f"(subscribers: {subscriber_count}, broker_action: {broker_subscription_status})")
            
        except Exception as e:
            logger.error(f"Error handling subscription: {e}")
            await websocket.send(json.dumps({
                'status': 'error',
                'code': 'SERVER_ERROR',
                'message': str(e)
            }))
    
    async def _handle_unsubscription(self, websocket, data):
        """Handle client unsubscriptions with proper reference counting"""
        try:
            symbol = data.get('symbol')
            exchange = data.get('exchange', 'NSE')
            mode = data.get('mode', 1)
            
            if not symbol:
                await websocket.send(json.dumps({
                    'type': 'unsubscription',
                    'status': 'error',
                    'message': 'Symbol is required'
                }))
                return
            
            client_id = f"client-{id(websocket)}"
            
            # Use subscription manager to handle the unsubscription
            success, should_unsubscribe_from_broker, message = self.subscription_manager.unsubscribe_client(
                client_id, symbol, exchange, mode
            )
            
            if not success:
                await websocket.send(json.dumps({
                    'type': 'unsubscribe',
                    'status': 'error',
                    'message': message
                }))
                return
            
            # Get user_id and broker info for broker unsubscription
            user_id = None
            broker_name = 'unknown'
            if hasattr(self, 'client_user_mapping'):
                user_id = self.client_user_mapping.get(id(websocket))
            
            # Only unsubscribe from broker if this was the last client for this symbol
            broker_unsubscription_status = 'kept'
            if should_unsubscribe_from_broker and user_id and hasattr(self, 'broker_adapters') and user_id in self.broker_adapters:
                try:
                    adapter = self.broker_adapters[user_id]
                    result = adapter.unsubscribe(symbol, exchange, mode)
                    
                    if result and not result.get('success', True):
                        logger.warning(f"Broker unsubscription failed: {result.get('error')}")
                        broker_unsubscription_status = 'failed'
                    else:
                        logger.debug(f"Broker unsubscription successful")
                        broker_unsubscription_status = 'success'
                        broker_name = result.get('broker', 'unknown') if result else 'unknown'
                    
                except Exception as e:
                    logger.error(f"Error in broker unsubscription: {e}")
                    broker_unsubscription_status = 'error'
            elif should_unsubscribe_from_broker:
                logger.warning(f"Should unsubscribe from broker for {symbol} but no adapter available")
                broker_unsubscription_status = 'no_adapter'
            
            # Get remaining subscription count for this symbol
            remaining_subscribers = self.subscription_manager.get_symbol_subscribers_count(symbol, exchange, mode)
            
            await websocket.send(json.dumps({
                'type': 'unsubscribe',
                'status': 'success',
                'message': message,
                'successful': [{
                    'symbol': symbol,
                    'exchange': exchange,
                    'status': 'success',
                    'broker': broker_name,
                    'broker_unsubscription': broker_unsubscription_status,
                    'remaining_subscribers': remaining_subscribers
                }],
                'failed': [],
                'broker': broker_name
            }))
            
            mode_str = {1: 'LTP', 2: 'QUOTE', 3: 'DEPTH'}.get(mode, 'LTP')
            logger.info(f"Client {client_id} unsubscribed from {exchange}_{symbol}_{mode_str} "
                       f"(remaining: {remaining_subscribers}, broker_action: {broker_unsubscription_status})")
            
        except Exception as e:
            logger.error(f"Error handling unsubscription: {e}")
            await websocket.send(json.dumps({
                'status': 'error',
                'code': 'SERVER_ERROR',
                'message': str(e)
            }))
    
    async def _handle_authenticate(self, websocket, data):
        """Handle authentication requests"""
        try:
            # Import database functions if available
            try:
                from database.auth_db import verify_api_key, get_broker_name
            except ImportError:
                # If auth is not available, allow all connections
                await websocket.send(json.dumps({
                    'type': 'auth',
                    'status': 'success',
                    'message': 'Authentication not required',
                    'user_id': 'anonymous',
                    'broker': 'none'
                }))
                return
            
            api_key = data.get('api_key')
            if not api_key:
                await websocket.send(json.dumps({
                    'type': 'auth',
                    'status': 'error',
                    'message': 'API key is required'
                }))
                return
            
            # Verify API key
            user_id = verify_api_key(api_key)
            if not user_id:
                await websocket.send(json.dumps({
                    'type': 'auth',
                    'status': 'error',
                    'message': 'Invalid API key'
                }))
                return
            
            # Get broker name
            broker_name = None
            try:
                broker_name = get_broker_name(api_key)
            except Exception:
                broker_name = 'unknown'
            
            # Mark client as authenticated
            if websocket in self.client_metrics:
                # Store user mapping for broker adapter creation
                if not hasattr(self, 'client_user_mapping'):
                    self.client_user_mapping = {}
                self.client_user_mapping[id(websocket)] = user_id
                
                # Create broker adapter for this user
                await self._create_broker_adapter_for_user(user_id, broker_name)
            
            # Check if broker adapter connected successfully
            broker_connected = False
            if hasattr(self, 'broker_adapters') and user_id in self.broker_adapters:
                broker_connected = True
            
            auth_message = {
                'type': 'auth',
                'status': 'success',
                'message': 'Authentication successful',
                'broker': broker_name or 'unknown',
                'user_id': user_id,
                'supported_features': {
                    'ltp': True,
                    'quote': True,
                    'depth': True
                }
            }
            
            # Add helpful message if broker connection failed
            if not broker_connected and broker_name == 'angel':
                auth_message['broker_status'] = 'disconnected'
                auth_message['broker_message'] = 'Login to Angel via http://127.0.0.1:5000 to get real market data'
            
            await websocket.send(json.dumps(auth_message))
            
            logger.info(f"Client authenticated: user_id={user_id}, broker={broker_name}")
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await websocket.send(json.dumps({
                'type': 'auth',
                'status': 'error',
                'message': 'Authentication failed'
            }))
    
    async def _create_broker_adapter_for_user(self, user_id: str, broker_name: str):
        """Create and connect broker adapter for user"""
        try:
            # Import broker factory
            try:
                from ..adapters.broker_factory import create_broker_adapter
            except ImportError:
                logger.warning("Broker factory not available")
                return
            
            # Check if adapter already exists
            if hasattr(self, 'broker_adapters') and user_id in self.broker_adapters:
                logger.info(f"Broker adapter already exists for user {user_id}")
                return
            
            # Initialize broker adapters dict if not exists
            if not hasattr(self, 'broker_adapters'):
                self.broker_adapters = {}
            if not hasattr(self, 'client_user_mapping'):
                self.client_user_mapping = {}
            
            # Create adapter
            logger.info(f"Creating {broker_name} adapter for user {user_id}")
            adapter = create_broker_adapter(broker_name)
            if not adapter:
                logger.error(f"Failed to create {broker_name} adapter")
                return
            logger.info(f"Successfully created {broker_name} adapter")
            
            # Set proxy reference in adapter
            if hasattr(adapter, 'set_proxy'):
                adapter.set_proxy(self)
                logger.info(f"Set proxy reference in {broker_name} adapter using set_proxy method")
            elif hasattr(adapter, 'zbp_proxy'):
                adapter.zbp_proxy = self
                logger.info(f"Set proxy reference in {broker_name} adapter directly")
            
            # Initialize adapter
            logger.info(f"Initializing {broker_name} adapter for user {user_id}")
            try:
                result = adapter.initialize(broker_name, user_id)
                logger.info(f"Adapter initialization result: {result}")
                if result and not result.get('success', True):
                    logger.error(f"Failed to initialize {broker_name} adapter: {result.get('error')}")
                    return
            except Exception as e:
                logger.error(f"Exception during adapter initialization: {e}")
                return
            
            # Connect adapter
            logger.info(f"Connecting {broker_name} adapter")
            try:
                connect_result = adapter.connect()
                logger.info(f"Adapter connection result: {connect_result}")
                if connect_result and not connect_result.get('success', True):
                    logger.warning(f"Failed to connect {broker_name} adapter: {connect_result.get('error')}")
                    logger.info(f"WebSocket subscriptions will work, but no real market data until broker is connected")
            except Exception as e:
                logger.error(f"Exception during adapter connection: {e}")
                logger.info(f"WebSocket subscriptions will work, but no real market data until broker is connected")
            
            # Store adapter
            self.broker_adapters[user_id] = adapter
            
            if connect_result and connect_result.get('success', True):
                logger.info(f"Created and connected {broker_name} adapter for user {user_id}")
            else:
                logger.info(f"Created {broker_name} adapter for user {user_id} (connection pending valid tokens)")
            
        except Exception as e:
            logger.error(f"Error creating broker adapter: {e}")
    
    async def _cleanup_client_subscriptions(self, websocket, cleanup_info):
        """Handle broker unsubscriptions when client disconnects"""
        try:
            # Get user_id for broker operations
            user_id = None
            if hasattr(self, 'client_user_mapping'):
                user_id = self.client_user_mapping.get(id(websocket))
            
            if not user_id or not hasattr(self, 'broker_adapters') or user_id not in self.broker_adapters:
                logger.debug("No broker adapter available for client cleanup")
                return
            
            adapter = self.broker_adapters[user_id]
            
            # Process each broker unsubscription needed
            for unsub_info in cleanup_info['broker_unsubscriptions_needed']:
                try:
                    result = adapter.unsubscribe(
                        unsub_info['symbol'], 
                        unsub_info['exchange'], 
                        unsub_info['mode']
                    )
                    
                    if result and not result.get('success', True):
                        logger.warning(f"Broker cleanup unsubscription failed for {unsub_info['topic']}: {result.get('error')}")
                    else:
                        logger.debug(f"Broker cleanup unsubscription successful for {unsub_info['topic']}")
                        
                except Exception as e:
                    logger.error(f"Error in broker cleanup unsubscription for {unsub_info['topic']}: {e}")
            
            logger.info(f"Processed {len(cleanup_info['broker_unsubscriptions_needed'])} broker unsubscriptions during client cleanup")
            
        except Exception as e:
            logger.error(f"Error in _cleanup_client_subscriptions: {e}")

    async def _broadcast_to_topic(self, topic: str, message: str):
        """Broadcast message to all clients subscribed to topic using subscription manager"""
        try:
            # Get all clients subscribed to this topic from subscription manager
            clients_to_notify = []
            
            for websocket in self.clients:
                client_id = f"client-{id(websocket)}"
                client_subscriptions = self.subscription_manager.get_client_subscriptions(client_id)
                if topic in client_subscriptions:
                    clients_to_notify.append(websocket)
            
            # Send message to all subscribed clients
            for websocket in clients_to_notify:
                try:
                    await websocket.send(message)
                except Exception as e:
                    client_id = f"client-{id(websocket)}"
                    logger.debug(f"Failed to send to client {client_id}: {e}")
            
            if clients_to_notify:
                logger.debug(f"Broadcasted message to {len(clients_to_notify)} clients for topic {topic}")
                
        except Exception as e:
            logger.error(f"Error in _broadcast_to_topic: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'architecture': 'lightweight-disruptor',
            'ring_buffer': self.ring_buffer.get_stats(),
            'event_processor': self.event_processor.get_stats() if self.event_processor else {},
            'clients': {'count': len(self.clients)},
            'subscription_manager': self.subscription_manager.get_stats(),
            'running': self.running
        }
    
    def get_subscription_details(self) -> Dict[str, Any]:
        """Get detailed subscription information for debugging"""
        return self.subscription_manager.get_detailed_status()
    
    async def _orphaned_symbol_cleanup_loop(self):
        """Periodic cleanup of orphaned symbols"""
        logger.info(f"Started orphaned symbol cleanup loop (interval: {self.cleanup_interval}s)")
        
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                if not self.running:
                    break
                
                # Perform cleanup
                cleanup_info = self.subscription_manager.cleanup_orphaned_symbols()
                
                # Process broker unsubscriptions for orphaned symbols
                if cleanup_info['broker_unsubscriptions_needed']:
                    await self._process_orphaned_broker_unsubscriptions(cleanup_info['broker_unsubscriptions_needed'])
                
                if cleanup_info['cleaned_count'] > 0:
                    logger.info(f"Periodic cleanup: removed {cleanup_info['cleaned_count']} orphaned symbols")
                
            except asyncio.CancelledError:
                logger.info("Orphaned symbol cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in orphaned symbol cleanup loop: {e}")
                # Continue running despite errors
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _process_orphaned_broker_unsubscriptions(self, unsubscriptions_needed):
        """Process broker unsubscriptions for orphaned symbols"""
        try:
            # Find any available broker adapter to perform unsubscriptions
            adapter = None
            if hasattr(self, 'broker_adapters') and self.broker_adapters:
                adapter = next(iter(self.broker_adapters.values()))
            
            if not adapter:
                logger.warning("No broker adapter available for orphaned symbol cleanup")
                return
            
            successful_cleanups = 0
            failed_cleanups = 0
            
            for unsub_info in unsubscriptions_needed:
                try:
                    result = adapter.unsubscribe(
                        unsub_info['symbol'],
                        unsub_info['exchange'],
                        unsub_info['mode']
                    )
                    
                    if result and result.get('success', True):
                        successful_cleanups += 1
                        logger.debug(f"Successfully cleaned orphaned broker subscription: {unsub_info['topic']}")
                    else:
                        failed_cleanups += 1
                        logger.warning(f"Failed to clean orphaned broker subscription: {unsub_info['topic']}")
                    
                except Exception as e:
                    failed_cleanups += 1
                    logger.error(f"Error cleaning orphaned broker subscription {unsub_info['topic']}: {e}")
            
            logger.info(f"Orphaned broker cleanup: {successful_cleanups} successful, {failed_cleanups} failed")
            
        except Exception as e:
            logger.error(f"Error in _process_orphaned_broker_unsubscriptions: {e}")
    
    async def _handle_orphaned_cleanup(self) -> Dict[str, Any]:
        """Handle manual orphaned symbol cleanup request"""
        try:
            cleanup_info = self.subscription_manager.cleanup_orphaned_symbols()
            
            # Process broker unsubscriptions
            if cleanup_info['broker_unsubscriptions_needed']:
                await self._process_orphaned_broker_unsubscriptions(cleanup_info['broker_unsubscriptions_needed'])
            
            return cleanup_info
            
        except Exception as e:
            logger.error(f"Error in manual orphaned cleanup: {e}")
            return {'error': str(e), 'cleaned_count': 0}
    
    async def _handle_force_cleanup_symbol(self, symbol: str, exchange: str, mode: int) -> bool:
        """Handle force cleanup of a specific symbol"""
        try:
            # Force cleanup in subscription manager
            was_broker_subscribed = self.subscription_manager.force_cleanup_symbol(symbol, exchange, mode)
            
            # If it was broker subscribed, try to unsubscribe from broker
            if was_broker_subscribed:
                if hasattr(self, 'broker_adapters') and self.broker_adapters:
                    adapter = next(iter(self.broker_adapters.values()))
                    try:
                        result = adapter.unsubscribe(symbol, exchange, mode)
                        if result and not result.get('success', True):
                            logger.warning(f"Force cleanup broker unsubscription failed for {symbol}: {result.get('error')}")
                    except Exception as e:
                        logger.error(f"Error in force cleanup broker unsubscription for {symbol}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in force cleanup symbol {symbol}: {e}")
            return False
    
    def force_cleanup_orphaned_symbols(self):
        """Force cleanup of orphaned symbols (synchronous method for external calls)"""
        try:
            cleanup_info = self.subscription_manager.cleanup_orphaned_symbols()
            
            # Schedule broker unsubscriptions if main loop is available
            if cleanup_info['broker_unsubscriptions_needed'] and hasattr(self, '_main_loop') and not self._main_loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._process_orphaned_broker_unsubscriptions(cleanup_info['broker_unsubscriptions_needed']),
                    self._main_loop
                )
            
            return cleanup_info
            
        except Exception as e:
            logger.error(f"Error in force cleanup orphaned symbols: {e}")
            return {'error': str(e), 'cleaned_count': 0}
    
    async def shutdown(self):
        """Shutdown the lightweight proxy"""
        logger.info("Shutting down Lightweight WebSocket Proxy")
        
        self.running = False
        
        # Stop cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Stop event processor
        if self.event_processor:
            self.event_processor.stop()
        
        # Stop client monitoring
        pass  # Simplified client management
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("Lightweight WebSocket Proxy shutdown complete")
    
    def shutdown_sync(self):
        """Synchronous shutdown method for cleanup handlers"""
        logger.info("Shutting down Lightweight WebSocket Proxy (sync)")
        
        self.running = False
        
        # Signal shutdown to the main loop
        if hasattr(self, '_shutdown_future') and not self._shutdown_future.done():
            try:
                self._shutdown_future.set_result(None)
            except Exception as e:
                logger.debug(f"Could not signal shutdown future: {e}")
        
        # Stop event processor
        if self.event_processor:
            self.event_processor.stop()
        
        # Stop client monitoring
        pass  # Simplified client management
        
        # Close server (sync)
        if self.server:
            try:
                self.server.close()
                # Don't wait for server close in sync mode to avoid event loop issues
                logger.info("WebSocket server close initiated")
            except Exception as e:
                logger.warning(f"Error closing WebSocket server: {e}")
        
        logger.info("Lightweight WebSocket Proxy shutdown complete (sync)")
