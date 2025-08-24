import asyncio as aio
import websockets
import json
import time
from utils.logging import get_logger, highlight_url
import signal
import threading
import os
import socket
from typing import Dict, Set, Any, Optional
from dotenv import load_dotenv

from .port_check import is_port_in_use, find_available_port
from database.auth_db import get_broker_name, verify_api_key
from .broker_factory import create_broker_adapter
from websocket_proxy.cross_platform_structures import CrossPlatformRingBuffer, MarketTick, CrossPlatformAtomic

# Initialize logger
logger = get_logger("ultra_websocket_proxy")

class UltraWebSocketProxy:
    """
    Ultra-Low Latency WebSocket Proxy Server using shared memory ring buffers
    Eliminates ZeroMQ overhead for sub-microsecond market data delivery
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        
        # Check port availability
        if is_port_in_use(host, port, wait_time=2.0):
            error_msg = (
                f"WebSocket port {port} is already in use on {host}.\n"
                f"Cannot start WebSocket server with port switching as it would break SDK clients."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.clients = {}  # Maps client_id to websocket connection
        self.subscriptions = {}  # Maps client_id to set of subscriptions
        self.broker_adapters = {}  # Maps user_id to broker adapter
        self.user_mapping = {}  # Maps client_id to user_id
        self.user_broker_mapping = {}  # Maps user_id to broker_name
        self.running = False
        
        # Performance counters
        self.message_count = CrossPlatformAtomic(0)
        self.client_count = CrossPlatformAtomic(0)
        
        # Get shared ring buffer from adapters
        self.ring_buffer = None
        self._init_shared_buffer()

        logger.info(f"Ultra WebSocket server initialized on {host}:{port}")

    def _init_shared_buffer(self):
        """Initialize connection to shared memory ring buffer"""
        try:
            # We'll get the ring buffer reference from the first adapter that connects
            # For now, just mark as None - will be set when first adapter initializes
            self.ring_buffer = None
            logger.info("Shared memory ring buffer will be initialized by first adapter")
        except Exception as e:
            logger.error(f"Failed to initialize shared buffer connection: {e}")
            raise

    async def start(self):
        """Start the ultra-low latency WebSocket server"""
        self.running = True
        
        try:
            # Start shared memory listener
            logger.info("Starting shared memory listener")
            shm_task = aio.create_task(self.shared_memory_listener())
            
            # Start WebSocket server
            stop = aio.Future()
            
            async def monitor_shutdown():
                while self.running:
                    await aio.sleep(0.5)
                stop.set_result(None)
            
            monitor_task = aio.create_task(monitor_shutdown())
            
            highlighted_address = highlight_url(f"{self.host}:{self.port}")
            logger.info(f"Starting Ultra-Low Latency WebSocket server on {highlighted_address}")
            
            # Start WebSocket server
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                reuse_port=True if hasattr(socket, 'SO_REUSEPORT') else False
            )
            
            logger.info(f"Ultra WebSocket server started on {highlighted_address}")
            
            await stop
            
            # Cancel monitor task
            monitor_task.cancel()
            try:
                await monitor_task
            except aio.CancelledError:
                pass
                
        except Exception as e:
            logger.exception(f"Failed to start ultra WebSocket server: {e}")
            raise

    async def shared_memory_listener(self):
        """Listen for messages from shared memory ring buffer"""
        logger.info("Starting shared memory listener")
        
        consecutive_empty_reads = 0
        max_empty_reads = 100
        
        while self.running:
            try:
                # Get ring buffer reference from adapter if not yet available
                if not self.ring_buffer:
                    self._try_get_ring_buffer()
                    if not self.ring_buffer:
                        await aio.sleep(0.001)  # 1ms wait
                        continue
                
                # Try to consume message from ring buffer
                message_bytes = self.ring_buffer.consume()
                
                if message_bytes:
                    consecutive_empty_reads = 0
                    
                    # Deserialize market tick
                    try:
                        tick = MarketTick.from_bytes(message_bytes)
                        await self.process_market_tick(tick)
                        self.message_count.fetch_add(1)
                        
                    except Exception as e:
                        logger.error(f"Error deserializing market tick: {e}")
                        logger.debug(f"Message bytes length: {len(message_bytes) if message_bytes else 'None'}")
                        continue
                        
                else:
                    # No message available
                    consecutive_empty_reads += 1
                    
                    if consecutive_empty_reads < max_empty_reads:
                        # Busy wait for ultra-low latency
                        continue
                    else:
                        # Brief sleep to prevent CPU spinning
                        await aio.sleep(0.0001)  # 100 microseconds
                        consecutive_empty_reads = 0
                        
            except Exception as e:
                logger.error(f"Error in shared memory listener: {e}")
                await aio.sleep(0.001)

    def _try_get_ring_buffer(self):
        """Try to get ring buffer reference"""
        try:
            from websocket_proxy.ultra_low_latency_adapter import UltraLowLatencyAdapter
            
            if UltraLowLatencyAdapter._shared_ring_buffer:
                self.ring_buffer = UltraLowLatencyAdapter._shared_ring_buffer
                stats = self.ring_buffer.get_stats()
                logger.info(f"Connected to {stats.get('implementation', 'unknown')} ring buffer")
                return True
        except Exception as e:
            logger.debug(f"Ring buffer not yet available: {e}")
        
        return False

    async def process_market_tick(self, tick: MarketTick):
        """Process market tick and forward to subscribed clients"""
        try:
            # Get symbol info from tick
            symbol = tick.symbol
            exchange = tick.exchange
            mode = tick.mode
            
            # Convert to client format
            market_data = {
                "type": "market_data",
                "symbol": symbol,
                "exchange": exchange,
                "mode": mode,
                "broker": "flattrade",  # TODO: Get from tick
                "data": {
                    "ltp": tick.ltp,
                    "volume": tick.volume,
                    "open": tick.open,
                    "high": tick.high,
                    "low": tick.low,
                    "close": tick.close,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "bid_qty": tick.bid_qty,
                    "ask_qty": tick.ask_qty,
                    "timestamp": tick.timestamp,
                    "sequence": tick.sequence
                }
            }
            
            # Find subscribed clients
            clients_to_notify = []
            for client_id, subscriptions in self.subscriptions.items():
                for sub_json in subscriptions:
                    try:
                        sub = json.loads(sub_json)
                        if (sub.get("symbol") == symbol and 
                            sub.get("exchange") == exchange and
                            sub.get("mode") == mode):
                            clients_to_notify.append(client_id)
                            break
                    except json.JSONDecodeError:
                        continue
            
            # Send to clients (parallel)
            if clients_to_notify:
                message_json = json.dumps(market_data)
                send_tasks = []
                
                for client_id in clients_to_notify:
                    if client_id in self.clients:
                        send_tasks.append(self.send_message_to_client(client_id, message_json))
                
                if send_tasks:
                    await aio.gather(*send_tasks, return_exceptions=True)
                    
        except Exception as e:
            logger.error(f"Error processing market tick: {e}")

    async def send_message_to_client(self, client_id: int, message_json: str):
        """Send message to specific client"""
        try:
            websocket = self.clients.get(client_id)
            if websocket:
                await websocket.send(message_json)
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"Connection closed while sending to client {client_id}")
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")

    async def handle_client(self, websocket):
        """Handle client connections"""
        client_id = id(websocket)
        self.clients[client_id] = websocket
        self.subscriptions[client_id] = set()
        self.client_count.fetch_add(1)
        
        path = getattr(websocket, 'path', '/unknown')
        logger.info(f"Ultra client connected: {client_id} from path: {path}")
        
        try:
            async for message in websocket:
                try:
                    await self.process_client_message(client_id, message)
                except Exception as e:
                    logger.exception(f"Error processing message from client {client_id}: {e}")
                    try:
                        await self.send_error(client_id, "PROCESSING_ERROR", str(e))
                    except:
                        pass
                        
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Ultra client disconnected: {client_id}, code: {e.code}")
        except Exception as e:
            logger.exception(f"Unexpected error handling ultra client {client_id}: {e}")
        finally:
            await self.cleanup_client(client_id)

    async def cleanup_client(self, client_id):
        """Clean up client resources"""
        # Remove client
        if client_id in self.clients:
            del self.clients[client_id]
            self.client_count.fetch_add(-1)
        
        # Clean up subscriptions
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        
        # Remove from user mapping
        if client_id in self.user_mapping:
            user_id = self.user_mapping[client_id]
            del self.user_mapping[client_id]
            
            # Check if last client for this user
            is_last_client = True
            for other_client_id, other_user_id in self.user_mapping.items():
                if other_client_id != client_id and other_user_id == user_id:
                    is_last_client = False
                    break
            
            # Disconnect adapter if last client
            if is_last_client and user_id in self.broker_adapters:
                adapter = self.broker_adapters[user_id]
                try:
                    adapter.disconnect()
                    del self.broker_adapters[user_id]
                    if user_id in self.user_broker_mapping:
                        del self.user_broker_mapping[user_id]
                    logger.info(f"Disconnected adapter for user {user_id}")
                except Exception as e:
                    logger.error(f"Error disconnecting adapter for user {user_id}: {e}")

    async def process_client_message(self, client_id, message):
        """Process client messages (same interface as original)"""
        try:
            data = json.loads(message)
            action = data.get("action") or data.get("type")
            
            if action in ["authenticate", "auth"]:
                await self.authenticate_client(client_id, data)
            elif action == "subscribe":
                await self.subscribe_client(client_id, data)
            elif action in ["unsubscribe", "unsubscribe_all"]:
                await self.unsubscribe_client(client_id, data)
            elif action == "get_performance_stats":
                await self.get_performance_stats(client_id)
            else:
                await self.send_error(client_id, "INVALID_ACTION", f"Invalid action: {action}")
                
        except json.JSONDecodeError:
            await self.send_error(client_id, "INVALID_JSON", "Invalid JSON message")
        except Exception as e:
            logger.exception(f"Error processing client message: {e}")
            await self.send_error(client_id, "SERVER_ERROR", str(e))

    async def authenticate_client(self, client_id, data):
        """Authenticate client using ultra-low latency adapter"""
        api_key = data.get("api_key")
        if not api_key:
            await self.send_error(client_id, "AUTHENTICATION_ERROR", "API key is required")
            return
        
        # Verify API key
        user_id = verify_api_key(api_key)
        if not user_id:
            await self.send_error(client_id, "AUTHENTICATION_ERROR", "Invalid API key")
            return
        
        # Store user mapping
        self.user_mapping[client_id] = user_id
        
        # Get broker name
        broker_name = get_broker_name(api_key)
        if not broker_name:
            await self.send_error(client_id, "BROKER_ERROR", "No broker configuration found")
            return
        
        self.user_broker_mapping[user_id] = broker_name
        
        # Create or reuse ultra-low latency adapter
        if user_id not in self.broker_adapters:
            try:
                # Import the ultra adapter
                from broker.flattrade.streaming.flattrade_ultra_adapter import FlattradeUltraAdapter
                adapter = FlattradeUltraAdapter()
                
                # Initialize and connect
                adapter.initialize(broker_name, user_id)
                adapter.connect()
                
                self.broker_adapters[user_id] = adapter
                logger.info(f"Created ultra-low latency {broker_name} adapter for user {user_id}")
                
            except Exception as e:
                logger.error(f"Failed to create ultra adapter: {e}")
                await self.send_error(client_id, "BROKER_ERROR", str(e))
                return
        
        # Send success response
        await self.send_message(client_id, {
            "type": "auth",
            "status": "success",
            "message": "Ultra-low latency authentication successful",
            "broker": broker_name,
            "user_id": user_id,
            "adapter_type": "ultra_low_latency"
        })

    async def subscribe_client(self, client_id, data):
        """Subscribe client using ultra-low latency adapter"""
        if client_id not in self.user_mapping:
            await self.send_error(client_id, "NOT_AUTHENTICATED", "Authentication required")
            return
        
        # Get subscription parameters
        symbols = data.get("symbols") or []
        mode_str = data.get("mode", "Quote")
        
        # Handle single symbol format
        if not symbols and (data.get("symbol") and data.get("exchange")):
            symbols = [{"symbol": data.get("symbol"), "exchange": data.get("exchange")}]
        
        if not symbols:
            await self.send_error(client_id, "INVALID_PARAMETERS", "Symbols required")
            return
        
        # Map string mode to numeric
        mode_mapping = {"LTP": 1, "Quote": 2, "Depth": 3}
        mode = mode_mapping.get(mode_str, mode_str) if isinstance(mode_str, str) else mode_str
        
        # Get adapter
        user_id = self.user_mapping[client_id]
        if user_id not in self.broker_adapters:
            await self.send_error(client_id, "BROKER_ERROR", "Adapter not found")
            return
        
        adapter = self.broker_adapters[user_id]
        broker_name = self.user_broker_mapping.get(user_id, "unknown")
        
        # Process subscriptions
        subscription_responses = []
        
        for symbol_info in symbols:
            symbol = symbol_info.get("symbol")
            exchange = symbol_info.get("exchange")
            
            if not symbol or not exchange:
                continue
            
            # Subscribe using ultra adapter
            response = adapter.subscribe(symbol, exchange, mode)
            
            if response.get("status") == "success":
                # Store subscription
                subscription_info = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "mode": mode,
                    "broker": broker_name
                }
                
                self.subscriptions[client_id].add(json.dumps(subscription_info))
                
                subscription_responses.append({
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "success",
                    "mode": mode_str,
                    "broker": broker_name
                })
            else:
                subscription_responses.append({
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "error",
                    "message": response.get("message", "Subscription failed"),
                    "broker": broker_name
                })
        
        await self.send_message(client_id, {
            "type": "subscribe",
            "status": "success",
            "subscriptions": subscription_responses,
            "broker": broker_name,
            "adapter_type": "ultra_low_latency"
        })

    async def unsubscribe_client(self, client_id, data):
        """Unsubscribe client using ultra-low latency adapter"""
        if client_id not in self.user_mapping:
            await self.send_error(client_id, "NOT_AUTHENTICATED", "Authentication required")
            return
        
        # Get unsubscription parameters
        symbol = data.get("symbol")
        exchange = data.get("exchange")
        mode = data.get("mode", 1)
        
        if not symbol or not exchange:
            await self.send_error(client_id, "INVALID_PARAMETERS", "Symbol and exchange required")
            return
        
        # Get adapter
        user_id = self.user_mapping[client_id]
        if user_id not in self.broker_adapters:
            await self.send_error(client_id, "BROKER_ERROR", "Adapter not found")
            return
        
        adapter = self.broker_adapters[user_id]
        broker_name = self.user_broker_mapping.get(user_id, "unknown")
        
        # Unsubscribe using ultra adapter
        response = adapter.unsubscribe(symbol, exchange, mode)
        
        if response.get("status") == "success":
            # Remove from local subscriptions
            subscription_info = {
                "symbol": symbol,
                "exchange": exchange,
                "mode": mode,
                "broker": broker_name
            }
            subscription_json = json.dumps(subscription_info)
            self.subscriptions[client_id].discard(subscription_json)
            
            await self.send_message(client_id, {
                "type": "unsubscribe",
                "status": "success",
                "symbol": symbol,
                "exchange": exchange,
                "mode": mode,
                "broker": broker_name,
                "adapter_type": "ultra_low_latency"
            })
        else:
            await self.send_error(client_id, "UNSUBSCRIBE_ERROR", response.get("message", "Unsubscription failed"))

    async def get_performance_stats(self, client_id):
        """Get ultra-low latency performance statistics"""
        try:
            # Get adapter stats
            adapter_stats = {}
            for user_id, adapter in self.broker_adapters.items():
                if hasattr(adapter, 'get_performance_stats'):
                    adapter_stats[user_id] = adapter.get_performance_stats()
            
            # Server stats
            server_stats = {
                'server_type': 'UltraLowLatency',
                'message_count': self.message_count.load(),
                'client_count': self.client_count.load(),
                'running': self.running,
                'ring_buffer_connected': self.ring_buffer is not None
            }
            
            if self.ring_buffer:
                server_stats['ring_buffer_stats'] = self.ring_buffer.get_stats()
            
            await self.send_message(client_id, {
                "type": "performance_stats",
                "status": "success",
                "server_stats": server_stats,
                "adapter_stats": adapter_stats
            })
            
        except Exception as e:
            await self.send_error(client_id, "STATS_ERROR", str(e))

    async def send_message(self, client_id, message):
        """Send message to client"""
        if client_id in self.clients:
            try:
                await self.clients[client_id].send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                logger.debug(f"Connection closed while sending to client {client_id}")

    async def send_error(self, client_id, code, message):
        """Send error message to client"""
        await self.send_message(client_id, {
            "status": "error",
            "code": code,
            "message": message
        })

    async def stop(self):
        """Stop the server and clean up resources"""
        logger.info("Stopping Ultra WebSocket server...")
        self.running = False
        
        try:
            # Close server
            if hasattr(self, 'server') and self.server:
                self.server.close()
                await self.server.wait_closed()
            
            # Close all clients
            close_tasks = []
            for client_id, websocket in self.clients.items():
                try:
                    close_tasks.append(websocket.close())
                except:
                    pass
            
            if close_tasks:
                await aio.gather(*close_tasks, return_exceptions=True)
            
            # Disconnect adapters
            for user_id, adapter in self.broker_adapters.items():
                try:
                    adapter.disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting adapter for user {user_id}: {e}")
            
            logger.info("Ultra WebSocket server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping server: {e}")

# Entry point
async def ultra_main():
    """Main entry point for ultra-low latency server"""
    proxy = None
    try:
        load_dotenv()
        
        ws_host = os.getenv('WEBSOCKET_HOST', '127.0.0.1')
        ws_port = int(os.getenv('WEBSOCKET_PORT', '8765'))
        
        proxy = UltraWebSocketProxy(host=ws_host, port=ws_port)
        await proxy.start()
        
    except KeyboardInterrupt:
        logger.info("Ultra server stopped by user")
    except Exception as e:
        logger.error(f"Ultra server error: {e}")
        raise
    finally:
        if proxy:
            try:
                await proxy.stop()
            except Exception as e:
                logger.error(f"Error during ultra server cleanup: {e}")

if __name__ == "__main__":
    aio.run(ultra_main())