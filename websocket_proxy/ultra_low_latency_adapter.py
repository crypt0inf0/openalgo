import threading
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from utils.logging import get_logger

from .cross_platform_structures import (
    CrossPlatformRingBuffer,
    ConflatedMarketData,
    MarketTick,
    MessageType,
    CrossPlatformAtomic
)

logger = get_logger(__name__)

class UltraLowLatencyAdapter(ABC):
    """
    Cross-platform ultra-low latency adapter
    Uses shared memory on Unix, queue-based on Windows
    """
    
    # Shared ring buffer (singleton)
    _shared_ring_buffer: Optional[CrossPlatformRingBuffer] = None
    _buffer_lock = threading.Lock()
    
    def __init__(self):
        self.logger = get_logger("ultra_low_latency_adapter")
        self.logger.info("Initializing Ultra-Low Latency Adapter (Cross-Platform)")
        
        # Initialize shared ring buffer
        with self._buffer_lock:
            if UltraLowLatencyAdapter._shared_ring_buffer is None:
                buffer_size = int(os.getenv('RING_BUFFER_SIZE', '16384'))  # Reduced default size
                
                # FIXED: Use automatic item size calculation
                calculated_item_size = MarketTick.get_serialized_size()
                env_item_size = int(os.getenv('MARKET_TICK_SIZE', str(calculated_item_size)))
                
                # Ensure we use the larger of calculated or environment setting
                item_size = max(calculated_item_size, env_item_size)
                
                self.logger.info(f"Ring buffer configuration:")
                self.logger.info(f"  - Calculated item size: {calculated_item_size} bytes")
                self.logger.info(f"  - Environment item size: {env_item_size} bytes") 
                self.logger.info(f"  - Using item size: {item_size} bytes")
                self.logger.info(f"  - Buffer size: {buffer_size} entries")
                self.logger.info(f"  - Total memory: {buffer_size * item_size + 64} bytes")
                
                try:
                    UltraLowLatencyAdapter._shared_ring_buffer = CrossPlatformRingBuffer(
                        size=buffer_size,
                        item_size=item_size  # Use calculated size
                    )
                    self.logger.info(f"Created cross-platform ring buffer: size={buffer_size}, item_size={item_size}")
                except Exception as e:
                    self.logger.error(f"Failed to create ring buffer: {e}")
                    # Create a minimal fallback with correct size
                    UltraLowLatencyAdapter._shared_ring_buffer = CrossPlatformRingBuffer(
                        size=1024, 
                        item_size=calculated_item_size
                    )
        
        # Instance variables
        self.conflated_cache = ConflatedMarketData()
        self.connected = False
        self.running = False
        self.subscriptions = {}
        self.token_to_symbol = {}
        
        # Performance counters
        self.message_count = CrossPlatformAtomic(0)
        self.last_publish_time = CrossPlatformAtomic(0)
        
        # Symbol mapping
        self._symbol_id_counter = CrossPlatformAtomic(1)
        self._symbol_to_id: Dict[str, int] = {}
        self._id_to_symbol: Dict[int, str] = {}
        self._symbol_lock = threading.RLock()
        
        self.logger.info("Ultra-Low Latency Adapter initialized successfully")
    
    def _get_symbol_id(self, symbol: str, exchange: str) -> int:
        """Get or create symbol ID"""
        key = f"{exchange}:{symbol}"
        
        with self._symbol_lock:
            if key in self._symbol_to_id:
                return self._symbol_to_id[key]
            
            symbol_id = self._symbol_id_counter.fetch_add(1)
            self._symbol_to_id[key] = symbol_id
            self._id_to_symbol[symbol_id] = key
            
            return symbol_id
    
    def publish_market_data_direct(self, symbol: str, exchange: str, mode: int, data: Dict[str, Any]):
        """Publish market data directly to the ring buffer"""
        try:
            self.logger.debug("="*50)
            self.logger.debug("publish_market_data_direct called with:")
            self.logger.debug(f"symbol: {symbol}")
            self.logger.debug(f"exchange: {exchange}")
            self.logger.debug(f"mode: {mode}")
            self.logger.debug(f"data keys: {list(data.keys())}")
            self.logger.debug(f"data values: {data}")
            
            # Create a MarketTick object
            tick = MarketTick(
                symbol_id=0,  # Will be set based on symbol
                timestamp=time.time_ns(),
                sequence=self.message_count.fetch_add(1),
                message_type=MessageType.MARKET_DATA,
                ltp=float(data.get('ltp', 0.0)),
                open_price=float(data.get('open', 0.0)),
                high=float(data.get('high', 0.0)),
                low=float(data.get('low', 0.0)),
                close=float(data.get('close', 0.0)),
                bid=float(data.get('bid', 0.0)),
                ask=float(data.get('ask', 0.0)),
                volume=int(data.get('volume', 0)),
                bid_qty=int(data.get('bid_qty', 0)),
                ask_qty=int(data.get('ask_qty', 0)),
                changed_fields=int(data.get('changed_fields', 0)),
                exchange=str(data.get('exchange', '')),
                symbol=str(data.get('symbol', '')),
                mode=int(data.get('mode', 0))
            )
            
            self.logger.debug(f"Created MarketTick: {tick}")
            self.logger.debug(f"MarketTick fields count: {len(tick.__dataclass_fields__)}")
            
            # Publish to ring buffer
            self.ring_buffer.write(tick)
            
        except Exception as e:
            self.logger.error(f"Error in publish_market_data_direct: {str(e)}", exc_info=True)
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        ring_stats = UltraLowLatencyAdapter._shared_ring_buffer.get_stats()
        conflated_stats = self.conflated_cache.get_stats()
        
        return {
            'adapter_type': f'UltraLowLatency-{ring_stats.get("implementation", "unknown")}',
            'message_count': self.message_count.load(),
            'last_publish_time': self.last_publish_time.load(),
            'connected': self.connected,
            'running': self.running,
            'ring_buffer': ring_stats,
            'conflated_cache': conflated_stats,
            'subscriptions': len(self.subscriptions),
            'symbol_mappings': len(self._symbol_to_id)
        }
    
    def cleanup_shared_resources(self):
        """Clean up shared resources"""
        with self._buffer_lock:
            if UltraLowLatencyAdapter._shared_ring_buffer:
                try:
                    UltraLowLatencyAdapter._shared_ring_buffer.cleanup()
                    UltraLowLatencyAdapter._shared_ring_buffer = None
                    self.logger.info("Cleaned up ring buffer")
                except Exception as e:
                    self.logger.error(f"Error cleaning up: {e}")
    
    # Abstract methods remain the same
    @abstractmethod
    def initialize(self, broker_name: str, user_id: str, auth_data: Optional[Dict[str, str]] = None):
        pass
    
    @abstractmethod
    def subscribe(self, symbol: str, exchange: str, mode: int = 2, depth_level: int = 5):
        pass
    
    @abstractmethod
    def unsubscribe(self, symbol: str, exchange: str, mode: int = 2):
        pass
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass
    
    def _create_success_response(self, message: str, **kwargs) -> Dict[str, Any]:
        response = {'status': 'success', 'message': message}
        response.update(kwargs)
        return response
    
    def _create_error_response(self, code: str, message: str) -> Dict[str, Any]:
        return {'status': 'error', 'code': code, 'message': message}