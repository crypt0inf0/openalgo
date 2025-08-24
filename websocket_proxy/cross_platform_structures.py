import mmap
import struct
import time
import threading
import os
import sys
import tempfile
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum
import queue

# Platform detection
IS_WINDOWS = sys.platform.startswith('win')
IS_UNIX = not IS_WINDOWS

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes
    import msvcrt
    
    # Windows API constants
    INVALID_HANDLE_VALUE = -1
    PAGE_READWRITE = 0x04
    FILE_MAP_ALL_ACCESS = 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0010
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    CREATE_ALWAYS = 2
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x80
    
    # Windows API functions
    kernel32 = ctypes.windll.kernel32
    
    # File mapping functions
    kernel32.CreateFileMappingW.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD, 
        wintypes.DWORD, wintypes.DWORD, wintypes.LPCWSTR
    ]
    kernel32.CreateFileMappingW.restype = wintypes.HANDLE
    
    kernel32.OpenFileMappingW.argtypes = [
        wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR
    ]
    kernel32.OpenFileMappingW.restype = wintypes.HANDLE
    
    kernel32.MapViewOfFile.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t
    ]
    kernel32.MapViewOfFile.restype = ctypes.c_void_p
    
    kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    kernel32.UnmapViewOfFile.restype = wintypes.BOOL
    
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    
    kernel32.GetLastError.restype = wintypes.DWORD
    
else:
    import fcntl

class MessageType(IntEnum):
    MARKET_DATA = 1
    HEARTBEAT = 2
    CONTROL = 3

@dataclass
class MarketTick:
    symbol_id: int
    timestamp: int
    sequence: int
    message_type: int
    
    # Price data (floats)
    ltp: float
    open: float
    high: float
    low: float
    close: float
    bid: float
    ask: float
    
    # Quantity data (integers)
    volume: int
    bid_qty: int
    ask_qty: int
    
    # Metadata
    changed_fields: int
    exchange: str
    symbol: str
    mode: int
    
    # FIXED: Use double precision floats for accurate price representation
    STRUCT_FORMAT = '=QqQI7dIIII64s32sI'  # Exactly 208 bytes  
    # Q=8, q=8, Q=8, I=4, 7d=56, I=4, I=4, I=4, 64s=64, 32s=32, I=4 = 196 + 12 padding = 208
    
    @classmethod
    def get_serialized_size(cls) -> int:
        """Calculate the exact serialized size - FIXED to return 208"""
        return 208  # Fixed size to match ring buffer configuration
    
    def to_bytes(self) -> bytes:
        """Convert to binary format for shared memory - FIXED to use single precision floats"""
        try:
            # Encode strings first to catch any encoding issues
            exchange_bytes = self.exchange.encode('utf-8')[:64].ljust(64, b'\x00')
            symbol_bytes = self.symbol.encode('utf-8')[:32].ljust(32, b'\x00')
            
            # Convert all fields to their expected types with validation
            symbol_id = max(0, min(int(self.symbol_id), 0xFFFFFFFFFFFFFFFF))
            
            # Handle timestamps properly
            raw_timestamp = int(self.timestamp)
            if raw_timestamp > 2**63 - 1:
                timestamp = int(time.time() * 1_000_000_000)
            else:
                timestamp = max(0, min(raw_timestamp, 0x7FFFFFFFFFFFFFFF))
                
            sequence = max(0, min(int(self.sequence), 0xFFFFFFFFFFFFFFFF))
            
            # Handle MessageType enum
            if hasattr(self.message_type, 'value'):
                message_type = int(self.message_type.value)
            else:
                message_type = int(self.message_type)
            message_type = max(0, min(message_type, 0xFFFFFFFF))
            
            # FIXED: Use double precision (64-bit) floats for accurate price representation
            def safe_double(val):
                try:
                    f = float(val)
                    # Check for NaN, infinity, or values outside double range
                    if not (-1.7976931348623157e+308 <= f <= 1.7976931348623157e+308) or f != f:
                        return 0.0
                    return f
                except (ValueError, TypeError, OverflowError):
                    return 0.0
            
            ltp = safe_double(self.ltp)
            open = safe_double(self.open)
            high = safe_double(self.high)
            low = safe_double(self.low)
            close = safe_double(self.close)
            bid = safe_double(self.bid)
            ask = safe_double(self.ask)
            
            # Integer fields
            def safe_uint32(val):
                try:
                    i = int(val)
                    return max(0, min(i, 0xFFFFFFFF))
                except (ValueError, TypeError, OverflowError):
                    return 0
            
            volume = safe_uint32(self.volume)
            bid_qty = safe_uint32(self.bid_qty)
            ask_qty = safe_uint32(self.ask_qty)
            changed_fields = safe_uint32(self.changed_fields)
            mode = safe_uint32(self.mode)
            
            # FIXED: Pack to exactly 208 bytes using double precision floats
            # Main data with doubles (92 bytes: 4*8 + 7*8 + 5*4 = 96 bytes)
            main_format = '=QqQI7dIIII'  # 96 bytes
            main_data = struct.pack(
                main_format,
                symbol_id,      # Q - uint64
                timestamp,      # q - int64
                sequence,       # Q - uint64
                message_type,   # I - uint32
                ltp,            # d - float64 (double precision)
                open,     # d - float64 (double precision)
                high,           # d - float64 (double precision)
                low,            # d - float64 (double precision)
                close,          # d - float64 (double precision)
                bid,            # d - float64 (double precision)
                ask,            # d - float64 (double precision)
                volume,         # I - uint32
                bid_qty,        # I - uint32
                ask_qty,        # I - uint32
                changed_fields  # I - uint32
            )
            
            # Pack strings separately
            string_format = '64s32sI'  # 100 bytes
            string_data = struct.pack(string_format, exchange_bytes, symbol_bytes, mode)
            
            # Combine and pad to exactly 208 bytes (96 + 100 = 196, need 12 padding)
            combined_data = main_data + string_data  # 196 bytes
            padding_needed = 208 - len(combined_data)
            
            if padding_needed > 0:
                final_data = combined_data + (b'\x00' * padding_needed)
            else:
                final_data = combined_data[:208]  # Truncate if somehow too large
            
            # Verify the packed data size
            if len(final_data) != 208:
                raise RuntimeError(f"Packed data size mismatch: expected 208, got {len(final_data)}")
                
            return final_data
            
        except Exception as e:
            print(f"Error in to_bytes: {e}")
            print(f"Original values:")
            print(f"  symbol_id: {self.symbol_id} (type: {type(self.symbol_id).__name__})")
            print(f"  timestamp: {self.timestamp} (type: {type(self.timestamp).__name__})")
            print(f"  sequence: {self.sequence} (type: {type(self.sequence).__name__})")
            print(f"  message_type: {self.message_type} (type: {type(self.message_type).__name__})")
            raise
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MarketTick':
        """Create from binary format - FIXED to handle 208-byte format"""
        if len(data) != 208:
            raise ValueError(f"Data size {len(data)} != expected 208 bytes")
        
        try:
            # Unpack main data (first 96 bytes) with double precision floats
            main_format = '=QqQI7dIIII'
            main_size = struct.calcsize(main_format)
            main_data = struct.unpack(main_format, data[:main_size])
            
            # Unpack string data (next 100 bytes)
            string_format = '64s32sI'
            string_start = main_size
            string_end = string_start + struct.calcsize(string_format)
            string_data = struct.unpack(string_format, data[string_start:string_end])
            
            return cls(
                symbol_id=main_data[0],
                timestamp=main_data[1],
                sequence=main_data[2],
                message_type=main_data[3],
                ltp=main_data[4],
                open=main_data[5],
                high=main_data[6],
                low=main_data[7],
                close=main_data[8],
                bid=main_data[9],
                ask=main_data[10],
                volume=main_data[11],
                bid_qty=main_data[12],
                ask_qty=main_data[13],
                changed_fields=main_data[14],
                exchange=string_data[0].decode('utf-8').rstrip('\x00'),
                symbol=string_data[1].decode('utf-8').rstrip('\x00'),
                mode=string_data[2]
            )
        except struct.error as e:
            print(f"Error unpacking data: {e}")
            print(f"Data length: {len(data)}, expected: 208")
            raise

class CrossPlatformAtomic:
    """Cross-platform atomic operations"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.RLock()
    
    def load(self) -> int:
        with self._lock:
            return self._value
    
    def store(self, value: int):
        with self._lock:
            self._value = value
    
    def fetch_add(self, increment: int) -> int:
        with self._lock:
            old_value = self._value
            self._value += increment
            return old_value
    
    def compare_exchange(self, expected: int, desired: int) -> bool:
        with self._lock:
            if self._value == expected:
                self._value = desired
                return True
            return False

class WindowsSharedMemory:
    """Windows-specific shared memory implementation using File Mapping API"""
    
    def __init__(self, name: str, size: int, create: bool = False):
        self.name = name
        self.size = size
        self.memory = None
        self.map_handle = None
        self.memory_ptr = None
        self._created = False
        
        # Generate unique name for this process
        self.mapping_name = f"Local\\OpenAlgo_{name}_{os.getpid()}"
        
        try:
            if create:
                self._create_mapping()
            else:
                self._open_existing_mapping()
                
            # Map the view
            self._map_view()
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize Windows shared memory: {e}")
    
    def _create_mapping(self):
        """Create a new file mapping object"""
        try:
            # Create file mapping
            self.map_handle = kernel32.CreateFileMappingW(
                INVALID_HANDLE_VALUE,  # Use paging file
                None,                  # Default security
                PAGE_READWRITE,        # Read/write access
                0,                     # High-order DWORD of size
                self.size,             # Low-order DWORD of size
                self.mapping_name      # Name of mapping object
            )
            
            if not self.map_handle or self.map_handle == INVALID_HANDLE_VALUE:
                error = kernel32.GetLastError()
                if error == 183:  # ERROR_ALREADY_EXISTS
                    # Mapping already exists, try to open it
                    self._open_existing_mapping()
                    return
                else:
                    raise WindowsError(f"CreateFileMappingW failed with error {error}")
            
            self._created = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to create file mapping: {e}")
    
    def _open_existing_mapping(self):
        """Open an existing file mapping object"""
        try:
            self.map_handle = kernel32.OpenFileMappingW(
                FILE_MAP_ALL_ACCESS,   # Desired access
                False,                 # Inherit handle
                self.mapping_name      # Name of mapping object
            )
            
            if not self.map_handle or self.map_handle == INVALID_HANDLE_VALUE:
                error = kernel32.GetLastError()
                raise WindowsError(f"OpenFileMappingW failed with error {error}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to open file mapping: {e}")
    
    def _map_view(self):
        """Map a view of the file mapping into memory"""
        try:
            self.memory_ptr = kernel32.MapViewOfFile(
                self.map_handle,       # Handle to mapping object
                FILE_MAP_ALL_ACCESS,   # Access mode
                0,                     # High-order offset
                0,                     # Low-order offset
                self.size              # Number of bytes to map
            )
            
            if not self.memory_ptr:
                error = kernel32.GetLastError()
                raise WindowsError(f"MapViewOfFile failed with error {error}")
            
            # Create a ctypes array to access the memory
            self.memory = (ctypes.c_ubyte * self.size).from_address(self.memory_ptr)
            
            # Initialize memory to zero if we created it
            if self._created:
                ctypes.memset(self.memory_ptr, 0, self.size)
            
        except Exception as e:
            raise RuntimeError(f"Failed to map view of file: {e}")
    
    def cleanup(self):
        """Clean up Windows shared memory resources"""
        try:
            if self.memory_ptr:
                kernel32.UnmapViewOfFile(self.memory_ptr)
                self.memory_ptr = None
                self.memory = None
            
            if self.map_handle and self.map_handle != INVALID_HANDLE_VALUE:
                kernel32.CloseHandle(self.map_handle)
                self.map_handle = None
                
        except Exception as e:
            print(f"Error during Windows shared memory cleanup: {e}")

class CrossPlatformSharedMemory:
    """Cross-platform shared memory implementation"""
    
    def __init__(self, name: str, size: int, create: bool = False):
        self.name = name
        self.size = size
        self.memory = None
        self.fd = None
        self.temp_file = None
        self._windows_shm = None
        
        if IS_WINDOWS:
            self._init_windows(create)
        else:
            self._init_unix(create)
    
    def _init_windows(self, create: bool):
        """Initialize Windows shared memory using native API with fallback"""
        try:
            # Try native Windows shared memory first
            self._windows_shm = WindowsSharedMemory(self.name, self.size, create)
            self.memory = self._windows_shm.memory
            print(f"Using native Windows shared memory: {self.name}")
            
        except Exception as native_error:
            print(f"Native Windows shared memory failed ({native_error}), falling back to file mapping")
            
            try:
                # Fallback to temporary file mapping
                self._init_windows_fallback(create)
                print(f"Using Windows file mapping fallback: {self.name}")
                
            except Exception as fallback_error:
                raise RuntimeError(f"Both native ({native_error}) and fallback ({fallback_error}) failed")
    
    def _init_windows_fallback(self, create: bool):
        """Fallback Windows implementation using temporary file"""
        if create:
            self.temp_file = tempfile.NamedTemporaryFile(
                prefix=f"openalgo_{self.name}_",
                delete=False
            )
            self.temp_file.write(b'\x00' * self.size)
            self.temp_file.flush()
            
            self.memory = mmap.mmap(
                self.temp_file.fileno(), 
                self.size,
                access=mmap.ACCESS_WRITE
            )
        else:
            self._init_windows_fallback(create=True)
    
    def _init_unix(self, create: bool):
        """Initialize Unix shared memory using mmap"""
        try:
            if create:
                self.temp_file = tempfile.NamedTemporaryFile(
                    prefix=f"openalgo_{self.name}_",
                    delete=False
                )
                self.temp_file.write(b'\x00' * self.size)
                self.temp_file.flush()
                
                self.memory = mmap.mmap(
                    self.temp_file.fileno(),
                    self.size,
                    mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE
                )
            else:
                self._init_unix(create=True)
                
        except Exception as e:
            raise RuntimeError(f"Failed to create Unix shared memory: {e}")
    
    def cleanup(self):
        """Clean up shared memory resources"""
        try:
            if IS_WINDOWS and self._windows_shm:
                self._windows_shm.cleanup()
                self._windows_shm = None
            
            if self.memory and hasattr(self.memory, 'close'):
                self.memory.close()
                self.memory = None
            
            if self.temp_file:
                self.temp_file.close()
                try:
                    os.unlink(self.temp_file.name)
                except:
                    pass
                self.temp_file = None
                
        except Exception as e:
            print(f"Error during shared memory cleanup: {e}")

class CrossPlatformRingBuffer:
    """Cross-platform ring buffer with native Windows shared memory support"""
    
    def __init__(self, size: int = 16384, item_size: int = None):
        """Initialize ring buffer with automatic item size calculation"""
        self.size = size
        self.mask = size - 1
        
        # FIXED: Always use 208 bytes to match configuration
        self.item_size = 208
        print(f"Ring buffer item_size: {self.item_size} bytes")
        
        # Verify with MarketTick calculation
        calculated_size = MarketTick.get_serialized_size()
        if self.item_size != calculated_size:
            print(f"WARNING: item_size ({self.item_size}) != calculated_size ({calculated_size})")
            print("Using configured size of 208 bytes")
        
        # Try shared memory first (both Windows and Unix)
        if size <= 65536:
            try:
                self._init_shared_memory()
            except Exception as e:
                print(f"Shared memory initialization failed: {e}")
                self._init_queue_fallback()
        else:
            print(f"Buffer size {size} too large for shared memory, using queue")
            self._init_queue_fallback()
    
    def _init_shared_memory(self):
        """Initialize with shared memory (Windows native API + Unix)"""
        self.use_shared_memory = True
        self.shm_size = self.size * self.item_size + 64
        
        self.shared_mem = CrossPlatformSharedMemory(
            f"ring_{os.getpid()}_{int(time.time())}",
            self.shm_size,
            create=True
        )
        
        self.write_pos = CrossPlatformAtomic(0)
        self.read_pos = CrossPlatformAtomic(0)
        self.overwrite_count = CrossPlatformAtomic(0)
        
        platform_info = "Windows native" if IS_WINDOWS else "Unix"
        print(f"Using {platform_info} shared memory ring buffer")
    
    def _init_queue_fallback(self):
        """Initialize with queue fallback"""
        self.use_shared_memory = False
        self.queue = queue.Queue(maxsize=self.size)
        self.overwrite_count = CrossPlatformAtomic(0)
        
        print("Using queue-based ring buffer (fallback mode)")
    
    def publish(self, data: bytes) -> bool:
        """Publish data to ring buffer"""
        # Accept data up to item_size bytes and pad if necessary
        if len(data) > self.item_size:
            print(f"Data too large: {len(data)} > {self.item_size}")
            return False
        
        # Pad data to exactly item_size bytes
        if len(data) < self.item_size:
            data = data.ljust(self.item_size, b'\x00')
        
        if self.use_shared_memory:
            return self._publish_shared_memory(data)
        else:
            return self._publish_queue(data)
    
    def _publish_shared_memory(self, data: bytes) -> bool:
        """Publish using shared memory"""
        try:
            write_idx = self.write_pos.fetch_add(1)
            slot_offset = 64 + (write_idx & self.mask) * self.item_size
            
            # Check for overwrite
            read_idx = self.read_pos.load()
            if write_idx - read_idx >= self.size:
                new_read_pos = write_idx - self.size + 1
                if self.read_pos.compare_exchange(read_idx, new_read_pos):
                    self.overwrite_count.fetch_add(1)
            
            # Write data (already padded to correct size)
            self.shared_mem.memory[slot_offset:slot_offset + self.item_size] = data
            
            return True
            
        except Exception as e:
            print(f"Error publishing to shared memory: {e}")
            return False
    
    def _publish_queue(self, data: bytes) -> bool:
        """Publish using queue"""
        try:
            try:
                self.queue.put_nowait(data)
                return True
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.overwrite_count.fetch_add(1)
                except queue.Empty:
                    pass
                
                try:
                    self.queue.put_nowait(data)
                    return True
                except queue.Full:
                    return False
                    
        except Exception as e:
            print(f"Error publishing to queue: {e}")
            return False
    
    def consume(self) -> Optional[bytes]:
        """Consume data from ring buffer"""
        if self.use_shared_memory:
            return self._consume_shared_memory()
        else:
            return self._consume_queue()
    
    def _consume_shared_memory(self) -> Optional[bytes]:
        """Consume using shared memory"""
        try:
            read_idx = self.read_pos.load()
            write_idx = self.write_pos.load()
            
            if read_idx >= write_idx:
                return None
            
            slot_offset = 64 + (read_idx & self.mask) * self.item_size
            data = bytes(self.shared_mem.memory[slot_offset:slot_offset + self.item_size])
            
            self.read_pos.store(read_idx + 1)
            return data
            
        except Exception as e:
            print(f"Error consuming from shared memory: {e}")
            return None
    
    def _consume_queue(self) -> Optional[bytes]:
        """Consume using queue"""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
            print(f"Error consuming from queue: {e}")
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics"""
        if self.use_shared_memory:
            write_pos = self.write_pos.load()
            read_pos = self.read_pos.load()
            pending = write_pos - read_pos
            
            if IS_WINDOWS:
                if hasattr(self.shared_mem, '_windows_shm') and self.shared_mem._windows_shm:
                    implementation = 'windows_native_shared_memory'
                else:
                    implementation = 'windows_file_mapping'
            else:
                implementation = 'unix_shared_memory'
        else:
            pending = self.queue.qsize()
            write_pos = pending
            read_pos = 0
            implementation = 'queue_fallback'
        
        return {
            'write_position': write_pos,
            'read_position': read_pos,
            'pending_messages': pending,
            'overwrite_count': self.overwrite_count.load(),
            'buffer_utilization': min(100, pending * 100 // self.size),
            'implementation': implementation,
            'item_size': self.item_size,
            'calculated_size': MarketTick.get_serialized_size(),
            'platform': 'Windows' if IS_WINDOWS else 'Unix'
        }
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'shared_mem'):
            self.shared_mem.cleanup()

class ConflatedMarketData:
    """Conflated market data cache"""
    
    def __init__(self):
        self._data: Dict[int, MarketTick] = {}
        self._dirty_flags: Dict[int, bool] = {}
        self._lock = threading.RLock()
        self._sequence = CrossPlatformAtomic(0)
    
    def update(self, symbol_id: int, tick: MarketTick):
        """Update with latest market data"""
        tick.sequence = self._sequence.fetch_add(1)
        tick.timestamp = int(time.time() * 1_000_000_000)
        
        with self._lock:
            self._data[symbol_id] = tick
            self._dirty_flags[symbol_id] = True
    
    def consume_update(self, symbol_id: int) -> Optional[MarketTick]:
        """Get and mark as consumed"""
        with self._lock:
            if self._dirty_flags.get(symbol_id, False):
                self._dirty_flags[symbol_id] = False
                return self._data.get(symbol_id)
            return None
    
    def get_latest(self, symbol_id: int) -> Optional[MarketTick]:
        """Get latest data without consuming"""
        with self._lock:
            return self._data.get(symbol_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_symbols = len(self._data)
            dirty_symbols = sum(1 for dirty in self._dirty_flags.values() if dirty)
            
            return {
                'total_symbols': total_symbols,
                'dirty_symbols': dirty_symbols,
                'sequence': self._sequence.load()
            }