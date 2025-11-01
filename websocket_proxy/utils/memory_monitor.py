"""
Memory Monitor for WebSocket Proxy
Tracks memory usage and performs automatic cleanup to prevent memory leaks
"""

import gc
import os
import psutil
import threading
import time
from typing import Dict, Any, Optional
from utils.logging import get_logger

logger = get_logger(__name__)


class MemoryMonitor:
    """
    Monitors memory usage and performs automatic cleanup
    """
    
    def __init__(self, 
                 max_memory_mb: int = 2048,
                 warning_threshold_mb: int = 1536,
                 check_interval: int = 30,
                 cleanup_threshold_mb: int = 1792):
        """
        Initialize memory monitor
        
        Args:
            max_memory_mb: Maximum memory before forced cleanup (MB)
            warning_threshold_mb: Memory level to start warnings (MB)
            check_interval: How often to check memory (seconds)
            cleanup_threshold_mb: Memory level to trigger cleanup (MB)
        """
        self.max_memory_mb = max_memory_mb
        self.warning_threshold_mb = warning_threshold_mb
        self.cleanup_threshold_mb = cleanup_threshold_mb
        self.check_interval = check_interval
        
        self.process = psutil.Process()
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'peak_memory_mb': 0,
            'cleanup_count': 0,
            'warning_count': 0,
            'last_cleanup_time': 0,
            'memory_history': []
        }
        
        # Cleanup callbacks
        self.cleanup_callbacks = []
        
    def add_cleanup_callback(self, callback, name: str):
        """Add a cleanup callback function"""
        self.cleanup_callbacks.append({
            'callback': callback,
            'name': name
        })
        logger.info(f"Added cleanup callback: {name}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Update peak memory
            if memory_mb > self.stats['peak_memory_mb']:
                self.stats['peak_memory_mb'] = memory_mb
            
            return {
                'rss_mb': memory_mb,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': self.process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'available_mb': 0}
    
    def check_memory(self) -> bool:
        """
        Check memory usage and trigger cleanup if needed
        
        Returns:
            bool: True if cleanup was triggered
        """
        memory_info = self.get_memory_usage()
        memory_mb = memory_info['rss_mb']
        
        # Add to history (keep last 100 readings)
        self.stats['memory_history'].append({
            'timestamp': time.time(),
            'memory_mb': memory_mb
        })
        if len(self.stats['memory_history']) > 100:
            self.stats['memory_history'].pop(0)
        
        # Check thresholds
        if memory_mb >= self.max_memory_mb:
            logger.critical(f"CRITICAL: Memory usage {memory_mb:.1f}MB exceeds maximum {self.max_memory_mb}MB - FORCING CLEANUP")
            self.force_cleanup()
            return True
            
        elif memory_mb >= self.cleanup_threshold_mb:
            logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds cleanup threshold {self.cleanup_threshold_mb}MB - triggering cleanup")
            self.trigger_cleanup()
            return True
            
        elif memory_mb >= self.warning_threshold_mb:
            self.stats['warning_count'] += 1
            if self.stats['warning_count'] % 10 == 1:  # Log every 10th warning to avoid spam
                logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds warning threshold {self.warning_threshold_mb}MB")
        
        return False
    
    def trigger_cleanup(self):
        """Trigger gentle cleanup"""
        logger.info("Starting memory cleanup...")
        cleanup_results = []
        
        for callback_info in self.cleanup_callbacks:
            try:
                result = callback_info['callback']()
                cleanup_results.append(f"{callback_info['name']}: {result}")
                logger.debug(f"Cleanup callback {callback_info['name']} completed")
            except Exception as e:
                logger.error(f"Error in cleanup callback {callback_info['name']}: {e}")
                cleanup_results.append(f"{callback_info['name']}: ERROR - {e}")
        
        # Force garbage collection
        collected = gc.collect()
        cleanup_results.append(f"GC collected {collected} objects")
        
        self.stats['cleanup_count'] += 1
        self.stats['last_cleanup_time'] = time.time()
        
        # Check memory after cleanup
        memory_after = self.get_memory_usage()['rss_mb']
        logger.info(f"Memory cleanup completed. Memory usage: {memory_after:.1f}MB. Results: {'; '.join(cleanup_results)}")
    
    def force_cleanup(self):
        """Force aggressive cleanup"""
        logger.critical("FORCING AGGRESSIVE MEMORY CLEANUP")
        
        # Run all cleanup callbacks
        self.trigger_cleanup()
        
        # Multiple GC passes
        for i in range(3):
            collected = gc.collect()
            logger.info(f"Aggressive GC pass {i+1}: collected {collected} objects")
        
        # Check if we're still over limit
        memory_after = self.get_memory_usage()['rss_mb']
        if memory_after >= self.max_memory_mb:
            logger.critical(f"CRITICAL: Memory still at {memory_after:.1f}MB after aggressive cleanup!")
            # Could trigger process restart here if needed
    
    def start_monitoring(self):
        """Start the memory monitoring thread"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Memory monitor started - Max: {self.max_memory_mb}MB, Warning: {self.warning_threshold_mb}MB, Cleanup: {self.cleanup_threshold_mb}MB")
    
    def stop_monitoring(self):
        """Stop the memory monitoring thread"""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Memory monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.check_memory()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in memory monitor loop: {e}")
                time.sleep(self.check_interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory monitor statistics"""
        current_memory = self.get_memory_usage()
        
        return {
            'current_memory': current_memory,
            'thresholds': {
                'max_mb': self.max_memory_mb,
                'warning_mb': self.warning_threshold_mb,
                'cleanup_mb': self.cleanup_threshold_mb
            },
            'stats': self.stats.copy(),
            'status': self._get_status(current_memory['rss_mb'])
        }
    
    def _get_status(self, memory_mb: float) -> str:
        """Get current memory status"""
        if memory_mb >= self.max_memory_mb:
            return "CRITICAL"
        elif memory_mb >= self.cleanup_threshold_mb:
            return "HIGH"
        elif memory_mb >= self.warning_threshold_mb:
            return "WARNING"
        else:
            return "OK"


# Global memory monitor instance
_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance"""
    global _memory_monitor
    if _memory_monitor is None:
        # Get configuration from environment
        max_memory = int(os.getenv('MAX_MEMORY_MB', '2048'))
        warning_threshold = int(os.getenv('WARNING_MEMORY_MB', '1536'))
        cleanup_threshold = int(os.getenv('CLEANUP_MEMORY_MB', '1792'))
        check_interval = int(os.getenv('MEMORY_CHECK_INTERVAL', '30'))
        
        _memory_monitor = MemoryMonitor(
            max_memory_mb=max_memory,
            warning_threshold_mb=warning_threshold,
            cleanup_threshold_mb=cleanup_threshold,
            check_interval=check_interval
        )
    return _memory_monitor


def start_memory_monitoring():
    """Start global memory monitoring"""
    monitor = get_memory_monitor()
    monitor.start_monitoring()


def stop_memory_monitoring():
    """Stop global memory monitoring"""
    global _memory_monitor
    if _memory_monitor:
        _memory_monitor.stop_monitoring()


def get_memory_stats() -> Dict[str, Any]:
    """Get memory statistics"""
    monitor = get_memory_monitor()
    return monitor.get_stats()