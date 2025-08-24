import asyncio
import threading
import sys
import platform
import os
import signal
import atexit

from .ultra_server import ultra_main
from utils.logging import get_logger, highlight_url

# Set event loop policy for Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Global variables
_ultra_server_started = False
_ultra_proxy_instance = None
_ultra_thread = None

logger = get_logger(__name__)

def should_start_ultra_websocket():
    """Determine if current process should start ultra WebSocket server"""
    if os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true'):
        return os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    return True

def cleanup_ultra_server():
    """Clean up ultra WebSocket server resources"""
    global _ultra_proxy_instance, _ultra_thread
    
    try:
        logger.info("Cleaning up Ultra WebSocket server...")
        
        if _ultra_proxy_instance:
            _ultra_proxy_instance.running = False
            
            # Clean up shared memory resources
            try:
                from .ultra_low_latency_adapter import UltraLowLatencyAdapter
                if hasattr(UltraLowLatencyAdapter, '_shared_ring_buffer'):
                    adapter = UltraLowLatencyAdapter()
                    adapter.cleanup_shared_resources()
            except Exception as e:
                logger.warning(f"Error cleaning up shared resources: {e}")
        
        if _ultra_thread and _ultra_thread.is_alive():
            logger.info("Waiting for ultra WebSocket thread to finish...")
            _ultra_thread.join(timeout=3.0)
            
        logger.info("Ultra WebSocket server cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during ultra WebSocket cleanup: {e}")
    finally:
        _ultra_proxy_instance = None
        _ultra_thread = None

def ultra_signal_handler(signum, frame):
    """Handle signals for ultra server"""
    logger.info(f"Ultra server received signal {signum}, shutting down...")
    cleanup_ultra_server()
    os._exit(0)

def start_ultra_websocket_server():
    """Start ultra-low latency WebSocket server"""
    global _ultra_proxy_instance, _ultra_thread
    
    logger.info("Starting Ultra-Low Latency WebSocket server in separate thread")
    
    def run_ultra_server():
        """Run ultra WebSocket server"""
        global _ultra_proxy_instance
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Set environment variables for ultra mode
            os.environ['ULTRA_LOW_LATENCY_MODE'] = 'true'
            os.environ['RING_BUFFER_SIZE'] = os.getenv('RING_BUFFER_SIZE', '65536')
            os.environ['MARKET_TICK_SIZE'] = os.getenv('MARKET_TICK_SIZE', '256')
            
            # Run ultra server
            loop.run_until_complete(ultra_main())
            
        except Exception as e:
            logger.exception(f"Error in ultra WebSocket server thread: {e}")
        finally:
            _ultra_proxy_instance = None
    
    # Start ultra server thread
    _ultra_thread = threading.Thread(target=run_ultra_server, daemon=False)
    _ultra_thread.start()
    
    # Register cleanup handlers
    atexit.register(cleanup_ultra_server)
    
    try:
        signal.signal(signal.SIGINT, ultra_signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, ultra_signal_handler)
        logger.info("Ultra signal handlers registered")
    except Exception as e:
        logger.warning(f"Could not register ultra signal handlers: {e}")
    
    logger.info("Ultra-Low Latency WebSocket server thread started")
    return _ultra_thread

def start_ultra_websocket_proxy(app):
    """Integrate ultra WebSocket proxy with Flask app"""
    global _ultra_server_started
    
    if should_start_ultra_websocket():
        if not _ultra_server_started:
            _ultra_server_started = True
            logger.info("Starting Ultra WebSocket server in Flask application")
            start_ultra_websocket_server()
            logger.info("Ultra WebSocket server integration complete")
        else:
            logger.info("Ultra WebSocket server already running")
    else:
        logger.info("Skipping Ultra WebSocket server in parent process")
