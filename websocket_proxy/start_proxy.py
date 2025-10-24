"""
Standalone WebSocket proxy starter for OpenAlgo
"""

import asyncio
import signal
import sys
import platform

from websocket_proxy import start_disruptor_proxy, cleanup_disruptor_server
from utils.logging import get_logger

logger = get_logger(__name__)

# Set correct event loop policy for Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, cleaning up...")
    cleanup_disruptor_server()
    sys.exit(0)

def main():
    """Start the WebSocket proxy server in standalone mode"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting standalone WebSocket proxy server...")
    # We pass None as app since we're running standalone
    start_disruptor_proxy(None)
    
    # Keep the main thread running
    try:
        while True:
            asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        cleanup_disruptor_server()

if __name__ == '__main__':
    main()