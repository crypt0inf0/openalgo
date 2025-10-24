import socket
import time
from utils.logging import get_logger

logger = get_logger(__name__)

def is_port_in_use(host, port, wait_time=0):
    """
    Check if a port is in use
    
    Args:
        host: Host to check
        port: Port to check
        wait_time: Time to wait before checking (for cleanup)
        
    Returns:
        bool: True if port is in use
    """
    if wait_time > 0:
        time.sleep(wait_time)
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(1.0)
            s.bind((host, port))
            return False
    except socket.error:
        return True

def find_available_port(start_port=8899, max_attempts=10):
    """
    Find an available port starting from start_port
    
    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try
        
    Returns:
        int: Available port number, or None if none found
    """
    for i in range(max_attempts):
        port = start_port + i
        if not is_port_in_use("127.0.0.1", port):
            return port
    
    logger.error(f"Could not find available port after {max_attempts} attempts starting from {start_port}")
    return None