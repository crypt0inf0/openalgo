"""
Utilities Module

This module contains utility functions for the WebSocket proxy.
"""

from .port_check import find_available_port, is_port_in_use

__all__ = ['find_available_port', 'is_port_in_use']