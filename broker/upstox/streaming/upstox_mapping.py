"""
Mapping utilities for Upstox broker WebSocket integration
"""
from typing import Dict, Any, List, Optional

class UpstoxExchangeMapper:
    """Map OpenAlgo exchange codes to Upstox format and vice versa"""
    OA_TO_UPSTOX: Dict[str, str] = {
        'NSE': 'NSE_EQ',
        'BSE': 'BSE_EQ',
        'NFO': 'NSE_FO',
        'BFO': 'BSE_FO',
        'CDS': 'NSE_CURRENCY',
        'BCD': 'BSE_CURRENCY',
        'MCX': 'MCX_FO',
        'NSE_INDEX': 'NSE_INDEX',
        'BSE_INDEX': 'BSE_INDEX',
    }

    UPSTOX_TO_OA: Dict[str, str] = {v: k for k, v in OA_TO_UPSTOX.items()}

    @classmethod
    def to_upstox_exchange(cls, oa_exchange: str) -> str:
        return cls.OA_TO_UPSTOX.get(oa_exchange.upper(), oa_exchange.upper())

    @classmethod
    def to_oa_exchange(cls, upstox_exchange: str) -> str:
        return cls.UPSTOX_TO_OA.get(upstox_exchange.upper(), upstox_exchange.upper())

class UpstoxCapabilityRegistry:
    """Registry of Upstox WebSocket capabilities"""
    MODES = ['ltpc', 'option_greeks', 'full', 'full_d30']
    MAX_SUBSCRIPTIONS = {
        'ltpc': 5000,
        'option_greeks': 3000,
        'full': 2000,
        'full_d30': 50
    }
    @classmethod
    def is_mode_supported(cls, mode: str) -> bool:
        return mode in cls.MODES

    @classmethod
    def get_max_subscriptions(cls, mode: str) -> int:
        return cls.MAX_SUBSCRIPTIONS.get(mode, 0)