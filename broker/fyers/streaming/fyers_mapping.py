"""
Fyers-specific mapping utilities for the WebSocket adapter - Updated to use brsymbol
"""
from typing import Dict, Set, Optional

class FyersExchangeMapper:
    """Maps between OpenAlgo exchange names and Fyers exchange codes"""
    
    # OpenAlgo to Fyers exchange mapping (based on Fyers SDK exch_seg_dict)
    EXCHANGE_MAP = {
        'NSE': 'NSE',
        'BSE': 'BSE', 
        'NFO': 'NFO',
        'BFO': 'BFO',
        'MCX': 'MCX',
        'CDS': 'CDS',
        'NSE_INDEX': 'NSE',  # Indices use base exchange
        'BSE_INDEX': 'BSE'
    }
    
    # Fyers internal segment codes (from SDK)
    FYERS_SEGMENT_CODES = {
        'NSE': '1010',    # NSE CM
        'NFO': '1011',    # NSE FO
        'BSE': '1210',    # BSE CM
        'BFO': '1211',    # BSE FO
        'MCX': '1120',    # MCX FO
        'CDS': '1012',    # Currency derivatives
        'CDE': '1012',    # Currency derivatives exchange
        'BCS': '1212',    # BSE currency segment
        'NSE_COM': '1020' # NSE commodity
    }
    
    # Reverse mapping
    FYERS_TO_OPENALGO = {v: k for k, v in EXCHANGE_MAP.items()}
    
    @classmethod
    def to_fyers_exchange(cls, oa_exchange: str) -> Optional[str]:
        """Convert OpenAlgo exchange to Fyers exchange format"""
        return cls.EXCHANGE_MAP.get(oa_exchange.upper())
    
    @classmethod
    def to_oa_exchange(cls, fyers_exchange: str) -> Optional[str]:
        """Convert Fyers exchange to OpenAlgo exchange format"""
        return cls.FYERS_TO_OPENALGO.get(fyers_exchange.upper())
    
    @classmethod
    def get_segment_code(cls, exchange: str) -> Optional[str]:
        """Get Fyers segment code for exchange"""
        return cls.FYERS_SEGMENT_CODES.get(exchange.upper())


class FyersCapabilityRegistry:
    """Registry for Fyers-specific capabilities and limits"""
    
    # Supported subscription modes (from SDK data_type options)
    SUPPORTED_MODES = {1, 2, 3}  # LTP, Quote, Depth
    
    # Fyers data types from SDK
    DATA_TYPES = {
        'SymbolUpdate': 'SymbolUpdate',
        'DepthUpdate': 'DepthUpdate'
    }
    
    # Maximum subscriptions per connection (from SDK)
    MAX_SUBSCRIPTIONS = 5000
    
    # Maximum symbols per subscription request
    MAX_SYMBOLS_PER_REQUEST = 500  # Based on SDK chunking
    
    # Supported depth levels (Fyers supports 5-level depth)
    SUPPORTED_DEPTH_LEVELS = {5}
    
    @classmethod
    def is_mode_supported(cls, mode: int) -> bool:
        """Check if a subscription mode is supported"""
        return mode in cls.SUPPORTED_MODES
    
    @classmethod
    def is_depth_level_supported(cls, depth_level: int) -> bool:
        """Check if a depth level is supported"""
        return depth_level in cls.SUPPORTED_DEPTH_LEVELS
    
    @classmethod
    def get_fallback_depth_level(cls, requested_depth: int) -> int:
        """Get the fallback depth level (always 5 for Fyers)"""
        return 5
    
    @classmethod
    def get_capabilities(cls) -> Dict[str, any]:
        """Get all capabilities"""
        return {
            'supported_modes': list(cls.SUPPORTED_MODES),
            'supported_depth_levels': list(cls.SUPPORTED_DEPTH_LEVELS),
            'max_subscriptions': cls.MAX_SUBSCRIPTIONS,
            'max_symbols_per_request': cls.MAX_SYMBOLS_PER_REQUEST,
            'data_types': list(cls.DATA_TYPES.keys())
        }


class FyersSymbolConverter:
    """Helper class for Fyers symbol conversion logic - Updated to use fytoken-based approach"""
    
    # Exchange segment codes (first 4 digits of fytoken)
    EXCHANGE_SEGMENTS = {
        "1010": "nse_cm",    # NSE Cash Market
        "1011": "nse_fo",    # NSE F&O
        "1120": "mcx_fo",    # MCX F&O
        "1210": "bse_cm",    # BSE Cash Market
        "1211": "bse_fo",    # BSE F&O (BFO)
        "1212": "bcs_fo",    # BSE Currency
        "1012": "cde_fo",    # CDE F&O
        "1020": "nse_com"    # NSE Commodity
    }
    
    # Known index mappings (from working version)
    INDEX_MAPPINGS = {
        "NSE:NIFTY50-INDEX": "Nifty 50",
        "NSE:NIFTYBANK-INDEX": "Nifty Bank", 
        "NSE:FINNIFTY-INDEX": "Nifty Fin Service",
        "NSE:INDIAVIX-INDEX": "India VIX",
        "NSE:NIFTY100-INDEX": "Nifty 100",
        "NSE:NIFTYNEXT50-INDEX": "Nifty Next 50",
        "NSE:NIFTYMIDCAP50-INDEX": "Nifty Midcap 50",
        "NSE:NIFTYSMLCAP50-INDEX": "NIFTY SMLCAP 50",
        "BSE:SENSEX-INDEX": "SENSEX",
        "BSE:BANKEX-INDEX": "BANKEX",
        "BSE:BSE500-INDEX": "BSE500",
        "BSE:BSE100-INDEX": "BSE100",
        "BSE:BSE200-INDEX": "BSE200"
    }
    
    @classmethod
    def create_hsm_symbol_with_database_token(cls, brsymbol: str, database_token: str, data_type: str) -> Optional[str]:
        """
        Create HSM symbol format using token from database (more efficient than API calls)
        
        Args:
            brsymbol: Broker symbol format (e.g., NSE:TCS-EQ)
            database_token: Token from database (e.g., "101000000011536")
            data_type: SymbolUpdate or DepthUpdate
            
        Returns:
            HSM formatted symbol or None if invalid
        """
        try:
            return cls._convert_to_hsm_token(brsymbol, database_token, data_type)
        except Exception as e:
            import logging
            logging.error(f"Error creating HSM symbol with database token for {brsymbol}: {e}")
            return None
    
    
    @classmethod
    def _convert_to_hsm_token(cls, symbol: str, fytoken: str, data_type: str) -> Optional[str]:
        """
        Convert a single symbol and fytoken to HSM token format (from working version)
        """
        try:
            if len(fytoken) < 10:
                return None
                
            ex_sg = fytoken[:4]
            
            if ex_sg not in cls.EXCHANGE_SEGMENTS:
                return None
                
            segment = cls.EXCHANGE_SEGMENTS[ex_sg]
            is_index = symbol.endswith("-INDEX")
            
            if is_index:
                if data_type == "DepthUpdate":
                    return None  # Index doesn't support depth
                
                # Use proper index mapping from working version
                if symbol in cls.INDEX_MAPPINGS:
                    token_name = cls.INDEX_MAPPINGS[symbol]
                else:
                    token_name = symbol.split(":")[1].replace("-INDEX", "")
                hsm_token = f"if|{segment}|{token_name}"
            elif data_type == "DepthUpdate":
                token_suffix = fytoken[10:]
                hsm_token = f"dp|{segment}|{token_suffix}"
            else:
                token_suffix = fytoken[10:]
                hsm_token = f"sf|{segment}|{token_suffix}"
            
            return hsm_token
            
        except Exception:
            return None
    
    @classmethod
    def create_hsm_symbol(cls, brsymbol: str, brexchange: str, data_type: str) -> Optional[str]:
        """
        Legacy method - kept for backward compatibility but should use create_hsm_symbol_with_fytoken
        """
        # This method is deprecated - use create_hsm_symbol_with_fytoken instead
        return None