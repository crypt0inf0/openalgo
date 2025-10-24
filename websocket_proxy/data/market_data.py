"""Market data message format for shared memory communication."""

import json
import struct
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class MarketDataMessage:
    """Market data message structure for efficient serialization."""
    
    symbol: str
    timestamp: float
    price: float
    exchange: str = "NSE"  # ADD THIS FIELD
    volume: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_quantity: float = 0.0
    ask_quantity: float = 0.0
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    average_price: float = 0.0
    percent_change: float = 0.0
    last_quantity: float = 0.0
    
    # Totals
    total_buy_quantity: float = 0.0
    total_sell_quantity: float = 0.0
    
    # Depth levels (5 levels each side): price, quantity, orders
    bid_price_1: float = 0.0
    bid_qty_1: float = 0.0
    bid_orders_1: float = 0.0
    bid_price_2: float = 0.0
    bid_qty_2: float = 0.0
    bid_orders_2: float = 0.0
    bid_price_3: float = 0.0
    bid_qty_3: float = 0.0
    bid_orders_3: float = 0.0
    bid_price_4: float = 0.0
    bid_qty_4: float = 0.0
    bid_orders_4: float = 0.0
    bid_price_5: float = 0.0
    bid_qty_5: float = 0.0
    bid_orders_5: float = 0.0
    
    ask_price_1: float = 0.0
    ask_qty_1: float = 0.0
    ask_orders_1: float = 0.0
    ask_price_2: float = 0.0
    ask_qty_2: float = 0.0
    ask_orders_2: float = 0.0
    ask_price_3: float = 0.0
    ask_qty_3: float = 0.0
    ask_orders_3: float = 0.0
    ask_price_4: float = 0.0
    ask_qty_4: float = 0.0
    ask_orders_4: float = 0.0
    ask_price_5: float = 0.0
    ask_qty_5: float = 0.0
    ask_orders_5: float = 0.0
    
    message_type: int = 1  # 1=LTP, 2=Quote, 3=Depth
    
    # Binary format for maximum speed - UPDATED to include exchange field
    _STRUCT_FORMAT = "<32s32s" + ("d" * (14 + 2 + 30)) + "B"
    _STRUCT_SIZE = struct.calcsize(_STRUCT_FORMAT)
    
    @classmethod
    def get_size(cls) -> int:
        """Get the fixed size of the serialized message in bytes."""
        return cls._STRUCT_SIZE
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes for shared memory."""
        symbol_bytes = self.symbol.encode('ascii')[:32].ljust(32, b'\x00')
        exchange_bytes = self.exchange.encode('ascii')[:32].ljust(32, b'\x00')  # ADD THIS
        
        return struct.pack(
            self._STRUCT_FORMAT,
            symbol_bytes,
            exchange_bytes,  # ADD THIS
            self.timestamp, self.price, self.volume,
            self.bid_price, self.ask_price, self.bid_quantity, self.ask_quantity,
            self.open_price, self.high_price, self.low_price, self.close_price,
            self.average_price, self.percent_change, self.last_quantity,
            self.total_buy_quantity, self.total_sell_quantity,
            # Bid depth
            self.bid_price_1, self.bid_qty_1, self.bid_orders_1,
            self.bid_price_2, self.bid_qty_2, self.bid_orders_2,
            self.bid_price_3, self.bid_qty_3, self.bid_orders_3,
            self.bid_price_4, self.bid_qty_4, self.bid_orders_4,
            self.bid_price_5, self.bid_qty_5, self.bid_orders_5,
            # Ask depth  
            self.ask_price_1, self.ask_qty_1, self.ask_orders_1,
            self.ask_price_2, self.ask_qty_2, self.ask_orders_2,
            self.ask_price_3, self.ask_qty_3, self.ask_orders_3,
            self.ask_price_4, self.ask_qty_4, self.ask_orders_4,
            self.ask_price_5, self.ask_qty_5, self.ask_orders_5,
            self.message_type
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MarketDataMessage':
        """Deserialize message from bytes."""
        try:
            unpacked = struct.unpack(cls._STRUCT_FORMAT, data[:cls._STRUCT_SIZE])
            
            (symbol_bytes, exchange_bytes, timestamp, price, volume, bid_price, ask_price, 
             bid_quantity, ask_quantity, open_price, high_price, low_price, 
             close_price, average_price, percent_change, last_quantity,
             total_buy_quantity, total_sell_quantity,
             bp1, bq1, bo1, bp2, bq2, bo2, bp3, bq3, bo3, bp4, bq4, bo4, bp5, bq5, bo5,
             ap1, aq1, ao1, ap2, aq2, ao2, ap3, aq3, ao3, ap4, aq4, ao4, ap5, aq5, ao5,
             message_type) = unpacked
            
            symbol = symbol_bytes.decode('ascii').rstrip('\x00')
            exchange = exchange_bytes.decode('ascii').rstrip('\x00')  # ADD THIS
            
            return cls(
                symbol=symbol, exchange=exchange, timestamp=timestamp, price=price, volume=volume,  # ADD exchange
                bid_price=bid_price, ask_price=ask_price, bid_quantity=bid_quantity,
                ask_quantity=ask_quantity, open_price=open_price, high_price=high_price,
                low_price=low_price, close_price=close_price, average_price=average_price,
                percent_change=percent_change, last_quantity=last_quantity,
                total_buy_quantity=total_buy_quantity, total_sell_quantity=total_sell_quantity,
                bid_price_1=bp1, bid_qty_1=bq1, bid_orders_1=bo1,
                bid_price_2=bp2, bid_qty_2=bq2, bid_orders_2=bo2,
                bid_price_3=bp3, bid_qty_3=bq3, bid_orders_3=bo3,
                bid_price_4=bp4, bid_qty_4=bq4, bid_orders_4=bo4,
                bid_price_5=bp5, bid_qty_5=bq5, bid_orders_5=bo5,
                ask_price_1=ap1, ask_qty_1=aq1, ask_orders_1=ao1,
                ask_price_2=ap2, ask_qty_2=aq2, ask_orders_2=ao2,
                ask_price_3=ap3, ask_qty_3=aq3, ask_orders_3=ao3,
                ask_price_4=ap4, ask_qty_4=aq4, ask_orders_4=ao4,
                ask_price_5=ap5, ask_qty_5=aq5, ask_orders_5=ao5,
                message_type=message_type
            )
        except struct.error as e:
            raise ValueError(f"Failed to deserialize message: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        base = {
            'symbol': self.symbol,
            'exchange': self.exchange,  # ADD THIS LINE
            'timestamp': self.timestamp,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid_price,
            'ask': self.ask_price,
            'bid_qty': self.bid_quantity,
            'ask_qty': self.ask_quantity,
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'average_price': self.average_price,
            'percent_change': self.percent_change,
            'last_quantity': self.last_quantity,
            'total_buy_quantity': self.total_buy_quantity,
            'total_sell_quantity': self.total_sell_quantity,
            'type': self.get_message_type_str(),
            'time': datetime.fromtimestamp(self.timestamp).isoformat()
        }
        
        # Include depth for DEPTH messages or when depth levels are populated
        depth_buy = [
            {'price': self.bid_price_1, 'quantity': self.bid_qty_1, 'orders': self.bid_orders_1},
            {'price': self.bid_price_2, 'quantity': self.bid_qty_2, 'orders': self.bid_orders_2},
            {'price': self.bid_price_3, 'quantity': self.bid_qty_3, 'orders': self.bid_orders_3},
            {'price': self.bid_price_4, 'quantity': self.bid_qty_4, 'orders': self.bid_orders_4},
            {'price': self.bid_price_5, 'quantity': self.bid_qty_5, 'orders': self.bid_orders_5},
        ]
        
        depth_sell = [
            {'price': self.ask_price_1, 'quantity': self.ask_qty_1, 'orders': self.ask_orders_1},
            {'price': self.ask_price_2, 'quantity': self.ask_qty_2, 'orders': self.ask_orders_2},
            {'price': self.ask_price_3, 'quantity': self.ask_qty_3, 'orders': self.ask_orders_3},
            {'price': self.ask_price_4, 'quantity': self.ask_qty_4, 'orders': self.ask_orders_4},
            {'price': self.ask_price_5, 'quantity': self.ask_qty_5, 'orders': self.ask_orders_5},
        ]
        
        # Add depth if any values are non-zero
        if any(level['price'] or level['quantity'] or level['orders'] 
               for level in depth_buy + depth_sell):
            base['depth'] = {
                'buy': depth_buy,
                'sell': depth_sell,
            }
        
        return base
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    def get_message_type_str(self) -> str:
        """Get message type as string."""
        return {1: 'LTP', 2: 'QUOTE', 3: 'DEPTH'}.get(self.message_type, 'UNKNOWN')