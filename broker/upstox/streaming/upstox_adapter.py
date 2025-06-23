"""
Upstox WebSocket adapter for OpenAlgo WebSocket proxy
"""
import logging
import threading
import sys
import os
import uuid
from typing import Dict, Any, Optional, List

# Ensure debug logs are visible for this adapter
logging.getLogger("upstox_adapter").setLevel(logging.DEBUG)

from websocket_proxy.base_adapter import BaseBrokerWebSocketAdapter
from websocket_proxy.mapping import SymbolMapper
from .upstox_mapping import UpstoxExchangeMapper, UpstoxCapabilityRegistry
from .upstox_websocket import UpstoxWebSocket
from database.auth_db import get_auth_token
from database.token_db import get_token

class UpstoxWebSocketAdapter(BaseBrokerWebSocketAdapter):
    """Upstox-specific implementation of the WebSocket adapter"""
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("upstox_adapter")
        self.ws_client = None
        self.user_id = None
        self.broker_name = "upstox"
        self.running = False
        self.lock = threading.Lock()
        self.access_token = None
        self.subscriptions = {}  # {(symbol, oa_exchange): mode}
        self.ltp_data = {}  # Cache for LTP data
        self.instrument_key_map = {}  # {instrument_key: (symbol, oa_exchange, mode)}
        self.token_to_symbol = {}  # {token or scrip: OpenAlgo symbol}

    def initialize(self, broker_name: str, user_id: str = None, auth_data: Optional[Dict[str, str]] = None) -> None:
        self.logger.debug(f"Initializing adapter: broker_name={broker_name}, user_id={user_id}, auth_data={auth_data}")
        self.broker_name = broker_name
        self.user_id = user_id
        token = get_auth_token(user_id) if user_id else None
        if token:
            self.access_token = token
        else:
            self.logger.warning(f"No Upstox access token found for user_id '{user_id}' in DB.")

    def connect(self) -> None:
        self.logger.debug("Connecting Upstox adapter...")
        if not self.access_token:
            raise Exception("Upstox access_token required for WebSocket connection.")
        self.ws_client = UpstoxWebSocket(
            access_token=self.access_token,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        self.ws_client.connect()
        self.running = True

    def disconnect(self) -> None:
        self.logger.debug("Disconnecting Upstox adapter...")
        if self.ws_client:
            self.ws_client.disconnect()
        self.running = False

    def subscribe(self, symbol: str, exchange: str, mode: str = "ltpc", depth_level: int = 5) -> Dict[str, Any]:
        self.logger.debug(f"Subscribe called: symbol={symbol}, exchange={exchange}, mode={mode}, depth_level={depth_level}")
        # Get token for the symbol
        token = get_token(symbol, exchange)
        self.logger.debug(f"[DEBUG] get_token({symbol!r}, {exchange!r}) returned: {token!r}")
        if not token:
            # Try to print all available tokens for debugging
            try:
                from database import token_db
                all_tokens = getattr(token_db, 'TOKENS', None)
                if all_tokens:
                    self.logger.error(f"[DEBUG] Available tokens in token_db: {list(all_tokens.keys())}")
                else:
                    self.logger.error("[DEBUG] Could not find TOKENS in token_db.")
            except Exception as e:
                self.logger.error(f"[DEBUG] Exception while listing all tokens: {e}")
            self.logger.error(f"Token not found for {symbol} on {exchange}")
            return {
                "status": "error",
                "error": f"Token not found for {symbol} on {exchange}",
                "symbol": symbol,
                "exchange": exchange
            }
        guid = str(uuid.uuid4())
        upstox_exchange = UpstoxExchangeMapper.to_upstox_exchange(exchange)
        # If token already contains a pipe, use as is
        if "|" in str(token):
            instrument_key = str(token)
            token_part = instrument_key.split("|")[-1]
        else:
            instrument_key = f"{upstox_exchange}|{token}"
            token_part = str(token)
        # Map OpenAlgo mode to Upstox mode and topic mode string
        upstox_mode = "ltpc"  # Default to ltpc
        topic_mode_str = "LTP"
        if mode in ["full", "quote", 2, "2", 3, "3"]:
            upstox_mode = "full"
            topic_mode_str = "FULL" if str(mode).lower() == "full" or str(mode) in ["3", 3] else "QUOTE"
        elif mode == "depth":
            upstox_mode = "full_d30" if depth_level > 5 else "full"
            topic_mode_str = "FULL"
        elif str(mode).lower() == "ltpc" or str(mode) in ["1", 1]:
            topic_mode_str = "LTP"
        self.logger.debug(f"Adapter subscribing: symbol={symbol}, exchange={exchange}, token={token}, mode={upstox_mode}, guid={guid}, instrument_key={instrument_key}")
        self.ws_client.subscribe(guid, upstox_mode, [instrument_key])
        # Store using OA exchange for correct lookup in _on_message
        oa_exchange = UpstoxExchangeMapper.to_oa_exchange(exchange)
        # Use topic_mode_str for subscription topic
        self.subscriptions[(symbol, oa_exchange)] = topic_mode_str
        self.instrument_key_map[instrument_key] = (symbol, oa_exchange, topic_mode_str)
        # Map Upstox token/scrip to OpenAlgo symbol for normalization
        self.token_to_symbol[token_part] = symbol
        # Return in a structure compatible with server/client expectations
        return {
            "subscriptions": [
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "status": "subscribed",
                    "mode": topic_mode_str,
                    "token": token,
                    "broker": self.broker_name
                }
            ],
            "status": "success"
        }

    def unsubscribe(self, symbol: str, exchange: str, mode: str = "ltpc") -> Dict[str, Any]:
        self.logger.debug(f"Unsubscribe called: symbol={symbol}, exchange={exchange}, mode={mode}")
        # Get token for the symbol
        token = get_token(symbol, exchange)
        if not token:
            self.logger.error(f"Token not found for {symbol} on {exchange}")
            return {
                "status": "error",
                "error": f"Token not found for {symbol} on {exchange}",
                "symbol": symbol,
                "exchange": exchange
            }
        guid = str(uuid.uuid4())
        upstox_exchange = UpstoxExchangeMapper.to_upstox_exchange(exchange)
        if "|" in str(token):
            instrument_key = str(token)
        else:
            instrument_key = f"{upstox_exchange}|{token}"
        self.logger.debug(f"Adapter unsubscribing: symbol={symbol}, exchange={exchange}, token={token}, mode={mode}, guid={guid}, instrument_key={instrument_key}")
        self.ws_client.unsubscribe(guid, [instrument_key])
        oa_exchange = UpstoxExchangeMapper.to_oa_exchange(exchange)
        self.subscriptions.pop((symbol, oa_exchange), None)
        self.instrument_key_map.pop(instrument_key, None)
        self.ltp_data.pop(f"{exchange}|{symbol}", None)  # Clear cached LTP data
        return {"status": "unsubscribed", "symbol": symbol, "exchange": exchange, "mode": mode, "token": token}

    def _on_open(self, ws):
        self.logger.info("Upstox WebSocket connection opened.")

    def _on_message(self, ws, message):
        self.logger.debug(f"Raw message received in adapter: {message}")
        try:
            if isinstance(message, dict):
                normalized = self._normalize_market_data(message)
                self.logger.debug(f"Normalized market data: {normalized}")
                if normalized and 'feeds' in normalized:
                    for feed in normalized['feeds']:
                        # Find the instrument_key for this feed
                        instrument_key = None
                        for key, val in self.instrument_key_map.items():
                            _, oa_exchange, _ = val
                            if oa_exchange == feed['exchange'] and key.split('|')[-1] == feed['symbol']:
                                instrument_key = key
                                break
                        if instrument_key and instrument_key in self.instrument_key_map:
                            symbol, oa_exchange, mode = self.instrument_key_map[instrument_key]
                        else:
                            mode = 'LTP'
                        # Map mode to string for topic (OpenAlgo expects uppercase)
                        mode_map = {
                            "1": "LTP", 1: "LTP",
                            "2": "QUOTE", 2: "QUOTE",
                            "3": "FULL", 3: "FULL",
                            "ltpc": "LTP",
                            "quote": "QUOTE",
                            "full": "FULL",
                            "depth": "FULL"
                        }
                        # Determine topic based on available data
                        if 'quote' in feed or 'depth' in feed or 'quote' in feed.get('quote', {}):
                            mode_str = 'QUOTE'
                        elif 'ltp' in feed:
                            mode_str = 'LTP'
                        else:
                            mode_str = mode_map.get(mode, str(mode).upper())
                        topic = f"{feed['exchange']}_{feed['symbol']}_{mode_str}"
                        # For QUOTE, flatten OHLC and LTP fields to top-level for client compatibility
                        if mode_str == 'QUOTE':
                            # Use OHLC if present
                            ohlc = feed.get('ohlc')
                            if ohlc and isinstance(ohlc, list) and len(ohlc) > 0:
                                latest_ohlc = ohlc[0]
                                for k in ['open', 'high', 'low', 'close']:
                                    feed[k] = latest_ohlc.get(k, 0)
                            # Use LTP if present
                            if 'ltp' in feed:
                                feed['ltp'] = feed['ltp']
                            elif 'quote' in feed and 'buy' in feed['quote'] and len(feed['quote']['buy']) > 0:
                                feed['ltp'] = feed['quote']['buy'][0].get('price', 0)
                            # Ensure all fields are present
                            for k in ['open', 'high', 'low', 'close', 'ltp']:
                                if k not in feed:
                                    feed[k] = 0
                        self.publish_market_data(topic, feed)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def _normalize_market_data(self, message: dict) -> dict:
        self.logger.debug(f"Normalizing market data: {message}")
        """Normalize Upstox market data to OpenAlgo format"""
        try:
            if not isinstance(message, dict):
                return None

            if "feeds" not in message:
                return None

            normalized_feeds = []
            for instrument_key, feed in message["feeds"].items():
                parts = instrument_key.split("|")
                if len(parts) == 2:
                    exchange, symbol = parts
                elif len(parts) == 3:
                    exchange, _, symbol = parts
                else:
                    exchange, symbol = parts[0], "|".join(parts[1:])
                oa_exchange = UpstoxExchangeMapper.to_oa_exchange(exchange)

                # Use OpenAlgo symbol if available
                oa_symbol = self.token_to_symbol.get(symbol, symbol)
                normalized_data = {
                    "exchange": oa_exchange,
                    "symbol": oa_symbol,
                    "timestamp": message.get("currentTs", 0)
                }

                # Handle different feed types
                # LTP
                if "ltpc" in feed:
                    ltpc = feed["ltpc"]
                    normalized_data.update({
                        "ltp": ltpc.get("ltp", 0),
                        "ltq": ltpc.get("ltq", 0),
                        "ltt": ltpc.get("ltt", 0),
                        "close": ltpc.get("cp", 0)
                    })
                    # Cache LTP data
                    self.ltp_data[f"{oa_exchange}|{oa_symbol}"] = normalized_data
                # QUOTE (bid/ask)
                # Upstox fullFeed/marketFF/marketLevel/bidAskQuote
                if "fullFeed" in feed:
                    ff = feed["fullFeed"]
                    # Index quote
                    if "indexFF" in ff and "marketOHLC" in ff["indexFF"]:
                        ohlc = ff["indexFF"]["marketOHLC"].get("ohlc", [])
                        normalized_data["ohlc"] = ohlc
                    # Market quote
                    if "marketFF" in ff:
                        mff = ff["marketFF"]
                        if "marketLevel" in mff and "bidAskQuote" in mff["marketLevel"]:
                            quotes = mff["marketLevel"]["bidAskQuote"]
                            if quotes:
                                normalized_data["quote"] = {
                                    "buy": [{"quantity": q.get("bidQ", q.get("askQ", 0)), "price": q.get("bidP", q.get("askP", 0))} for q in quotes if q],
                                    "sell": [{"quantity": q.get("askQ", 0), "price": q.get("askP", 0)} for q in quotes if q]
                                }
                        # Add OHLC if present
                        if "marketOHLC" in mff:
                            normalized_data["ohlc"] = mff["marketOHLC"].get("ohlc", [])
                        # Add ATP, VTT, TSQ if present
                        for k in ["atp", "vtt", "tsq"]:
                            if k in mff:
                                normalized_data[k] = mff[k]
                normalized_feeds.append(normalized_data)

            return {"feeds": normalized_feeds} if normalized_feeds else None

        except Exception as e:
            self.logger.error(f"Error normalizing market data: {e}")
            return None

    def _on_error(self, ws, error):
        self.logger.error(f"WebSocket error in adapter: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.info(f"WebSocket closed in adapter: {close_status_code} {close_msg}")
        self.connected = False

    def get_ltp_data(self) -> Dict[str, Any]:
        self.logger.debug("Returning cached LTP data from adapter.")
        """Return cached LTP data for all subscribed instruments"""
        return {"feeds": list(self.ltp_data.values())}

    def publish_market_data(self, topic, data):
        """
        Publish market data to ZeroMQ subscribers with debug logging
        """
        self.logger.debug(f"[DEBUG] publish_market_data called with topic: '{topic}', data: {data}")
        # Confirm topic matches any current subscription (case-sensitive)
        if hasattr(self, 'subscriptions'):
            sub_topics = [f"{ex}_{sym}_{str(mode).upper()}" for (sym, ex), mode in self.subscriptions.items()]
            self.logger.debug(f"[DEBUG] Current subscription topics: {sub_topics}")
            if topic in sub_topics:
                self.logger.debug(f"[DEBUG] Topic '{topic}' matches a current subscription.")
            else:
                self.logger.warning(f"[DEBUG] Topic '{topic}' does NOT match any current subscription!")
        try:
            super().publish_market_data(topic, data)
        except Exception as e:
            self.logger.exception(f"Error publishing market data: {e}")