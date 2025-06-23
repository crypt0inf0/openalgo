"""
Low-level WebSocket client for Upstox Market Data Feed V3
Handles connection, authentication, subscription, and message parsing.
"""
import threading
import websocket
import requests
import json
import logging
import ssl
from typing import Callable, List, Dict, Any, Optional
from . import MarketDataFeedV3_pb2 as pb

class UpstoxWebSocket:
    AUTH_URL = "https://api.upstox.com/v3/feed/market-data-feed/authorize"

    def __init__(self, access_token: str, on_message: Optional[Callable] = None, on_error: Optional[Callable] = None, on_close: Optional[Callable] = None, on_open: Optional[Callable] = None):
        self.access_token = access_token
        self.ws = None
        self.connected = False
        self.logger = logging.getLogger("upstox_websocket")
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.on_open = on_open
        self._thread = None
        self._stop_event = threading.Event()

    def _log_http_request(self, method, url, headers=None, data=None, response=None):
        self.logger.debug(f"HTTP {method} Request: URL={url}")
        if headers:
            self.logger.debug(f"Request Headers: {headers}")
        if data:
            self.logger.debug(f"Request Data: {data}")
        if response is not None:
            self.logger.debug(f"HTTP Response Status: {response.status_code}")
            self.logger.debug(f"Response Headers: {response.headers}")
            try:
                self.logger.debug(f"Response Body: {response.text}")
            except Exception as e:
                self.logger.debug(f"Could not decode response body: {e}")

    def _get_websocket_url(self) -> str:
        """Get authorized WebSocket URL from Upstox"""
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        url = self.AUTH_URL
        self.logger.debug("Preparing to send GET request for WebSocket authorization URL.")
        response = requests.get(url=url, headers=headers)
        self._log_http_request("GET", url, headers, None, response)
        if response.status_code != 200:
            raise Exception(f"Failed to get WebSocket URL. Status: {response.status_code}, Response: {response.text}")
        data = response.json()
        return data["data"]["authorized_redirect_uri"]

    def connect(self):
        try:
            ws_url = self._get_websocket_url()
            self.logger.debug(f"Connecting to Upstox WebSocket at {ws_url}")
            # Create SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            self.logger.debug("Creating WebSocketApp instance.")
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )

            def run_websocket():
                self.logger.debug("Starting WebSocket run_forever loop.")
                self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
                self.logger.debug("WebSocket run_forever loop ended.")

            self._thread = threading.Thread(target=run_websocket, daemon=True)
            self._thread.start()

            # Wait for connection
            for _ in range(30):
                if self.connected:
                    self.logger.debug("WebSocket connection established.")
                    return True
                self._stop_event.wait(0.1)
            self.logger.debug("WebSocket connection attempt timed out.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            if self.on_error:
                self.on_error(self.ws, e)
            return False

    def disconnect(self):
        self.logger.debug("Disconnecting Upstox WebSocket.")
        self._stop_event.set()
        if self.ws:
            self.logger.debug("Closing WebSocket connection.")
            self.ws.close()
        self.connected = False

    def _on_open(self, ws):
        self.logger.info("WebSocket connection opened.")
        if self.on_open:
            self.on_open(ws)

    def _on_message(self, ws, message):
        self.logger.debug(f"Raw message from broker: {message}")
        if isinstance(message, bytes):
            self.logger.debug(f"Received binary message of length {len(message)} bytes.")
        else:
            self.logger.debug(f"Received text message: {message}")
        # Protobuf decoding for binary messages
        try:
            if isinstance(message, bytes):
                self.logger.debug(f"Received binary message of length {len(message)} bytes.")
                feed_response = pb.FeedResponse()
                feed_response.ParseFromString(message)
                msg_dict = self._protobuf_to_dict(feed_response)
                self.logger.debug(f"Decoded protobuf message: {msg_dict}")
                if self.on_message:
                    self.on_message(ws, msg_dict)
            else:
                self.logger.debug(f"Received text message: {message}")
                if self.on_message:
                    self.on_message(ws, message)
        except Exception as e:
            self.logger.error(f"Failed to decode protobuf message: {e}")
            if self.on_error:
                self.on_error(ws, e)

    def _protobuf_to_dict(self, proto_msg):
        from google.protobuf.json_format import MessageToDict
        msg_dict = MessageToDict(proto_msg, preserving_proto_field_name=True)
        self.logger.debug(f"Protobuf to dict: {msg_dict}")
        return msg_dict

    def _on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")
        if self.on_error:
            self.on_error(ws, error)

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.info(f"WebSocket closed: {close_status_code} {close_msg}")
        if self.on_close:
            self.on_close(ws, close_status_code, close_msg)

    def send_binary(self, data: bytes):
        if self.ws and self.connected:
            self.logger.debug(f"Sending binary data of length {len(data)} bytes over WebSocket.")
            self.ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)

    def subscribe(self, guid: str, mode: str, instrument_keys: List[str]):
        req = {
            "guid": guid,
            "method": "sub",
            "data": {
                "mode": mode,
                "instrumentKeys": instrument_keys
            }
        }
        self.logger.debug(f"Sending subscribe request (binary): {req}")
        if self.ws:
            self.ws.send(json.dumps(req).encode('utf-8'), opcode=websocket.ABNF.OPCODE_BINARY)
        else:
            self.logger.error("WebSocket is not connected. Cannot send subscribe request.")

    def unsubscribe(self, guid: str, instrument_keys: List[str]):
        req = {
            "guid": guid,
            "method": "unsub",
            "data": {
                "instrumentKeys": instrument_keys
            }
        }
        self.logger.debug(f"Sending unsubscribe request (binary): {req}")
        if self.ws:
            self.ws.send(json.dumps(req).encode('utf-8'), opcode=websocket.ABNF.OPCODE_BINARY)
        else:
            self.logger.error("WebSocket is not connected. Cannot send unsubscribe request.")