"""
Debug timestamp format to understand the data structure
"""

import asyncio
import websockets
import json
import time


async def debug_timestamps():
    """Debug timestamp format"""
    print("🔍 DEBUGGING TIMESTAMP FORMAT")
    print("=" * 50)
    
    uri = "ws://127.0.0.1:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Handle connection message
            connection_response = await websocket.recv()
            connection_data = json.loads(connection_response)
            print(f"Connection: {connection_data}")
            
            # Authenticate
            auth_message = {
                "action": "authenticate",
                "api_key": "32a3d40880806508fa1397e3b7621a58acff440470d08562863adebdb4f6c194"
            }
            await websocket.send(json.dumps(auth_message))
            
            auth_response = await websocket.recv()
            auth_data = json.loads(auth_response)
            print(f"Auth: {auth_data}")
            
            # Subscribe to one symbol
            subscribe_message = {
                "action": "subscribe",
                "symbol": "TCS",
                "exchange": "NSE",
                "mode": 1
            }
            await websocket.send(json.dumps(subscribe_message))
            
            sub_response = await websocket.recv()
            sub_data = json.loads(sub_response)
            print(f"Subscription: {sub_data}")
            
            # Collect a few messages to analyze
            print("\n📊 Analyzing first 5 messages:")
            print("-" * 50)
            
            for i in range(5):
                message = await websocket.recv()
                client_time = time.time() * 1000  # Current time in ms
                
                try:
                    data = json.loads(message)
                    
                    print(f"\nMessage {i+1}:")
                    print(f"  Client receive time: {client_time}")
                    print(f"  Raw message: {json.dumps(data, indent=2)}")
                    
                    if data.get("type") == "market_data":
                        broker_ts = data.get('broker_timestamp')
                        proxy_ts = data.get('proxy_timestamp')
                        
                        print(f"  Broker timestamp: {broker_ts} (type: {type(broker_ts)})")
                        print(f"  Proxy timestamp: {proxy_ts} (type: {type(proxy_ts)})")
                        
                        if broker_ts:
                            print(f"  Broker timestamp analysis:")
                            print(f"    Raw value: {broker_ts}")
                            print(f"    As seconds: {broker_ts}")
                            print(f"    As milliseconds: {broker_ts * 1000}")
                            print(f"    Current time (s): {time.time()}")
                            print(f"    Current time (ms): {time.time() * 1000}")
                            
                            # Calculate different interpretations
                            if broker_ts < 1e10:  # Likely seconds
                                broker_ms = broker_ts * 1000
                                latency_if_seconds = client_time - broker_ms
                                print(f"    Latency if broker_ts is seconds: {latency_if_seconds:.3f}ms")
                            
                            if broker_ts > 1e12:  # Likely milliseconds or higher
                                latency_if_ms = client_time - broker_ts
                                print(f"    Latency if broker_ts is milliseconds: {latency_if_ms:.3f}ms")
                        
                        if proxy_ts:
                            print(f"  Proxy timestamp analysis:")
                            print(f"    Raw value: {proxy_ts}")
                            latency_proxy = client_time - proxy_ts
                            print(f"    Latency from proxy: {latency_proxy:.3f}ms")
                
                except json.JSONDecodeError as e:
                    print(f"  JSON decode error: {e}")
                except Exception as e:
                    print(f"  Error: {e}")
            
            # Unsubscribe
            unsubscribe_message = {
                "action": "unsubscribe",
                "symbol": "TCS",
                "exchange": "NSE",
                "mode": 1
            }
            await websocket.send(json.dumps(unsubscribe_message))
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")


if __name__ == "__main__":
    asyncio.run(debug_timestamps())