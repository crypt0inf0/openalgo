"""
WebSocket Proxy Stress Test
Tests the lightweight proxy under heavy load with multiple concurrent connections
"""

import sys
import os
import time
import json
import asyncio
import websockets
import threading
import statistics
from datetime import datetime
from collections import deque, defaultdict
import concurrent.futures

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ProxyStressTest:
    """Stress test for WebSocket proxy with multiple concurrent clients"""
    
    def __init__(self, ws_url="ws://127.0.0.1:8765", api_key="32a3d40880806508fa1397e3b7621a58acff440470d08562863adebdb4f6c194"):
        self.ws_url = ws_url
        self.api_key = api_key
        self.results = defaultdict(list)
        self.lock = threading.Lock()
        
    async def single_client_test(self, client_id, duration_seconds=30):
        """Run test for a single WebSocket client"""
        messages_received = 0
        latencies = deque(maxlen=100)
        connection_time = None
        
        try:
            # Connect to WebSocket
            connect_start = time.time()
            async with websockets.connect(self.ws_url) as websocket:
                connection_time = time.time() - connect_start
                
                # Authenticate
                auth_message = {
                    "action": "authenticate",
                    "api_key": self.api_key
                }
                await websocket.send(json.dumps(auth_message))
                
                # Wait for auth response
                auth_response = await websocket.recv()
                auth_data = json.loads(auth_response)
                
                if auth_data.get("type") != "auth" or auth_data.get("status") != "success":
                    raise Exception(f"Authentication failed: {auth_data}")
                
                # Subscribe to a test symbol
                subscribe_message = {
                    "action": "subscribe",
                    "symbol": "TCS",
                    "exchange": "NSE",
                    "mode": 1
                }
                await websocket.send(json.dumps(subscribe_message))
                
                # Wait for subscription response
                sub_response = await websocket.recv()
                
                # Start receiving data
                start_time = time.time()
                end_time = start_time + duration_seconds
                
                while time.time() < end_time:
                    try:
                        # Set timeout to avoid hanging
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        receive_time = time.time() * 1000
                        
                        data = json.loads(message)
                        
                        # Only count market data messages
                        if data.get("type") == "market_data":
                            messages_received += 1
                            
                            # Calculate latency if timestamp available
                            if "timestamp" in data:
                                broker_time = data["timestamp"]
                                if isinstance(broker_time, (int, float)):
                                    latency_ms = receive_time - broker_time
                                    if 0 <= latency_ms <= 10000:  # Reasonable range
                                        latencies.append(latency_ms)
                        
                    except asyncio.TimeoutError:
                        continue  # No message received, continue
                    except Exception as e:
                        print(f"Client {client_id} message error: {e}")
                        break
                
                # Unsubscribe
                unsubscribe_message = {
                    "action": "unsubscribe",
                    "symbol": "TCS",
                    "exchange": "NSE",
                    "mode": 1
                }
                await websocket.send(json.dumps(unsubscribe_message))
                
        except Exception as e:
            print(f"Client {client_id} error: {e}")
            return None
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies) if latencies else 0
        message_rate = messages_received / duration_seconds if duration_seconds > 0 else 0
        
        return {
            'client_id': client_id,
            'messages_received': messages_received,
            'message_rate': message_rate,
            'avg_latency_ms': avg_latency,
            'connection_time_ms': connection_time * 1000 if connection_time else 0,
            'latency_samples': len(latencies)
        }
    
    async def run_concurrent_test(self, num_clients=10, duration_seconds=30):
        """Run stress test with multiple concurrent clients"""
        print(f"🚀 Starting stress test with {num_clients} concurrent clients")
        print(f"⏱️  Duration: {duration_seconds} seconds")
        print(f"🔗 WebSocket URL: {self.ws_url}")
        
        # Create tasks for all clients
        tasks = []
        for i in range(num_clients):
            task = asyncio.create_task(
                self.single_client_test(i, duration_seconds)
            )
            tasks.append(task)
        
        # Run all clients concurrently
        print(f"\n🏃 Running {num_clients} concurrent clients...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Process results
        successful_results = [r for r in results if isinstance(r, dict)]
        failed_clients = len(results) - len(successful_results)
        
        print(f"\n✅ Test completed in {total_duration:.1f} seconds")
        print(f"📊 Successful clients: {len(successful_results)}/{num_clients}")
        
        if failed_clients > 0:
            print(f"❌ Failed clients: {failed_clients}")
        
        return successful_results, total_duration
    
    def print_stress_results(self, results, total_duration):
        """Print comprehensive stress test results"""
        if not results:
            print("❌ No successful client results to analyze")
            return
        
        print("\n" + "="*60)
        print("📊 WEBSOCKET PROXY STRESS TEST RESULTS")
        print("="*60)
        
        # Overall statistics
        total_messages = sum(r['messages_received'] for r in results)
        total_clients = len(results)
        
        print(f"\n🔗 CONNECTION STATISTICS:")
        print(f"  Concurrent clients: {total_clients}")
        print(f"  Success rate: {(total_clients/len(results)*100):.1f}%")
        
        connection_times = [r['connection_time_ms'] for r in results if r['connection_time_ms'] > 0]
        if connection_times:
            avg_connection_time = statistics.mean(connection_times)
            max_connection_time = max(connection_times)
            print(f"  Avg connection time: {avg_connection_time:.1f}ms")
            print(f"  Max connection time: {max_connection_time:.1f}ms")
        
        print(f"\n📈 THROUGHPUT STATISTICS:")
        print(f"  Total messages: {total_messages:,}")
        print(f"  Total duration: {total_duration:.1f}s")
        print(f"  Overall rate: {total_messages/total_duration:.1f} msg/sec")
        
        message_rates = [r['message_rate'] for r in results]
        if message_rates:
            avg_rate_per_client = statistics.mean(message_rates)
            max_rate_per_client = max(message_rates)
            print(f"  Avg rate per client: {avg_rate_per_client:.1f} msg/sec")
            print(f"  Max rate per client: {max_rate_per_client:.1f} msg/sec")
        
        print(f"\n⚡ LATENCY STATISTICS:")
        latencies = [r['avg_latency_ms'] for r in results if r['avg_latency_ms'] > 0]
        if latencies:
            overall_avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print(f"  Average latency: {overall_avg_latency:.2f}ms")
            print(f"  Min latency: {min_latency:.2f}ms")
            print(f"  Max latency: {max_latency:.2f}ms")
            
            # Latency distribution
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            if n >= 10:
                p50 = sorted_latencies[n//2]
                p95 = sorted_latencies[int(n*0.95)]
                print(f"  P50 latency: {p50:.2f}ms")
                print(f"  P95 latency: {p95:.2f}ms")
        else:
            print("  No latency data available")
        
        print(f"\n📊 PER-CLIENT BREAKDOWN:")
        print("  Client | Messages | Rate/sec | Latency(ms)")
        print("  -------|----------|----------|------------")
        
        for r in sorted(results, key=lambda x: x['messages_received'], reverse=True)[:10]:
            print(f"  {r['client_id']:6d} | {r['messages_received']:8d} | {r['message_rate']:8.1f} | {r['avg_latency_ms']:10.2f}")
        
        if len(results) > 10:
            print(f"  ... and {len(results)-10} more clients")
        
        # Performance rating
        print(f"\n🎯 PERFORMANCE RATING:")
        
        if total_messages/total_duration > 1000:
            throughput_rating = "🟢 EXCELLENT"
        elif total_messages/total_duration > 500:
            throughput_rating = "🟡 GOOD"
        elif total_messages/total_duration > 100:
            throughput_rating = "🟠 ACCEPTABLE"
        else:
            throughput_rating = "🔴 NEEDS IMPROVEMENT"
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            if avg_latency < 5:
                latency_rating = "🟢 EXCELLENT"
            elif avg_latency < 10:
                latency_rating = "🟡 GOOD"
            elif avg_latency < 50:
                latency_rating = "🟠 ACCEPTABLE"
            else:
                latency_rating = "🔴 NEEDS IMPROVEMENT"
        else:
            latency_rating = "❓ NO DATA"
        
        print(f"  Throughput: {throughput_rating} ({total_messages/total_duration:.1f} msg/sec)")
        print(f"  Latency: {latency_rating} ({statistics.mean(latencies):.2f}ms)" if latencies else f"  Latency: {latency_rating}")
        
        print("="*60)


async def main():
    """Run the stress test"""
    print("🚀 WebSocket Proxy Stress Test")
    print("=" * 40)
    
    stress_test = ProxyStressTest()
    
    # Test configurations
    test_configs = [
        {"clients": 5, "duration": 30},   # Light load
        {"clients": 10, "duration": 30},  # Medium load
        {"clients": 20, "duration": 30},  # Heavy load
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 TEST {i}/{len(test_configs)}: {config['clients']} clients, {config['duration']}s")
        print("-" * 40)
        
        try:
            results, duration = await stress_test.run_concurrent_test(
                num_clients=config['clients'],
                duration_seconds=config['duration']
            )
            
            stress_test.print_stress_results(results, duration)
            
            # Wait between tests
            if i < len(test_configs):
                print(f"\n⏳ Waiting 10 seconds before next test...")
                await asyncio.sleep(10)
                
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 All stress tests completed!")


if __name__ == "__main__":
    asyncio.run(main())