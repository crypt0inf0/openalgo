"""
WebSocket Proxy Stress Test
Maximum load testing to find performance limits
"""

import asyncio
import websockets
import json
import time
import statistics
import threading
from collections import deque, defaultdict
from datetime import datetime
import sys


class StressTest:
    def __init__(self, max_symbols=50, test_duration=300):
        self.max_symbols = max_symbols
        self.test_duration = test_duration
        
        # Performance tracking
        self.message_count = 0
        self.error_count = 0
        self.connection_drops = 0
        self.start_time = None
        
        # Latency measurements
        self.latencies = deque(maxlen=10000)
        self.throughput_samples = deque(maxlen=1000)
        
        # Resource monitoring
        self.memory_usage = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # All NSE symbols for stress testing
        self.all_symbols = [
            {"symbol": "TCS", "exchange": "NSE"},
            {"symbol": "RELIANCE", "exchange": "NSE"},
            {"symbol": "HDFCBANK", "exchange": "NSE"},
            {"symbol": "INFY", "exchange": "NSE"},
            {"symbol": "ICICIBANK", "exchange": "NSE"},
            {"symbol": "HINDUNILVR", "exchange": "NSE"},
            {"symbol": "ITC", "exchange": "NSE"},
            {"symbol": "SBIN", "exchange": "NSE"},
            {"symbol": "BHARTIARTL", "exchange": "NSE"},
            {"symbol": "KOTAKBANK", "exchange": "NSE"},
            {"symbol": "LT", "exchange": "NSE"},
            {"symbol": "ASIANPAINT", "exchange": "NSE"},
            {"symbol": "MARUTI", "exchange": "NSE"},
            {"symbol": "AXISBANK", "exchange": "NSE"},
            {"symbol": "TITAN", "exchange": "NSE"},
            {"symbol": "WIPRO", "exchange": "NSE"},
            {"symbol": "ULTRACEMCO", "exchange": "NSE"},
            {"symbol": "NESTLEIND", "exchange": "NSE"},
            {"symbol": "POWERGRID", "exchange": "NSE"},
            {"symbol": "NTPC", "exchange": "NSE"},
            {"symbol": "JSWSTEEL", "exchange": "NSE"},
            {"symbol": "TATAMOTORS", "exchange": "NSE"},
            {"symbol": "BAJFINANCE", "exchange": "NSE"},
            {"symbol": "HCLTECH", "exchange": "NSE"},
            {"symbol": "DRREDDY", "exchange": "NSE"},
            {"symbol": "SUNPHARMA", "exchange": "NSE"},
            {"symbol": "TECHM", "exchange": "NSE"},
            {"symbol": "ONGC", "exchange": "NSE"},
            {"symbol": "COALINDIA", "exchange": "NSE"},
            {"symbol": "INDUSINDBK", "exchange": "NSE"},
            {"symbol": "BAJAJFINSV", "exchange": "NSE"},
            {"symbol": "GRASIM", "exchange": "NSE"},
            {"symbol": "CIPLA", "exchange": "NSE"},
            {"symbol": "EICHERMOT", "exchange": "NSE"},
            {"symbol": "HEROMOTOCO", "exchange": "NSE"},
            {"symbol": "BRITANNIA", "exchange": "NSE"},
            {"symbol": "DIVISLAB", "exchange": "NSE"},
            {"symbol": "APOLLOHOSP", "exchange": "NSE"},
            {"symbol": "ADANIPORTS", "exchange": "NSE"},
            {"symbol": "TATACONSUM", "exchange": "NSE"},
            {"symbol": "BPCL", "exchange": "NSE"},
            {"symbol": "HINDALCO", "exchange": "NSE"},
            {"symbol": "TATASTEEL", "exchange": "NSE"},
            {"symbol": "SHREECEM", "exchange": "NSE"},
            {"symbol": "UPL", "exchange": "NSE"},
            {"symbol": "BAJAJ-AUTO", "exchange": "NSE"},
            {"symbol": "M&M", "exchange": "NSE"},
            {"symbol": "VEDL", "exchange": "NSE"},
            {"symbol": "GODREJCP", "exchange": "NSE"},
            {"symbol": "DABUR", "exchange": "NSE"}
        ][:max_symbols]

    async def run_stress_test(self):
        """Run comprehensive stress test"""
        print("🔥 WEBSOCKET PROXY STRESS TEST")
        print("=" * 60)
        print(f"⚡ Maximum Symbols: {self.max_symbols}")
        print(f"⏱️  Test Duration: {self.test_duration}s ({self.test_duration//60}m {self.test_duration%60}s)")
        print(f"🎯 Goal: Find performance limits and breaking points")
        print("-" * 60)
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(self._monitor_resources())
        
        try:
            # Run progressive load test
            await self._run_progressive_load_test()
            
        except Exception as e:
            print(f"❌ Stress test failed: {e}")
        finally:
            # Stop resource monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Generate stress test report
        self._generate_stress_report()

    async def _run_progressive_load_test(self):
        """Run progressive load test with increasing symbol count"""
        print("\n🚀 Starting Progressive Load Test...")
        
        # Test with increasing symbol counts
        symbol_counts = [5, 10, 15, 20, 25, 30, 40, 50]
        symbol_counts = [count for count in symbol_counts if count <= self.max_symbols]
        
        for symbol_count in symbol_counts:
            print(f"\n📊 Testing with {symbol_count} symbols...")
            
            success = await self._test_symbol_load(symbol_count, 60)  # 1 minute per test
            
            if not success:
                print(f"❌ Failed at {symbol_count} symbols - found breaking point!")
                break
            
            print(f"✅ {symbol_count} symbols handled successfully")
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        # Final endurance test with maximum successful load
        if symbol_counts:
            max_successful = min(symbol_counts[-1], len(self.all_symbols))
            print(f"\n🏃 Running endurance test with {max_successful} symbols for {self.test_duration}s...")
            await self._test_symbol_load(max_successful, self.test_duration)

    async def _test_symbol_load(self, symbol_count, duration):
        """Test specific symbol load"""
        uri = "ws://127.0.0.1:8765"
        test_symbols = self.all_symbols[:symbol_count]
        
        try:
            async with websockets.connect(uri) as websocket:
                # Handle connection message first
                connection_response = await websocket.recv()
                
                # Authenticate
                auth_message = {
                    "action": "authenticate",
                    "api_key": "32a3d40880806508fa1397e3b7621a58acff440470d08562863adebdb4f6c194"
                }
                await websocket.send(json.dumps(auth_message))
                auth_response = await websocket.recv()
                auth_data = json.loads(auth_response)
                
                if auth_data.get("status") != "success":
                    print(f"❌ Authentication failed: {auth_data}")
                    return False
                
                # Subscribe to symbols
                for symbol_info in test_symbols:
                    subscribe_message = {
                        "action": "subscribe",
                        "symbol": symbol_info["symbol"],
                        "exchange": symbol_info["exchange"],
                        "mode": 1
                    }
                    await websocket.send(json.dumps(subscribe_message))
                    
                    # Quick subscription response check
                    try:
                        sub_response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass  # Continue even if no immediate response
                
                # Run test for specified duration
                test_start = time.time()
                test_end = test_start + duration
                local_message_count = 0
                local_error_count = 0
                
                while time.time() < test_end:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        client_receive_time = time.time() * 1000
                        
                        # Process message
                        try:
                            data = json.loads(message)
                            if data.get("type") == "market_data":
                                local_message_count += 1
                                
                                # Calculate latency if possible
                                broker_timestamp = data.get('broker_timestamp')
                                if broker_timestamp:
                                    if broker_timestamp > 1e12:
                                        broker_timestamp = broker_timestamp / 1e6
                                    elif broker_timestamp > 1e9:
                                        broker_timestamp = broker_timestamp / 1e3
                                    
                                    latency = client_receive_time - broker_timestamp
                                    if 0 <= latency <= 1000:
                                        with self.lock:
                                            self.latencies.append(latency)
                                
                        except json.JSONDecodeError:
                            local_error_count += 1
                        
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        local_error_count += 1
                        continue
                
                # Update global counters
                with self.lock:
                    self.message_count += local_message_count
                    self.error_count += local_error_count
                
                # Calculate performance metrics
                actual_duration = time.time() - test_start
                throughput = local_message_count / actual_duration if actual_duration > 0 else 0
                error_rate = (local_error_count / (local_message_count + local_error_count)) * 100 if (local_message_count + local_error_count) > 0 else 0
                
                print(f"  📊 Messages: {local_message_count:,} | Throughput: {throughput:.1f}/sec | Errors: {error_rate:.1f}%")
                
                # Unsubscribe
                for symbol_info in test_symbols:
                    unsubscribe_message = {
                        "action": "unsubscribe",
                        "symbol": symbol_info["symbol"],
                        "exchange": symbol_info["exchange"],
                        "mode": 1
                    }
                    await websocket.send(json.dumps(unsubscribe_message))
                
                return True
                
        except Exception as e:
            print(f"❌ Connection error with {symbol_count} symbols: {e}")
            with self.lock:
                self.connection_drops += 1
            return False

    async def _monitor_resources(self):
        """Monitor system resources during stress test"""
        try:
            import psutil
            
            while True:
                # Get current process info
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                with self.lock:
                    self.memory_usage.append(memory_mb)
                    self.cpu_usage.append(cpu_percent)
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
        except ImportError:
            # psutil not available, skip resource monitoring
            pass
        except asyncio.CancelledError:
            pass

    def _generate_stress_report(self):
        """Generate comprehensive stress test report"""
        print("\n" + "=" * 80)
        print("🔥 STRESS TEST RESULTS")
        print("=" * 80)
        
        # Basic metrics
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"  📨 Total Messages Processed: {self.message_count:,}")
        print(f"  ❌ Total Errors: {self.error_count:,}")
        print(f"  🔌 Connection Drops: {self.connection_drops}")
        
        if self.message_count > 0:
            error_rate = (self.error_count / (self.message_count + self.error_count)) * 100
            print(f"  📊 Error Rate: {error_rate:.2f}%")
        
        # Latency analysis
        if self.latencies:
            latencies = list(self.latencies)
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            p95_latency = sorted_latencies[int(n*0.95)] if n > 0 else 0
            p99_latency = sorted_latencies[int(n*0.99)] if n > 0 else 0
            
            print(f"\n⚡ LATENCY UNDER STRESS:")
            print(f"  📊 Samples: {len(latencies):,}")
            print(f"  📈 Average: {avg_latency:.3f}ms")
            print(f"  📊 P95: {p95_latency:.3f}ms")
            print(f"  📊 P99: {p99_latency:.3f}ms")
            print(f"  📊 Min: {min_latency:.3f}ms")
            print(f"  📊 Max: {max_latency:.3f}ms")
            print(f"  📊 Jitter: {max_latency - min_latency:.3f}ms")
            
            # Performance under stress rating
            if avg_latency < 2:
                stress_rating = "🟢 EXCELLENT - Handles stress very well"
            elif avg_latency < 5:
                stress_rating = "🟡 GOOD - Acceptable under stress"
            elif avg_latency < 10:
                stress_rating = "🟠 FAIR - Shows stress impact"
            else:
                stress_rating = "🔴 POOR - Significant performance degradation"
            
            print(f"  🎯 Stress Performance: {stress_rating}")
        
        # Resource usage
        if self.memory_usage and self.cpu_usage:
            avg_memory = statistics.mean(list(self.memory_usage))
            max_memory = max(self.memory_usage)
            avg_cpu = statistics.mean(list(self.cpu_usage))
            max_cpu = max(self.cpu_usage)
            
            print(f"\n💻 RESOURCE USAGE:")
            print(f"  🧠 Memory - Average: {avg_memory:.1f}MB | Peak: {max_memory:.1f}MB")
            print(f"  ⚡ CPU - Average: {avg_cpu:.1f}% | Peak: {max_cpu:.1f}%")
        
        # Stability analysis
        print(f"\n🔒 STABILITY ANALYSIS:")
        
        if self.connection_drops == 0:
            stability_rating = "🟢 EXCELLENT - No connection issues"
        elif self.connection_drops <= 2:
            stability_rating = "🟡 GOOD - Minor connection issues"
        elif self.connection_drops <= 5:
            stability_rating = "🟠 FAIR - Some stability concerns"
        else:
            stability_rating = "🔴 POOR - Significant stability issues"
        
        print(f"  🎯 Connection Stability: {stability_rating}")
        
        # Recommendations
        print(f"\n💡 STRESS TEST RECOMMENDATIONS:")
        
        if self.latencies:
            avg_latency = statistics.mean(list(self.latencies))
            
            if avg_latency < 2 and self.connection_drops == 0:
                print("  ✅ System handles stress excellently")
                print("  ✅ Ready for high-volume production deployment")
                print("  💡 Consider testing with even higher loads")
            elif avg_latency < 5 and self.connection_drops <= 2:
                print("  ✅ Good stress performance")
                print("  💡 Monitor performance during peak trading hours")
                print("  💡 Consider load balancing for higher volumes")
            else:
                print("  ⚠️  Performance degrades under stress")
                print("  🔧 Optimize system resources and configuration")
                print("  🔧 Consider horizontal scaling for high loads")
        
        # Maximum capacity estimate
        if self.message_count > 0 and self.test_duration > 0:
            avg_throughput = self.message_count / self.test_duration
            estimated_daily_capacity = avg_throughput * 86400  # 24 hours
            
            print(f"\n📊 CAPACITY ESTIMATES:")
            print(f"  🚀 Sustained Throughput: {avg_throughput:.1f} messages/second")
            print(f"  📅 Estimated Daily Capacity: {estimated_daily_capacity:,.0f} messages")
        
        print("=" * 80)


async def main():
    """Main stress test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WebSocket Proxy Stress Test')
    parser.add_argument('--symbols', type=int, default=30, help='Maximum symbols to test (default: 30)')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds (default: 300)')
    
    args = parser.parse_args()
    
    stress_test = StressTest(
        max_symbols=args.symbols,
        test_duration=args.duration
    )
    
    await stress_test.run_stress_test()
    
    print("\n🏁 Stress test completed!")


if __name__ == "__main__":
    asyncio.run(main())