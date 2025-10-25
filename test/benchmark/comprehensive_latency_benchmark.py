"""
Comprehensive Latency Benchmark Suite
Advanced WebSocket proxy performance testing with detailed analytics
"""

import asyncio
import websockets
import json
import time
import statistics
import threading
from collections import deque, defaultdict
from datetime import datetime, timedelta
import sys
import argparse


class LatencyBenchmark:
    def __init__(self, test_duration=60, symbols_count=10):
        self.test_duration = test_duration
        self.symbols_count = symbols_count
        
        # Performance metrics
        self.message_count = 0
        self.start_time = None
        self.end_time = None
        
        # Latency measurements
        self.end_to_end_latencies = deque(maxlen=10000)
        self.broker_to_proxy_latencies = deque(maxlen=10000)
        self.proxy_to_client_latencies = deque(maxlen=10000)
        
        # Per-symbol tracking
        self.symbol_stats = defaultdict(lambda: {
            'message_count': 0,
            'latencies': deque(maxlen=1000),
            'first_message': None,
            'last_message': None,
            'total_bytes': 0
        })
        
        # Throughput tracking
        self.throughput_samples = deque(maxlen=1000)
        self.last_message_time = None
        
        # Connection quality metrics
        self.connection_drops = 0
        self.reconnection_attempts = 0
        self.message_gaps = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Test symbols (popular NSE stocks)
        self.test_symbols = [
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
            {"symbol": "TITAN", "exchange": "NSE"}
        ][:symbols_count]

    async def run_benchmark(self):
        """Run comprehensive latency benchmark"""
        print("🚀 COMPREHENSIVE LATENCY BENCHMARK")
        print("=" * 60)
        print(f"⏱️  Test Duration: {self.test_duration}s")
        print(f"📊 Symbols Count: {len(self.test_symbols)}")
        print(f"🎯 Target: Sub-millisecond latency measurement")
        print("-" * 60)
        
        uri = "ws://127.0.0.1:8765"
        
        try:
            async with websockets.connect(uri) as websocket:
                print("✅ Connected to WebSocket proxy")
                
                # Authenticate
                await self._authenticate(websocket)
                
                # Subscribe to symbols
                await self._subscribe_symbols(websocket)
                
                # Run benchmark
                await self._run_measurement_loop(websocket)
                
                # Unsubscribe
                await self._unsubscribe_symbols(websocket)
                
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return False
        
        # Generate comprehensive report
        self._generate_report()
        return True

    async def _authenticate(self, websocket):
        """Authenticate with the WebSocket proxy"""
        # First message is connection confirmation
        connection_response = await websocket.recv()
        connection_data = json.loads(connection_response)
        print(f"🔗 Connected: {connection_data.get('type', 'unknown')}")
        
        # Send authentication
        auth_message = {
            "action": "authenticate",
            "api_key": "32a3d40880806508fa1397e3b7621a58acff440470d08562863adebdb4f6c194"
        }
        await websocket.send(json.dumps(auth_message))
        
        # Wait for auth response
        auth_response = await websocket.recv()
        auth_data = json.loads(auth_response)
        
        if auth_data.get("status") != "success":
            raise Exception(f"Authentication failed: {auth_data}")
        
        print("🔐 Authentication successful")

    async def _subscribe_symbols(self, websocket):
        """Subscribe to test symbols"""
        print(f"📈 Subscribing to {len(self.test_symbols)} symbols...")
        
        for symbol_info in self.test_symbols:
            subscribe_message = {
                "action": "subscribe",
                "symbol": symbol_info["symbol"],
                "exchange": symbol_info["exchange"],
                "mode": 1
            }
            await websocket.send(json.dumps(subscribe_message))
            
            # Wait for subscription response
            sub_response = await websocket.recv()
            sub_data = json.loads(sub_response)
            
            if sub_data.get("status") == "success":
                print(f"  ✅ {symbol_info['symbol']}")
            else:
                print(f"  ❌ {symbol_info['symbol']}: {sub_data}")

    async def _unsubscribe_symbols(self, websocket):
        """Unsubscribe from test symbols"""
        print("📉 Unsubscribing from symbols...")
        
        for symbol_info in self.test_symbols:
            unsubscribe_message = {
                "action": "unsubscribe",
                "symbol": symbol_info["symbol"],
                "exchange": symbol_info["exchange"],
                "mode": 1
            }
            await websocket.send(json.dumps(unsubscribe_message))

    async def _run_measurement_loop(self, websocket):
        """Main measurement loop"""
        print(f"\n🏃 Starting {self.test_duration}s benchmark...")
        print("📊 Real-time latency monitoring:")
        print("-" * 80)
        
        self.start_time = time.time()
        end_time = self.start_time + self.test_duration
        
        while time.time() < end_time:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                client_receive_time = time.time() * 1000  # milliseconds
                
                # Process message
                await self._process_message(message, client_receive_time)
                
            except asyncio.TimeoutError:
                continue  # No message received, continue
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                self.connection_drops += 1
                continue
        
        self.end_time = time.time()
        print(f"\n🏁 Benchmark completed!")

    async def _process_message(self, message, client_receive_time):
        """Process individual WebSocket message"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        
        # Only process market data messages
        if data.get("type") != "market_data":
            return
        
        with self.lock:
            self.message_count += 1
            
            # Extract message data
            symbol = data.get('symbol', 'UNKNOWN')
            broker_timestamp = data.get('broker_timestamp')
            proxy_timestamp = data.get('proxy_timestamp')
            
            # Extract LTP from nested data
            ltp = 0
            if 'data' in data and isinstance(data['data'], dict):
                ltp = data['data'].get('ltp', 0)
            
            # Calculate message size
            message_size = len(message.encode('utf-8'))
            
            # Update symbol stats
            symbol_stat = self.symbol_stats[symbol]
            symbol_stat['message_count'] += 1
            symbol_stat['total_bytes'] += message_size
            symbol_stat['last_message'] = datetime.now()
            if symbol_stat['first_message'] is None:
                symbol_stat['first_message'] = datetime.now()
            
            # Calculate latencies if timestamps are available
            if broker_timestamp and proxy_timestamp:
                # Timestamps are already in milliseconds with decimal precision
                # No conversion needed - they're already in the correct format
                
                # Calculate individual latencies
                broker_to_proxy_ms = proxy_timestamp - broker_timestamp
                proxy_to_client_ms = client_receive_time - proxy_timestamp
                end_to_end_ms = client_receive_time - broker_timestamp
                
                # Store measurements (only if reasonable values)
                if 0 <= broker_to_proxy_ms <= 1000:
                    self.broker_to_proxy_latencies.append(broker_to_proxy_ms)
                
                if 0 <= proxy_to_client_ms <= 1000:
                    self.proxy_to_client_latencies.append(proxy_to_client_ms)
                
                if 0 <= end_to_end_ms <= 2000:
                    self.end_to_end_latencies.append(end_to_end_ms)
                    symbol_stat['latencies'].append(end_to_end_ms)
                
                # Calculate throughput
                current_time = time.time()
                if self.last_message_time:
                    time_diff = current_time - self.last_message_time
                    if time_diff > 0:
                        throughput = 1.0 / time_diff
                        self.throughput_samples.append(throughput)
                self.last_message_time = current_time
                
                # Print progress every 25 messages
                if self.message_count % 25 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.message_count / elapsed
                    
                    print(f"📊 Msg #{self.message_count:4d} | {symbol:10s} | LTP: {ltp:8.2f} | "
                          f"B→P: {broker_to_proxy_ms:5.2f}ms | P→C: {proxy_to_client_ms:5.2f}ms | "
                          f"E2E: {end_to_end_ms:5.2f}ms | Rate: {rate:5.1f}/sec")

    def _generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE LATENCY BENCHMARK RESULTS")
        print("=" * 80)
        
        # Test overview
        actual_duration = self.end_time - self.start_time if self.end_time else self.test_duration
        print(f"⏱️  Test Duration: {actual_duration:.1f}s")
        print(f"📨 Total Messages: {self.message_count:,}")
        print(f"📊 Symbols Tested: {len(self.test_symbols)}")
        
        if actual_duration > 0:
            avg_throughput = self.message_count / actual_duration
            print(f"🚀 Average Throughput: {avg_throughput:.1f} messages/second")
        
        # Connection quality
        print(f"\n🔗 CONNECTION QUALITY:")
        print(f"  📡 Connection Drops: {self.connection_drops}")
        print(f"  🔄 Reconnection Attempts: {self.reconnection_attempts}")
        print(f"  ⚠️  Message Gaps: {self.message_gaps}")
        
        # Latency analysis
        self._print_latency_analysis()
        
        # Per-symbol analysis
        self._print_symbol_analysis()
        
        # Throughput analysis
        self._print_throughput_analysis()
        
        # Performance rating
        self._print_performance_rating()

    def _print_latency_analysis(self):
        """Print detailed latency analysis"""
        print(f"\n⚡ LATENCY ANALYSIS:")
        
        if self.end_to_end_latencies:
            latencies = list(self.end_to_end_latencies)
            
            # Basic statistics
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
            
            # Percentiles
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            p50 = sorted_latencies[n//2] if n > 0 else 0
            p95 = sorted_latencies[int(n*0.95)] if n > 0 else 0
            p99 = sorted_latencies[int(n*0.99)] if n > 0 else 0
            p999 = sorted_latencies[int(n*0.999)] if n > 0 else 0
            
            print(f"  📊 End-to-End Latency (Broker → Client):")
            print(f"    📈 Samples: {len(latencies):,}")
            print(f"    📊 Average: {avg_latency:.3f}ms")
            print(f"    📊 Median (P50): {p50:.3f}ms")
            print(f"    📊 P95: {p95:.3f}ms")
            print(f"    📊 P99: {p99:.3f}ms")
            print(f"    📊 P99.9: {p999:.3f}ms")
            print(f"    📊 Min: {min_latency:.3f}ms")
            print(f"    📊 Max: {max_latency:.3f}ms")
            print(f"    📊 Std Dev: {std_dev:.3f}ms")
            
            # Jitter analysis
            jitter = max_latency - min_latency
            print(f"    📊 Jitter: {jitter:.3f}ms")
        
        # Component latencies
        if self.broker_to_proxy_latencies:
            b2p_avg = statistics.mean(list(self.broker_to_proxy_latencies))
            print(f"  🔄 Broker → Proxy: {b2p_avg:.3f}ms average")
        
        if self.proxy_to_client_latencies:
            p2c_avg = statistics.mean(list(self.proxy_to_client_latencies))
            print(f"  📡 Proxy → Client: {p2c_avg:.3f}ms average")

    def _print_symbol_analysis(self):
        """Print per-symbol performance analysis"""
        print(f"\n📈 PER-SYMBOL ANALYSIS:")
        
        if not self.symbol_stats:
            print("  ⚠️  No symbol data collected")
            return
        
        print(f"  {'Symbol':<12} {'Messages':<10} {'Avg Latency':<12} {'Data Rate':<12}")
        print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
        
        for symbol, stats in sorted(self.symbol_stats.items()):
            msg_count = stats['message_count']
            
            if stats['latencies']:
                avg_latency = statistics.mean(list(stats['latencies']))
                latency_str = f"{avg_latency:.3f}ms"
            else:
                latency_str = "N/A"
            
            # Calculate data rate
            if stats['first_message'] and stats['last_message']:
                duration = (stats['last_message'] - stats['first_message']).total_seconds()
                if duration > 0:
                    rate = msg_count / duration
                    rate_str = f"{rate:.1f}/sec"
                else:
                    rate_str = "N/A"
            else:
                rate_str = "N/A"
            
            print(f"  {symbol:<12} {msg_count:<10} {latency_str:<12} {rate_str:<12}")

    def _print_throughput_analysis(self):
        """Print throughput analysis"""
        print(f"\n🚀 THROUGHPUT ANALYSIS:")
        
        if self.throughput_samples:
            throughput_list = list(self.throughput_samples)
            avg_throughput = statistics.mean(throughput_list)
            max_throughput = max(throughput_list)
            min_throughput = min(throughput_list)
            
            print(f"  📊 Average Throughput: {avg_throughput:.1f} msg/sec")
            print(f"  📊 Peak Throughput: {max_throughput:.1f} msg/sec")
            print(f"  📊 Min Throughput: {min_throughput:.1f} msg/sec")
            
            # Calculate total data processed
            total_bytes = sum(stats['total_bytes'] for stats in self.symbol_stats.values())
            if total_bytes > 0:
                total_mb = total_bytes / (1024 * 1024)
                actual_duration = self.end_time - self.start_time if self.end_time else self.test_duration
                if actual_duration > 0:
                    data_rate_mbps = total_mb / actual_duration
                    print(f"  📊 Data Rate: {data_rate_mbps:.2f} MB/sec")

    def _print_performance_rating(self):
        """Print overall performance rating"""
        print(f"\n🎯 PERFORMANCE RATING:")
        
        if not self.end_to_end_latencies:
            print("  ⚠️  No latency data for rating")
            return
        
        avg_latency = statistics.mean(list(self.end_to_end_latencies))
        
        if avg_latency < 1:
            rating = "🟢 EXCELLENT"
            description = "Perfect for high-frequency trading"
        elif avg_latency < 5:
            rating = "🟡 VERY GOOD"
            description = "Suitable for algorithmic trading"
        elif avg_latency < 10:
            rating = "🟠 GOOD"
            description = "Good for retail trading"
        elif avg_latency < 50:
            rating = "🔴 ACCEPTABLE"
            description = "Acceptable for basic trading"
        else:
            rating = "⚫ POOR"
            description = "Needs optimization"
        
        print(f"  🏆 Overall Rating: {rating}")
        print(f"  📝 Assessment: {description}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if avg_latency < 1:
            print("  ✅ System is optimally configured for trading")
            print("  ✅ Ready for production deployment")
        elif avg_latency < 5:
            print("  💡 Consider optimizing network configuration")
            print("  💡 Monitor for consistency during peak hours")
        else:
            print("  ⚠️  Review system architecture for bottlenecks")
            print("  ⚠️  Consider hardware upgrades or network optimization")
        
        print("=" * 80)


async def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description='Comprehensive WebSocket Latency Benchmark')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds (default: 60)')
    parser.add_argument('--symbols', type=int, default=10, help='Number of symbols to test (default: 10)')
    
    args = parser.parse_args()
    
    benchmark = LatencyBenchmark(
        test_duration=args.duration,
        symbols_count=args.symbols
    )
    
    success = await benchmark.run_benchmark()
    
    if success:
        print("\n✅ Benchmark completed successfully!")
        return 0
    else:
        print("\n❌ Benchmark failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)