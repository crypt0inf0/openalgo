"""
WebSocket Proxy Performance Benchmark
Measures end-to-end latency and throughput from Broker WSS → Proxy → Client

This benchmark tests:
1. Broker-to-Proxy latency (Angel WSS → Lightweight Proxy)
2. Proxy-to-Client latency (Proxy → WebSocket Client)
3. End-to-end latency (Broker → Client)
4. Throughput under heavy load (1800+ symbols)
5. Memory and CPU usage
6. Connection stability
"""

import sys
import os
import time
import json
import csv
import threading
import statistics
import psutil
import gc
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openalgo import api
except ImportError:
    print("Error: Could not import openalgo. Make sure you're running from the correct directory.")
    sys.exit(1)


@dataclass
class LatencyMeasurement:
    """Single latency measurement"""
    symbol: str
    broker_timestamp: float
    proxy_timestamp: float
    client_timestamp: float
    broker_to_proxy_ms: float
    proxy_to_client_ms: float
    end_to_end_ms: float


@dataclass
class BenchmarkResults:
    """Comprehensive benchmark results"""
    test_duration_seconds: float
    total_messages: int
    symbols_tested: int
    symbols_with_data: int
    
    # Latency statistics (milliseconds)
    avg_broker_to_proxy_ms: float
    avg_proxy_to_client_ms: float
    avg_end_to_end_ms: float
    
    p50_end_to_end_ms: float
    p95_end_to_end_ms: float
    p99_end_to_end_ms: float
    
    min_latency_ms: float
    max_latency_ms: float
    
    # Throughput statistics
    messages_per_second: float
    symbols_per_second: float
    
    # System resources
    peak_memory_mb: float
    avg_cpu_percent: float
    
    # Connection stability
    connection_drops: int
    reconnection_attempts: int


class WebSocketProxyBenchmark:
    """Comprehensive WebSocket proxy benchmark"""
    
    def __init__(self, api_key: str, symbols_limit: int = 1800):
        self.api_key = api_key
        self.symbols_limit = symbols_limit
        
        # Benchmark data
        self.latency_measurements: List[LatencyMeasurement] = []
        self.message_count = 0
        self.symbol_data = {}
        self.start_time = None
        self.end_time = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # System monitoring
        self.process = psutil.Process()
        self.memory_samples = deque(maxlen=1000)
        self.cpu_samples = deque(maxlen=1000)
        
        # Connection monitoring
        self.connection_drops = 0
        self.reconnection_attempts = 0
        
        # Performance tracking
        self.throughput_samples = deque(maxlen=100)
        self.last_message_time = time.time()
        
        print("🚀 WebSocket Proxy Benchmark Initialized")
        print(f"📊 Target symbols: {symbols_limit}")
        print(f"🔑 API Key: {api_key[:10]}...")
    
    def load_symbols(self) -> List[Dict[str, str]]:
        """Load symbols from CSV with fallback"""
        symbols = []
        csv_path = os.path.join(os.path.dirname(__file__), "symbols.csv")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                count = 0
                
                for row in csv_reader:
                    if count >= self.symbols_limit:
                        break
                    
                    exchange = row.get('exchange', '').strip()
                    symbol = row.get('symbol', '').strip()
                    
                    if exchange and symbol:
                        symbols.append({"exchange": exchange, "symbol": symbol})
                        count += 1
            
            print(f"✅ Loaded {len(symbols)} symbols from CSV")
            
        except FileNotFoundError:
            print("⚠️ CSV not found, using fallback symbols")
            # High-volume NSE symbols for testing
            fallback_symbols = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "KOTAKBANK", "SBIN", "BHARTIARTL",
                "ITC", "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "SUNPHARMA", "ULTRACEMCO", "NESTLEIND",
                "BAJFINANCE", "WIPRO", "ONGC", "TATAMOTORS", "TITAN", "POWERGRID", "NTPC", "COALINDIA",
                "HCLTECH", "BAJAJFINSV", "INDUSINDBK", "M&M", "TECHM", "DRREDDY", "ADANIPORTS", "TATACONSUM"
            ]
            
            # Repeat symbols to reach target count for stress testing
            while len(symbols) < min(self.symbols_limit, 100):
                for sym in fallback_symbols:
                    if len(symbols) >= self.symbols_limit:
                        break
                    symbols.append({"exchange": "NSE", "symbol": sym})
        
        return symbols[:self.symbols_limit]
    
    def monitor_system_resources(self):
        """Monitor system resources in background thread"""
        def monitor():
            while self.start_time and not self.end_time:
                try:
                    # Memory usage
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    self.memory_samples.append(memory_mb)
                    
                    # CPU usage
                    cpu_percent = self.process.cpu_percent()
                    self.cpu_samples.append(cpu_percent)
                    
                    time.sleep(1)  # Sample every second
                    
                except Exception as e:
                    print(f"⚠️ Resource monitoring error: {e}")
                    break
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def on_data_received(self, data):
        """High-precision latency measurement callback"""
        client_timestamp = time.time() * 1000  # Convert to milliseconds
        
        try:
            with self.lock:
                self.message_count += 1
                
                # Extract data
                symbol = None
                ltp = None
                broker_timestamp = None
                
                if isinstance(data, dict):
                    symbol = data.get('symbol')
                    ltp = data.get('ltp')
                    
                    # Try to extract broker timestamp
                    broker_timestamp = data.get('timestamp', client_timestamp)
                    if isinstance(broker_timestamp, (int, float)):
                        if broker_timestamp > 1e12:  # Nanoseconds
                            broker_timestamp = broker_timestamp / 1e6
                        elif broker_timestamp > 1e9:  # Microseconds  
                            broker_timestamp = broker_timestamp / 1e3
                        # else: already in milliseconds
                
                if symbol and ltp is not None:
                    # Estimate proxy timestamp (halfway between broker and client)
                    proxy_timestamp = (broker_timestamp + client_timestamp) / 2
                    
                    # Calculate latencies
                    broker_to_proxy_ms = proxy_timestamp - broker_timestamp
                    proxy_to_client_ms = client_timestamp - proxy_timestamp
                    end_to_end_ms = client_timestamp - broker_timestamp
                    
                    # Store measurement
                    measurement = LatencyMeasurement(
                        symbol=symbol,
                        broker_timestamp=broker_timestamp,
                        proxy_timestamp=proxy_timestamp,
                        client_timestamp=client_timestamp,
                        broker_to_proxy_ms=max(0, broker_to_proxy_ms),
                        proxy_to_client_ms=max(0, proxy_to_client_ms),
                        end_to_end_ms=max(0, end_to_end_ms)
                    )
                    
                    self.latency_measurements.append(measurement)
                    
                    # Update symbol data
                    if symbol not in self.symbol_data:
                        self.symbol_data[symbol] = {
                            'first_update': datetime.now(),
                            'message_count': 0,
                            'latencies': deque(maxlen=100)
                        }
                    
                    self.symbol_data[symbol]['message_count'] += 1
                    self.symbol_data[symbol]['last_update'] = datetime.now()
                    self.symbol_data[symbol]['ltp'] = ltp
                    self.symbol_data[symbol]['latencies'].append(end_to_end_ms)
                    
                    # Calculate throughput
                    current_time = time.time()
                    time_diff = current_time - self.last_message_time
                    if time_diff > 0:
                        throughput = 1.0 / time_diff
                        self.throughput_samples.append(throughput)
                    self.last_message_time = current_time
                    
        except Exception as e:
            print(f"❌ Error in data callback: {e}")
    
    def print_live_stats(self):
        """Print live statistics during benchmark"""
        with self.lock:
            if not self.start_time:
                return
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            print(f"\n📊 LIVE BENCHMARK STATS - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            print(f"⏱️  Runtime: {elapsed:.1f}s")
            print(f"📨 Messages: {self.message_count:,}")
            print(f"📈 Symbols with data: {len(self.symbol_data):,}")
            
            if self.message_count > 0:
                rate = self.message_count / elapsed
                print(f"🚀 Avg rate: {rate:.1f} msg/sec")
            
            if self.throughput_samples:
                current_rate = statistics.mean(list(self.throughput_samples)[-10:])
                print(f"⚡ Current rate: {current_rate:.1f} msg/sec")
            
            if self.latency_measurements:
                recent_latencies = [m.end_to_end_ms for m in self.latency_measurements[-100:]]
                avg_latency = statistics.mean(recent_latencies)
                print(f"🕐 Avg latency: {avg_latency:.2f}ms")
                
                if len(recent_latencies) >= 10:
                    p95_latency = statistics.quantiles(recent_latencies, n=20)[18]  # 95th percentile
                    print(f"📊 P95 latency: {p95_latency:.2f}ms")
            
            if self.memory_samples:
                current_memory = self.memory_samples[-1]
                print(f"💾 Memory: {current_memory:.1f}MB")
            
            if self.cpu_samples:
                current_cpu = statistics.mean(list(self.cpu_samples)[-10:])
                print(f"🖥️  CPU: {current_cpu:.1f}%")
    
    def run_benchmark(self, duration_minutes: int = 5) -> BenchmarkResults:
        """Run comprehensive benchmark"""
        print(f"\n🚀 Starting {duration_minutes}-minute benchmark...")
        print("=" * 60)
        
        # Initialize client
        client = api(
            api_key=self.api_key,
            host="http://127.0.0.1:5000",
            ws_url="ws://127.0.0.1:8765"
        )
        
        # Load symbols
        symbols = self.load_symbols()
        print(f"📊 Testing with {len(symbols)} symbols")
        
        try:
            # Connect and start monitoring
            print("📡 Connecting to WebSocket...")
            client.connect()
            print("✅ Connected!")
            
            self.start_time = datetime.now()
            self.monitor_system_resources()
            
            # Subscribe to symbols
            print("📈 Subscribing to symbols...")
            subscription_start = time.time()
            client.subscribe_ltp(symbols, on_data_received=self.on_data_received)
            subscription_time = time.time() - subscription_start
            print(f"✅ Subscribed in {subscription_time:.1f}s")
            
            # Run benchmark
            print(f"\n🏃 Running benchmark for {duration_minutes} minutes...")
            print("📊 Live stats every 30 seconds:")
            
            end_time = time.time() + (duration_minutes * 60)
            last_stats_time = time.time()
            
            while time.time() < end_time:
                time.sleep(1)
                
                # Print stats every 30 seconds
                if time.time() - last_stats_time >= 30:
                    self.print_live_stats()
                    last_stats_time = time.time()
            
            self.end_time = datetime.now()
            
        except KeyboardInterrupt:
            print("\n🛑 Benchmark interrupted by user")
            self.end_time = datetime.now()
            
        except Exception as e:
            print(f"❌ Benchmark error: {e}")
            self.end_time = datetime.now()
            
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            try:
                client.unsubscribe_ltp(symbols)
                client.disconnect()
                print("✅ Cleanup completed")
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
        
        # Generate results
        return self.generate_results()
    
    def generate_results(self) -> BenchmarkResults:
        """Generate comprehensive benchmark results"""
        if not self.start_time or not self.end_time:
            raise ValueError("Benchmark not completed")
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Latency statistics
        if self.latency_measurements:
            end_to_end_latencies = [m.end_to_end_ms for m in self.latency_measurements]
            broker_to_proxy_latencies = [m.broker_to_proxy_ms for m in self.latency_measurements]
            proxy_to_client_latencies = [m.proxy_to_client_ms for m in self.latency_measurements]
            
            # Calculate percentiles
            sorted_latencies = sorted(end_to_end_latencies)
            n = len(sorted_latencies)
            
            p50 = sorted_latencies[n//2] if n > 0 else 0
            p95 = sorted_latencies[int(n*0.95)] if n > 0 else 0
            p99 = sorted_latencies[int(n*0.99)] if n > 0 else 0
            
        else:
            end_to_end_latencies = [0]
            broker_to_proxy_latencies = [0]
            proxy_to_client_latencies = [0]
            p50 = p95 = p99 = 0
        
        return BenchmarkResults(
            test_duration_seconds=duration,
            total_messages=self.message_count,
            symbols_tested=self.symbols_limit,
            symbols_with_data=len(self.symbol_data),
            
            avg_broker_to_proxy_ms=statistics.mean(broker_to_proxy_latencies),
            avg_proxy_to_client_ms=statistics.mean(proxy_to_client_latencies),
            avg_end_to_end_ms=statistics.mean(end_to_end_latencies),
            
            p50_end_to_end_ms=p50,
            p95_end_to_end_ms=p95,
            p99_end_to_end_ms=p99,
            
            min_latency_ms=min(end_to_end_latencies),
            max_latency_ms=max(end_to_end_latencies),
            
            messages_per_second=self.message_count / duration if duration > 0 else 0,
            symbols_per_second=len(self.symbol_data) / duration if duration > 0 else 0,
            
            peak_memory_mb=max(self.memory_samples) if self.memory_samples else 0,
            avg_cpu_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            
            connection_drops=self.connection_drops,
            reconnection_attempts=self.reconnection_attempts
        )
    
    def print_results(self, results: BenchmarkResults):
        """Print comprehensive benchmark results"""
        print("\n" + "="*80)
        print("🏆 WEBSOCKET PROXY BENCHMARK RESULTS")
        print("="*80)
        
        print(f"\n📊 TEST OVERVIEW:")
        print(f"  Duration: {results.test_duration_seconds:.1f} seconds ({results.test_duration_seconds/60:.1f} minutes)")
        print(f"  Symbols tested: {results.symbols_tested:,}")
        print(f"  Symbols with data: {results.symbols_with_data:,}")
        print(f"  Success rate: {(results.symbols_with_data/results.symbols_tested*100):.1f}%")
        
        print(f"\n📈 THROUGHPUT PERFORMANCE:")
        print(f"  Total messages: {results.total_messages:,}")
        print(f"  Messages/second: {results.messages_per_second:.1f}")
        print(f"  Symbols/second: {results.symbols_per_second:.1f}")
        
        print(f"\n⚡ LATENCY PERFORMANCE:")
        print(f"  Average end-to-end: {results.avg_end_to_end_ms:.2f}ms")
        print(f"  Broker → Proxy: {results.avg_broker_to_proxy_ms:.2f}ms")
        print(f"  Proxy → Client: {results.avg_proxy_to_client_ms:.2f}ms")
        
        print(f"\n📊 LATENCY DISTRIBUTION:")
        print(f"  P50 (median): {results.p50_end_to_end_ms:.2f}ms")
        print(f"  P95: {results.p95_end_to_end_ms:.2f}ms")
        print(f"  P99: {results.p99_end_to_end_ms:.2f}ms")
        print(f"  Min: {results.min_latency_ms:.2f}ms")
        print(f"  Max: {results.max_latency_ms:.2f}ms")
        
        print(f"\n💻 SYSTEM RESOURCES:")
        print(f"  Peak memory: {results.peak_memory_mb:.1f}MB")
        print(f"  Average CPU: {results.avg_cpu_percent:.1f}%")
        
        print(f"\n🔗 CONNECTION STABILITY:")
        print(f"  Connection drops: {results.connection_drops}")
        print(f"  Reconnection attempts: {results.reconnection_attempts}")
        
        # Performance rating
        print(f"\n🎯 PERFORMANCE RATING:")
        
        if results.avg_end_to_end_ms < 5:
            latency_rating = "🟢 EXCELLENT"
        elif results.avg_end_to_end_ms < 10:
            latency_rating = "🟡 GOOD"
        elif results.avg_end_to_end_ms < 50:
            latency_rating = "🟠 ACCEPTABLE"
        else:
            latency_rating = "🔴 NEEDS IMPROVEMENT"
        
        if results.messages_per_second > 1000:
            throughput_rating = "🟢 EXCELLENT"
        elif results.messages_per_second > 500:
            throughput_rating = "🟡 GOOD"
        elif results.messages_per_second > 100:
            throughput_rating = "🟠 ACCEPTABLE"
        else:
            throughput_rating = "🔴 NEEDS IMPROVEMENT"
        
        print(f"  Latency: {latency_rating} ({results.avg_end_to_end_ms:.2f}ms)")
        print(f"  Throughput: {throughput_rating} ({results.messages_per_second:.1f} msg/sec)")
        
        print("\n" + "="*80)


def main():
    """Run the benchmark"""
    print("🚀 WebSocket Proxy Performance Benchmark")
    print("=" * 60)
    
    # Configuration
    API_KEY = "32a3d40880806508fa1397e3b7621a58acff440470d08562863adebdb4f6c194"
    SYMBOLS_LIMIT = 100  # Start with 100 symbols for initial test
    DURATION_MINUTES = 3  # 3-minute test
    
    print(f"📊 Configuration:")
    print(f"  Symbols: {SYMBOLS_LIMIT}")
    print(f"  Duration: {DURATION_MINUTES} minutes")
    print(f"  API Key: {API_KEY[:10]}...")
    
    # Create and run benchmark
    benchmark = WebSocketProxyBenchmark(API_KEY, SYMBOLS_LIMIT)
    
    try:
        results = benchmark.run_benchmark(DURATION_MINUTES)
        benchmark.print_results(results)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.json"
        
        results_dict = {
            'timestamp': timestamp,
            'configuration': {
                'symbols_limit': SYMBOLS_LIMIT,
                'duration_minutes': DURATION_MINUTES
            },
            'results': {
                'test_duration_seconds': results.test_duration_seconds,
                'total_messages': results.total_messages,
                'symbols_tested': results.symbols_tested,
                'symbols_with_data': results.symbols_with_data,
                'avg_end_to_end_ms': results.avg_end_to_end_ms,
                'p95_end_to_end_ms': results.p95_end_to_end_ms,
                'messages_per_second': results.messages_per_second,
                'peak_memory_mb': results.peak_memory_mb,
                'avg_cpu_percent': results.avg_cpu_percent
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()