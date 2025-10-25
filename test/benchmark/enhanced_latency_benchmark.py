"""
Enhanced WebSocket Proxy Latency Benchmark
Measures precise end-to-end latency from Broker WSS → Proxy → Client
"""

import sys
import os
import time
import json
import threading
import statistics
from datetime import datetime
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openalgo import api
except ImportError:
    print("Error: Could not import openalgo")
    sys.exit(1)


class EnhancedLatencyBenchmark:
    def __init__(self):
        self.latency_measurements = deque(maxlen=1000)
        self.message_count = 0
        self.start_time = None
        self.lock = threading.Lock()
        
        # Detailed latency tracking
        self.broker_to_proxy_latencies = deque(maxlen=1000)
        self.proxy_to_client_latencies = deque(maxlen=1000)
        self.end_to_end_latencies = deque(maxlen=1000)
        
    def on_data_received(self, data):
        """Enhanced latency measurement with detailed breakdown"""
        client_receive_time = time.time() * 1000  # Convert to milliseconds
        
        with self.lock:
            self.message_count += 1
            
            # Extract timestamps from data
            broker_timestamp = None
            proxy_timestamp = None
            
            if isinstance(data, dict):
                broker_timestamp = data.get('broker_timestamp')
                proxy_timestamp = data.get('proxy_timestamp')
                symbol = data.get('symbol', 'UNKNOWN')
                ltp = data.get('ltp', 0)
                
                # Calculate latencies if timestamps are available
                if broker_timestamp and proxy_timestamp:
                    # Ensure timestamps are in milliseconds
                    if broker_timestamp > 1e12:  # Nanoseconds
                        broker_timestamp = broker_timestamp / 1e6
                    elif broker_timestamp > 1e9:  # Microseconds
                        broker_timestamp = broker_timestamp / 1e3
                    
                    if proxy_timestamp > 1e12:  # Nanoseconds
                        proxy_timestamp = proxy_timestamp / 1e6
                    elif proxy_timestamp > 1e9:  # Microseconds
                        proxy_timestamp = proxy_timestamp / 1e3
                    
                    # Calculate individual latencies
                    broker_to_proxy_ms = proxy_timestamp - broker_timestamp
                    proxy_to_client_ms = client_receive_time - proxy_timestamp
                    end_to_end_ms = client_receive_time - broker_timestamp
                    
                    # Store measurements (only if reasonable values)
                    if 0 <= broker_to_proxy_ms <= 1000:  # Max 1 second
                        self.broker_to_proxy_latencies.append(broker_to_proxy_ms)
                    
                    if 0 <= proxy_to_client_ms <= 1000:  # Max 1 second
                        self.proxy_to_client_latencies.append(proxy_to_client_ms)
                    
                    if 0 <= end_to_end_ms <= 2000:  # Max 2 seconds
                        self.end_to_end_latencies.append(end_to_end_ms)
                    
                    # Print detailed latency info every 25 messages
                    if self.message_count % 25 == 0:
                        elapsed = (time.time() - self.start_time) if self.start_time else 1
                        rate = self.message_count / elapsed
                        
                        print(f"📊 Msg #{self.message_count:3d} | {symbol:8s} | LTP: {ltp:8.2f} | "
                              f"B→P: {broker_to_proxy_ms:5.2f}ms | P→C: {proxy_to_client_ms:5.2f}ms | "
                              f"E2E: {end_to_end_ms:5.2f}ms | Rate: {rate:5.1f}/sec")
                
                elif broker_timestamp:
                    # Only broker timestamp available
                    end_to_end_ms = client_receive_time - broker_timestamp
                    if 0 <= end_to_end_ms <= 2000:
                        self.end_to_end_latencies.append(end_to_end_ms)
                    
                    if self.message_count % 25 == 0:
                        elapsed = (time.time() - self.start_time) if self.start_time else 1
                        rate = self.message_count / elapsed
                        print(f"📊 Msg #{self.message_count:3d} | {symbol:8s} | LTP: {ltp:8.2f} | "
                              f"E2E: {end_to_end_ms:5.2f}ms | Rate: {rate:5.1f}/sec")
                
                else:
                    # No timestamp data
                    if self.message_count % 25 == 0:
                        elapsed = (time.time() - self.start_time) if self.start_time else 1
                        rate = self.message_count / elapsed
                        print(f"📊 Msg #{self.message_count:3d} | {symbol:8s} | LTP: {ltp:8.2f} | "
                              f"No timestamp data | Rate: {rate:5.1f}/sec")
    
    def run_test(self, duration_seconds=120):
        """Run enhanced latency benchmark"""
        print("🚀 Enhanced WebSocket Proxy Latency Benchmark")
        print("=" * 60)
        print("📊 Measuring: Broker WSS → Proxy → Client latency")
        print("⏱️  Timestamps: Broker, Proxy, Client receive times")
        print("📈 Breakdown: Broker→Proxy, Proxy→Client, End-to-End")
        print("=" * 60)
        
        # Test symbols (popular NSE stocks for active data)
        symbols = [
            {"exchange": "NSE", "symbol": "TCS"},
            {"exchange": "NSE", "symbol": "RELIANCE"},
            {"exchange": "NSE", "symbol": "HDFCBANK"},
            {"exchange": "NSE", "symbol": "INFY"},
            {"exchange": "NSE", "symbol": "ICICIBANK"},
            {"exchange": "NSE", "symbol": "SBIN"},
            {"exchange": "NSE", "symbol": "BHARTIARTL"},
            {"exchange": "NSE", "symbol": "ITC"}
        ]
        
        # Initialize client
        client = api(
            api_key="32a3d40880806508fa1397e3b7621a58acff440470d08562863adebdb4f6c194",
            host="http://127.0.0.1:5000",
            ws_url="ws://127.0.0.1:8765"
        )
        
        try:
            print(f"📡 Connecting to WebSocket...")
            client.connect()
            print("✅ Connected!")
            
            print(f"📈 Subscribing to {len(symbols)} symbols...")
            client.subscribe_ltp(symbols, on_data_received=self.on_data_received)
            print("✅ Subscribed!")
            
            print(f"\n🏃 Running {duration_seconds}s latency test...")
            print("📊 Detailed latency breakdown every 25 messages:")
            print("    B→P = Broker to Proxy latency")
            print("    P→C = Proxy to Client latency") 
            print("    E2E = End-to-End latency")
            print("-" * 60)
            
            self.start_time = time.time()
            end_time = self.start_time + duration_seconds
            
            while time.time() < end_time:
                time.sleep(1)
            
            print(f"\n🏁 Test completed!")
            
        except KeyboardInterrupt:
            print(f"\n🛑 Test interrupted")
            
        finally:
            try:
                client.unsubscribe_ltp(symbols)
                client.disconnect()
            except:
                pass
        
        # Print comprehensive results
        self.print_enhanced_results(duration_seconds)
    
    def print_enhanced_results(self, duration):
        """Print comprehensive latency analysis"""
        print("\n" + "="*70)
        print("📊 ENHANCED LATENCY BENCHMARK RESULTS")
        print("="*70)
        
        print(f"⏱️  Test Duration: {duration}s")
        print(f"📨 Total Messages: {self.message_count}")
        
        if duration > 0:
            rate = self.message_count / duration
            print(f"🚀 Average Throughput: {rate:.1f} messages/second")
        
        # End-to-End Latency Analysis
        if self.end_to_end_latencies:
            e2e_latencies = list(self.end_to_end_latencies)
            avg_e2e = statistics.mean(e2e_latencies)
            min_e2e = min(e2e_latencies)
            max_e2e = max(e2e_latencies)
            
            # Calculate percentiles
            sorted_e2e = sorted(e2e_latencies)
            n = len(sorted_e2e)
            p50_e2e = sorted_e2e[n//2] if n > 0 else 0
            p95_e2e = sorted_e2e[int(n*0.95)] if n > 0 else 0
            p99_e2e = sorted_e2e[int(n*0.99)] if n > 0 else 0
            
            print(f"\n⚡ END-TO-END LATENCY (Broker WSS → Client):")
            print(f"  📊 Samples: {len(e2e_latencies)}")
            print(f"  📈 Average: {avg_e2e:.2f}ms")
            print(f"  📊 Median (P50): {p50_e2e:.2f}ms")
            print(f"  📊 P95: {p95_e2e:.2f}ms")
            print(f"  📊 P99: {p99_e2e:.2f}ms")
            print(f"  📊 Min: {min_e2e:.2f}ms")
            print(f"  📊 Max: {max_e2e:.2f}ms")
            
            # Performance rating
            if avg_e2e < 5:
                rating = "🟢 EXCELLENT"
            elif avg_e2e < 10:
                rating = "🟡 GOOD"
            elif avg_e2e < 50:
                rating = "🟠 ACCEPTABLE"
            else:
                rating = "🔴 NEEDS IMPROVEMENT"
            
            print(f"  🎯 Rating: {rating}")
        
        # Broker to Proxy Latency
        if self.broker_to_proxy_latencies:
            b2p_latencies = list(self.broker_to_proxy_latencies)
            avg_b2p = statistics.mean(b2p_latencies)
            min_b2p = min(b2p_latencies)
            max_b2p = max(b2p_latencies)
            
            print(f"\n🔄 BROKER → PROXY LATENCY:")
            print(f"  📊 Samples: {len(b2p_latencies)}")
            print(f"  📈 Average: {avg_b2p:.2f}ms")
            print(f"  📊 Min: {min_b2p:.2f}ms")
            print(f"  📊 Max: {max_b2p:.2f}ms")
        
        # Proxy to Client Latency
        if self.proxy_to_client_latencies:
            p2c_latencies = list(self.proxy_to_client_latencies)
            avg_p2c = statistics.mean(p2c_latencies)
            min_p2c = min(p2c_latencies)
            max_p2c = max(p2c_latencies)
            
            print(f"\n📡 PROXY → CLIENT LATENCY:")
            print(f"  📊 Samples: {len(p2c_latencies)}")
            print(f"  📈 Average: {avg_p2c:.2f}ms")
            print(f"  📊 Min: {min_p2c:.2f}ms")
            print(f"  📊 Max: {max_p2c:.2f}ms")
        
        # Latency Distribution Analysis
        if self.end_to_end_latencies:
            e2e_latencies = list(self.end_to_end_latencies)
            
            # Count latency ranges
            under_1ms = sum(1 for x in e2e_latencies if x < 1)
            under_5ms = sum(1 for x in e2e_latencies if x < 5)
            under_10ms = sum(1 for x in e2e_latencies if x < 10)
            under_50ms = sum(1 for x in e2e_latencies if x < 50)
            over_50ms = sum(1 for x in e2e_latencies if x >= 50)
            
            total = len(e2e_latencies)
            
            print(f"\n📊 LATENCY DISTRIBUTION:")
            print(f"  < 1ms:   {under_1ms:4d} ({under_1ms/total*100:5.1f}%)")
            print(f"  < 5ms:   {under_5ms:4d} ({under_5ms/total*100:5.1f}%)")
            print(f"  < 10ms:  {under_10ms:4d} ({under_10ms/total*100:5.1f}%)")
            print(f"  < 50ms:  {under_50ms:4d} ({under_50ms/total*100:5.1f}%)")
            print(f"  ≥ 50ms:  {over_50ms:4d} ({over_50ms/total*100:5.1f}%)")
        
        # Summary and Recommendations
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        if self.end_to_end_latencies:
            avg_latency = statistics.mean(list(self.end_to_end_latencies))
            if avg_latency < 5:
                print("  ✅ Excellent latency for real-time trading")
                print("  ✅ Suitable for high-frequency strategies")
            elif avg_latency < 10:
                print("  ✅ Good latency for most trading scenarios")
                print("  ✅ Suitable for algorithmic trading")
            elif avg_latency < 50:
                print("  ⚠️  Acceptable for retail trading")
                print("  ⚠️  May need optimization for HFT")
            else:
                print("  ❌ High latency - needs optimization")
                print("  ❌ Not suitable for time-sensitive trading")
        
        if self.message_count > 0 and duration > 0:
            throughput = self.message_count / duration
            if throughput > 10:
                print("  ✅ Good message throughput")
            elif throughput > 5:
                print("  ⚠️  Moderate message throughput")
            else:
                print("  ❌ Low message throughput")
        
        print("="*70)


if __name__ == "__main__":
    benchmark = EnhancedLatencyBenchmark()
    benchmark.run_test(duration_seconds=30)  # 30-second test