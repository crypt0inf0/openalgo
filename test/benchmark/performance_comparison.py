"""
Performance Comparison Tool
Compare WebSocket proxy performance across different configurations
"""

import asyncio
import json
import time
import statistics
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class PerformanceComparison:
    def __init__(self):
        self.test_results = []
        self.comparison_data = {}
    
    async def run_comparison_suite(self):
        """Run comprehensive performance comparison"""
        print("🔬 WEBSOCKET PROXY PERFORMANCE COMPARISON")
        print("=" * 60)
        
        # Test configurations
        test_configs = [
            {
                'name': 'Light Load',
                'symbols': 3,
                'duration': 30,
                'description': 'Baseline performance with minimal load'
            },
            {
                'name': 'Medium Load',
                'symbols': 8,
                'duration': 60,
                'description': 'Typical trading session load'
            },
            {
                'name': 'Heavy Load',
                'symbols': 15,
                'duration': 60,
                'description': 'High-frequency trading simulation'
            },
            {
                'name': 'Stress Test',
                'symbols': 20,
                'duration': 90,
                'description': 'Maximum capacity stress test'
            }
        ]
        
        for config in test_configs:
            print(f"\n🧪 Testing: {config['name']}")
            print(f"📝 {config['description']}")
            
            result = await self._run_single_test(
                config['name'],
                config['symbols'],
                config['duration']
            )
            
            if result:
                self.test_results.append(result)
                print(f"✅ {config['name']} completed")
            else:
                print(f"❌ {config['name']} failed")
        
        # Generate comparison report
        self._generate_comparison_report()
        
        # Create performance charts
        self._create_performance_charts()
    
    async def _run_single_test(self, test_name, symbols_count, duration):
        """Run a single performance test"""
        from comprehensive_latency_benchmark import LatencyBenchmark
        
        try:
            benchmark = LatencyBenchmark(
                test_duration=duration,
                symbols_count=symbols_count
            )
            
            success = await benchmark.run_benchmark()
            
            if success and benchmark.end_to_end_latencies:
                latencies = list(benchmark.end_to_end_latencies)
                
                return {
                    'name': test_name,
                    'symbols_count': symbols_count,
                    'duration': duration,
                    'message_count': benchmark.message_count,
                    'avg_latency': statistics.mean(latencies),
                    'min_latency': min(latencies),
                    'max_latency': max(latencies),
                    'p50_latency': sorted(latencies)[len(latencies)//2],
                    'p95_latency': sorted(latencies)[int(len(latencies)*0.95)],
                    'p99_latency': sorted(latencies)[int(len(latencies)*0.99)],
                    'throughput': benchmark.message_count / duration,
                    'jitter': max(latencies) - min(latencies),
                    'timestamp': datetime.now()
                }
            
        except Exception as e:
            print(f"❌ Test {test_name} failed: {e}")
        
        return None
    
    def _generate_comparison_report(self):
        """Generate detailed comparison report"""
        if not self.test_results:
            print("⚠️  No test results to compare")
            return
        
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE COMPARISON REPORT")
        print("=" * 80)
        
        # Summary table
        print(f"\n📋 PERFORMANCE SUMMARY:")
        print(f"{'Test Name':<15} {'Symbols':<8} {'Avg Latency':<12} {'P95':<8} {'P99':<8} {'Throughput':<12} {'Jitter':<8}")
        print("-" * 80)
        
        for result in self.test_results:
            print(f"{result['name']:<15} "
                  f"{result['symbols_count']:<8} "
                  f"{result['avg_latency']:.3f}ms{'':<5} "
                  f"{result['p95_latency']:.3f}{'':<3} "
                  f"{result['p99_latency']:.3f}{'':<3} "
                  f"{result['throughput']:.1f}/sec{'':<5} "
                  f"{result['jitter']:.3f}ms")
        
        # Performance analysis
        print(f"\n🔍 DETAILED ANALYSIS:")
        
        # Find best and worst performers
        best_latency = min(self.test_results, key=lambda x: x['avg_latency'])
        worst_latency = max(self.test_results, key=lambda x: x['avg_latency'])
        best_throughput = max(self.test_results, key=lambda x: x['throughput'])
        
        print(f"  🏆 Best Latency: {best_latency['name']} ({best_latency['avg_latency']:.3f}ms)")
        print(f"  📈 Best Throughput: {best_throughput['name']} ({best_throughput['throughput']:.1f}/sec)")
        print(f"  ⚠️  Worst Latency: {worst_latency['name']} ({worst_latency['avg_latency']:.3f}ms)")
        
        # Scalability analysis
        print(f"\n📈 SCALABILITY ANALYSIS:")
        
        # Calculate latency increase per symbol
        if len(self.test_results) >= 2:
            light_load = min(self.test_results, key=lambda x: x['symbols_count'])
            heavy_load = max(self.test_results, key=lambda x: x['symbols_count'])
            
            symbol_diff = heavy_load['symbols_count'] - light_load['symbols_count']
            latency_diff = heavy_load['avg_latency'] - light_load['avg_latency']
            
            if symbol_diff > 0:
                latency_per_symbol = latency_diff / symbol_diff
                print(f"  📊 Latency increase per symbol: {latency_per_symbol:.4f}ms")
                
                if latency_per_symbol < 0.01:
                    scalability_rating = "🟢 EXCELLENT"
                elif latency_per_symbol < 0.05:
                    scalability_rating = "🟡 GOOD"
                elif latency_per_symbol < 0.1:
                    scalability_rating = "🟠 ACCEPTABLE"
                else:
                    scalability_rating = "🔴 POOR"
                
                print(f"  🎯 Scalability Rating: {scalability_rating}")
        
        # Consistency analysis
        print(f"\n📊 CONSISTENCY ANALYSIS:")
        jitters = [result['jitter'] for result in self.test_results]
        avg_jitter = statistics.mean(jitters)
        max_jitter = max(jitters)
        
        print(f"  📊 Average Jitter: {avg_jitter:.3f}ms")
        print(f"  📊 Maximum Jitter: {max_jitter:.3f}ms")
        
        if max_jitter < 0.1:
            consistency_rating = "🟢 EXCELLENT"
        elif max_jitter < 0.5:
            consistency_rating = "🟡 GOOD"
        elif max_jitter < 1.0:
            consistency_rating = "🟠 ACCEPTABLE"
        else:
            consistency_rating = "🔴 POOR"
        
        print(f"  🎯 Consistency Rating: {consistency_rating}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        overall_avg_latency = statistics.mean([r['avg_latency'] for r in self.test_results])
        
        if overall_avg_latency < 1.0:
            print("  ✅ Excellent performance across all test scenarios")
            print("  ✅ System is ready for production high-frequency trading")
        elif overall_avg_latency < 5.0:
            print("  ✅ Good performance for most trading scenarios")
            print("  💡 Consider optimizing for high-frequency use cases")
        else:
            print("  ⚠️  Performance needs improvement for trading applications")
            print("  🔧 Review system architecture and network configuration")
        
        print("=" * 80)
    
    def _create_performance_charts(self):
        """Create performance visualization charts"""
        if not self.test_results:
            return
        
        try:
            # Create DataFrame for easier plotting
            df = pd.DataFrame(self.test_results)
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('WebSocket Proxy Performance Analysis', fontsize=16, fontweight='bold')
            
            # 1. Latency vs Symbol Count
            ax1.plot(df['symbols_count'], df['avg_latency'], 'bo-', label='Average')
            ax1.plot(df['symbols_count'], df['p95_latency'], 'ro-', label='P95')
            ax1.plot(df['symbols_count'], df['p99_latency'], 'go-', label='P99')
            ax1.set_xlabel('Number of Symbols')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Latency vs Symbol Count')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Throughput vs Symbol Count
            ax2.plot(df['symbols_count'], df['throughput'], 'mo-', linewidth=2)
            ax2.set_xlabel('Number of Symbols')
            ax2.set_ylabel('Throughput (msg/sec)')
            ax2.set_title('Throughput vs Symbol Count')
            ax2.grid(True, alpha=0.3)
            
            # 3. Latency Distribution
            latency_data = [df['min_latency'], df['avg_latency'], df['max_latency']]
            ax3.boxplot(latency_data, labels=['Min', 'Avg', 'Max'])
            ax3.set_ylabel('Latency (ms)')
            ax3.set_title('Latency Distribution')
            ax3.grid(True, alpha=0.3)
            
            # 4. Jitter Analysis
            ax4.bar(df['name'], df['jitter'], color=['green', 'yellow', 'orange', 'red'])
            ax4.set_xlabel('Test Scenario')
            ax4.set_ylabel('Jitter (ms)')
            ax4.set_title('Latency Jitter by Test Scenario')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_filename = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Performance chart saved: {chart_filename}")
            
            # Show chart (optional)
            # plt.show()
            
        except ImportError:
            print("📊 Chart generation skipped (matplotlib/pandas not available)")
        except Exception as e:
            print(f"❌ Chart generation failed: {e}")
    
    def save_results_to_json(self):
        """Save test results to JSON file"""
        if not self.test_results:
            return
        
        # Convert datetime objects to strings for JSON serialization
        json_results = []
        for result in self.test_results:
            json_result = result.copy()
            json_result['timestamp'] = result['timestamp'].isoformat()
            json_results.append(json_result)
        
        filename = f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'test_suite': 'WebSocket Proxy Performance Comparison',
                'timestamp': datetime.now().isoformat(),
                'results': json_results
            }, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")


async def main():
    """Main comparison execution"""
    comparison = PerformanceComparison()
    
    print("🚀 Starting WebSocket Proxy Performance Comparison...")
    
    await comparison.run_comparison_suite()
    
    # Save results
    comparison.save_results_to_json()
    
    print("\n✅ Performance comparison completed!")


if __name__ == "__main__":
    asyncio.run(main())