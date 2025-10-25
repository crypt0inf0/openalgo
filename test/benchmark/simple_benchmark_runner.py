"""
Simple Benchmark Runner (Windows Compatible)
Easy-to-use script for running different benchmark scenarios
"""

import subprocess
import sys
import time
from datetime import datetime


class SimpleBenchmarkRunner:
    def __init__(self):
        self.results = []
    
    def run_scenario(self, name, duration, symbols, description):
        """Run a specific benchmark scenario"""
        print(f"\nRUNNING SCENARIO: {name}")
        print(f"Description: {description}")
        print(f"Duration: {duration}s | Symbols: {symbols}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run the simple benchmark
            result = subprocess.run([
                sys.executable, 
                "simple_latency_benchmark.py",
                "--duration", str(duration),
                "--symbols", str(symbols)
            ], capture_output=True, text=True, cwd=".")
            
            end_time = time.time()
            actual_duration = end_time - start_time
            
            success = result.returncode == 0
            
            self.results.append({
                'name': name,
                'description': description,
                'duration': duration,
                'symbols': symbols,
                'actual_duration': actual_duration,
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr
            })
            
            if success:
                print(f"SUCCESS: {name} completed in {actual_duration:.1f}s")
                # Print key results from stdout
                self._extract_key_results(result.stdout)
            else:
                print(f"FAILED: {name} failed after {actual_duration:.1f}s")
                if result.stderr:
                    print(f"Error: {result.stderr}")
            
            return success
            
        except Exception as e:
            print(f"Failed to run {name}: {e}")
            return False
    
    def _extract_key_results(self, stdout):
        """Extract key performance metrics from stdout"""
        lines = stdout.split('\n')
        for line in lines:
            if 'Average:' in line and 'ms' in line:
                print(f"  -> {line.strip()}")
            elif 'Overall Rating:' in line:
                print(f"  -> {line.strip()}")
            elif 'Average Throughput:' in line:
                print(f"  -> {line.strip()}")
    
    def run_all_scenarios(self):
        """Run all predefined benchmark scenarios"""
        print("WEBSOCKET PROXY BENCHMARK SUITE")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        scenarios = [
            {
                'name': 'Quick Test',
                'duration': 30,
                'symbols': 3,
                'description': 'Fast 30-second test with 3 symbols'
            },
            {
                'name': 'Standard Test',
                'duration': 60,
                'symbols': 5,
                'description': 'Standard 1-minute test with 5 symbols'
            },
            {
                'name': 'Extended Test',
                'duration': 90,
                'symbols': 8,
                'description': 'Extended 90-second test with 8 symbols'
            },
            {
                'name': 'High Volume Test',
                'duration': 60,
                'symbols': 10,
                'description': 'High volume test with 10 symbols'
            }
        ]
        
        successful_tests = 0
        total_tests = len(scenarios)
        
        for scenario in scenarios:
            success = self.run_scenario(
                scenario['name'],
                scenario['duration'],
                scenario['symbols'],
                scenario['description']
            )
            
            if success:
                successful_tests += 1
            
            # Small delay between tests
            time.sleep(2)
        
        # Print summary
        self.print_summary(successful_tests, total_tests)
    
    def print_summary(self, successful_tests, total_tests):
        """Print benchmark suite summary"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUITE SUMMARY")
        print("=" * 80)
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Successful Tests: {successful_tests}/{total_tests}")
        print(f"Failed Tests: {total_tests - successful_tests}")
        
        if successful_tests == total_tests:
            print("ALL TESTS PASSED! WebSocket proxy is performing well.")
        elif successful_tests > 0:
            print("Some tests passed. Check failed tests for issues.")
        else:
            print("ALL TESTS FAILED! WebSocket proxy needs attention.")
        
        print("\nDETAILED RESULTS:")
        print("-" * 80)
        
        for result in self.results:
            status = "PASS" if result['success'] else "FAIL"
            print(f"{status} | {result['name']:<15} | {result['duration']:3d}s | {result['symbols']:2d} symbols | {result['actual_duration']:5.1f}s actual")
        
        print("=" * 80)


def main():
    """Main benchmark runner"""
    runner = SimpleBenchmarkRunner()
    
    if len(sys.argv) > 1:
        # Custom scenario
        if sys.argv[1] == "custom":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            symbols = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            
            runner.run_scenario(
                "Custom Test",
                duration,
                symbols,
                f"Custom test: {duration}s with {symbols} symbols"
            )
        else:
            print("Usage: python simple_benchmark_runner.py [custom] [duration] [symbols]")
            print("Example: python simple_benchmark_runner.py custom 120 10")
            return 1
    else:
        # Run all scenarios
        runner.run_all_scenarios()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)