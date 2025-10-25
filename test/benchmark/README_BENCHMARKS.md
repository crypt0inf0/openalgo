# WebSocket Proxy Latency Benchmark Suite

A comprehensive testing suite for measuring and analyzing WebSocket proxy performance with sub-millisecond precision.

## 🚀 Quick Start

### Run Standard Benchmark
```bash
cd test
python comprehensive_latency_benchmark.py
```

### Run All Benchmark Scenarios
```bash
python run_benchmark.py
```

### Run Custom Test
```bash
python run_benchmark.py custom 120 10  # 120 seconds, 10 symbols
```

## 📊 Available Tests

### 1. Comprehensive Latency Benchmark
**File:** `comprehensive_latency_benchmark.py`

**Purpose:** Detailed latency measurement with comprehensive analytics

**Features:**
- End-to-end latency measurement (Broker → Client)
- Component latency breakdown (Broker → Proxy → Client)
- Per-symbol performance tracking
- Throughput analysis
- Connection quality metrics
- Performance rating system

**Usage:**
```bash
python comprehensive_latency_benchmark.py --duration 60 --symbols 10
```

**Output:**
- Real-time latency monitoring
- Detailed performance statistics
- P50, P95, P99 percentile analysis
- Performance rating and recommendations

### 2. Benchmark Runner
**File:** `run_benchmark.py`

**Purpose:** Automated execution of multiple benchmark scenarios

**Scenarios:**
- **Quick Test:** 30s with 3 symbols
- **Standard Test:** 60s with 5 symbols  
- **Extended Test:** 120s with 10 symbols
- **High Volume Test:** 60s with 15 symbols

**Usage:**
```bash
python run_benchmark.py                    # Run all scenarios
python run_benchmark.py custom 90 8       # Custom: 90s, 8 symbols
```

### 3. Performance Comparison
**File:** `performance_comparison.py`

**Purpose:** Compare performance across different load conditions

**Features:**
- Multi-scenario testing
- Scalability analysis
- Performance visualization charts
- JSON result export
- Consistency analysis

**Usage:**
```bash
python performance_comparison.py
```

**Output:**
- Comparative performance tables
- Scalability metrics
- Performance charts (PNG)
- JSON results file

### 4. Stress Test
**File:** `stress_test.py`

**Purpose:** Maximum load testing to find performance limits

**Features:**
- Progressive load testing (5 → 50 symbols)
- Breaking point detection
- Resource usage monitoring
- Stability analysis
- Capacity estimation

**Usage:**
```bash
python stress_test.py --symbols 30 --duration 300
```

**Output:**
- Performance under stress analysis
- Resource usage statistics
- Stability ratings
- Capacity estimates

### 5. Direct WebSocket Test
**File:** `direct_websocket_latency_test.py`

**Purpose:** Raw WebSocket message analysis bypassing client libraries

**Features:**
- Direct WebSocket connection
- Raw message inspection
- Timestamp validation
- Ultra-precise latency measurement

## 📈 Performance Metrics

### Latency Classifications
- **🟢 EXCELLENT:** < 1ms (Perfect for HFT)
- **🟡 VERY GOOD:** 1-5ms (Algorithmic trading)
- **🟠 GOOD:** 5-10ms (Retail trading)
- **🔴 ACCEPTABLE:** 10-50ms (Basic trading)
- **⚫ POOR:** > 50ms (Needs optimization)

### Key Measurements
- **End-to-End Latency:** Total time from broker to client
- **Component Latency:** Broker→Proxy and Proxy→Client breakdown
- **Throughput:** Messages per second
- **Jitter:** Latency variance (consistency)
- **P95/P99:** 95th and 99th percentile latencies

## 🔧 Configuration

### WebSocket Connection
- **URL:** `ws://127.0.0.1:8765`
- **Auth Key:** `32a3d40880806508fa1397e3b7621a58acff440470d08562863adebdb4f6c194`

### Test Symbols
Default NSE symbols used for testing:
- TCS, RELIANCE, HDFCBANK, INFY, ICICIBANK
- HINDUNILVR, ITC, SBIN, BHARTIARTL, KOTAKBANK
- And more...

### Customization
All tests support command-line arguments for:
- Test duration (`--duration`)
- Number of symbols (`--symbols`)
- Custom symbol lists (modify source)

## 📊 Sample Results

### Excellent Performance Example
```
⚡ END-TO-END LATENCY (Broker WSS → Client):
  📊 Samples: 480
  📈 Average: 0.31ms
  📊 P95: 0.31ms
  📊 P99: 0.31ms
  🎯 Rating: 🟢 EXCELLENT
```

### Throughput Analysis
```
🚀 THROUGHPUT ANALYSIS:
  📊 Average Throughput: 8.0 msg/sec
  📊 Peak Throughput: 12.5 msg/sec
  📊 Data Rate: 0.15 MB/sec
```

## 🛠️ Dependencies

### Required
- Python 3.7+
- `asyncio`
- `websockets`
- `json`
- `statistics`

### Optional (for enhanced features)
- `matplotlib` (for performance charts)
- `pandas` (for data analysis)
- `psutil` (for resource monitoring)

### Install Optional Dependencies
```bash
pip install matplotlib pandas psutil
```

## 🎯 Best Practices

### For Accurate Results
1. **Close unnecessary applications** during testing
2. **Use wired network connection** (avoid WiFi)
3. **Run multiple test iterations** for consistency
4. **Test during different times** to account for network conditions
5. **Monitor system resources** during testing

### Interpreting Results
- **Focus on P95/P99** rather than just averages
- **Check jitter** for consistency assessment
- **Compare across different loads** for scalability
- **Monitor error rates** for stability

### Production Readiness
- **< 1ms average latency** for HFT applications
- **< 0.1ms jitter** for consistent performance
- **Zero connection drops** during stress tests
- **Linear scalability** up to expected load

## 🚨 Troubleshooting

### Common Issues

**High Latency:**
- Check network configuration
- Verify WebSocket proxy is running locally
- Monitor system resource usage
- Test with fewer symbols

**Connection Errors:**
- Ensure WebSocket proxy is running on port 8765
- Verify authentication key is correct
- Check firewall settings

**Inconsistent Results:**
- Run longer test durations
- Close background applications
- Use wired network connection
- Test multiple times

### Performance Optimization Tips
1. **Use local deployment** for minimum network latency
2. **Optimize system resources** (CPU, memory)
3. **Configure network buffers** appropriately
4. **Monitor and tune** WebSocket proxy settings

## 📝 Output Files

Tests generate various output files:
- `performance_results_YYYYMMDD_HHMMSS.json` - Detailed results
- `performance_comparison_YYYYMMDD_HHMMSS.png` - Performance charts
- Console output with real-time metrics

## 🎉 Expected Results

With the optimized lightweight WebSocket proxy, you should see:
- **Average latency:** 0.3-0.5ms
- **P95 latency:** < 1ms
- **Throughput:** 8-15 messages/second per symbol
- **Jitter:** < 0.1ms
- **Zero connection drops** under normal load

These results indicate **production-ready performance** suitable for real-time trading applications.