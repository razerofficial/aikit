# rzr-aikit Performance Benchmarks

Serving performance of rzr-aikit (vLLM 0.30.0) on two platforms: x86_64 with an NVIDIA RTX 5090 Laptop GPU (24GB), and ARM64 with an NVIDIA GB10.

Each model was served on its own and hit with 200 chat requests sent at once, so the figures below show behavior under heavy load, not single-request latency.

## Throughput

Output tokens per second, sorted by x86_64 throughput. Higher is better.

| Model | x86_64 (RTX 5090) | ARM64 (GB10) | Ratio |
|-------|-------------------|--------------|-------|
| MiniCPM5-1B | **6803** | 2729 | 2.5x |
| Qwen3.8-2B-Distill | 2911 | 1205 | 2.4x |
| granite-4.1-3b | 2212 | 875 | 2.5x |
| LFM2.5-2.6B | 2096 | 723 | 2.9x |
| TwIL-LM3 | 2087 | 681 | 3.1x |
| LFM2.5-8B-A1B | 1661 | 523 | 3.2x |
| Qwen3.8-4B-Distill | 1386 | 528 | 2.6x |
| VibeThinker-3B | 1000 | 254 | 3.9x |
| Nemotron-3-Nano-4B | 785 | 378 | 2.1x |

All 1800 requests per platform completed successfully.

## Latency

Values are p50 / p90 / p99 in milliseconds. Lower is better.

- **TTFT**: time to first token
- **TPOT**: average time per output token for a request
- **ITL**: gap between consecutive tokens
- **E2EL**: total time for a request, from send to last token

### x86_64 (RTX 5090 Laptop)

| Model | TTFT | TPOT | ITL | E2EL |
|---|---|---|---|---|
| MiniCPM5-1B | 576 / 1018 / 1108 | 7.8 / 39.4 / 46.5 | 5.2 / 6.1 / 48.8 | 1437 / 2804 / 3940 |
| Qwen3.8-2B-Distill | 1163 / 1788 / 2029 | 21.2 / 72.3 / 79.4 | 13.7 / 20.0 / 83.8 | 3730 / 6734 / 8819 |
| granite-4.1-3b | 2184 / 3850 / 4037 | 25.9 / 120.1 / 170.8 | 17.9 / 20.2 / 182.2 | 6025 / 10178 / 13445 |
| LFM2.5-2.6B | 1823 / 2957 / 3049 | 19.5 / 113.1 / 126.5 | 12.2 / 17.2 / 128.1 | 4821 / 8232 / 13239 |
| TwIL-LM3 | 1942 / 3324 / 3613 | 26.2 / 147.9 / 154.9 | 16.7 / 22.2 / 159.2 | 5921 / 9903 / 14884 |
| LFM2.5-8B-A1B | 1438 / 2221 / 2311 | 27.5 / 87.0 / 92.2 | 22.9 / 25.8 / 95.4 | 5411 / 12491 / 21799 |
| Qwen3.8-4B-Distill | 2504 / 4582 / 5558 | 48.5 / 167.8 / 191.7 | 30.4 / 45.8 / 197.6 | 9130 / 16012 / 24733 |
| VibeThinker-3B | 1806 / 3156 / 3457 | 60.9 / 147.8 / 152.8 | 11.3 / 140.7 / 155.4 | 3633 / 5797 / 9911 |
| Nemotron-3-Nano-4B | 2603 / 4338 / 4677 | 57.2 / 173.4 / 192.7 | 21.8 / 36.1 / 189.8 | 4593 / 10080 / 15189 |

### ARM64 (GB10)

| Model | TTFT | TPOT | ITL | E2EL |
|---|---|---|---|---|
| MiniCPM5-1B | 868 / 1429 / 1589 | 17.8 / 53.1 / 63.5 | 13.4 / 19.0 / 66.1 | 2665 / 5835 / 8928 |
| Qwen3.8-2B-Distill | 1720 / 2908 / 3313 | 51.1 / 113.4 / 132.4 | 36.9 / 50.2 / 141.4 | 7925 / 17241 / 23970 |
| granite-4.1-3b | 2906 / 4771 / 5345 | 54.6 / 173.7 / 230.3 | 42.9 / 45.7 / 237.9 | 8770 / 20767 / 31207 |
| LFM2.5-2.6B | 2078 / 3701 / 3838 | 40.7 / 146.7 / 162.6 | 31.5 / 37.0 / 163.9 | 8190 / 16655 / 34181 |
| TwIL-LM3 | 2586 / 4336 / 4667 | 54.0 / 180.7 / 192.5 | 43.2 / 47.1 / 197.9 | 9587 / 19726 / 27405 |
| LFM2.5-8B-A1B | 3367 / 5196 / 5294 | 86.1 / 148.0 / 192.6 | 78.9 / 84.3 / 256.2 | 15709 / 40002 / 57160 |
| Qwen3.8-4B-Distill | 4260 / 7141 / 7824 | 119.4 / 290.0 / 333.7 | 89.2 / 120.5 / 361.1 | 21148 / 43570 / 64530 |
| VibeThinker-3B | 2406 / 4005 / 4396 | 103.7 / 187.0 / 190.0 | 32.9 / 42.6 / 193.1 | 4683 / 10780 / 28227 |
| Nemotron-3-Nano-4B | 4093 / 6955 / 7503 | 126.4 / 264.5 / 299.4 | 74.9 / 114.3 / 304.0 | 7558 / 24396 / 33970 |

## Startup time

Time from launching the server until it is ready to accept requests, including model compilation and CUDA graph capture.

| Model | x86_64 | ARM64 |
|-------|--------|-------|
| MiniCPM5-1B | 129s | 73s |
| Qwen3.8-2B-Distill | 144s | 146s |
| granite-4.1-3b | 62s | 110s |
| LFM2.5-2.6B | 139s | 104s |
| TwIL-LM3 | 62s | 109s |
| LFM2.5-8B-A1B | 67s | 208s |
| Qwen3.8-4B-Distill | 88s | 197s |
| VibeThinker-3B | 62s | 124s |
| Nemotron-3-Nano-4B | 108s | 285s |

## Key takeaways

- **Throughput:** x86_64 is 2.7x faster than ARM64 on average (2327 vs 877 tok/s), ranging from 2.1x to 3.9x by model.
- **Fastest model:** MiniCPM5-1B on both platforms, at 6803 tok/s (x86_64) and 2729 tok/s (ARM64).
- **First token:** p50 TTFT is 0.6-2.6s on x86_64 and 0.9-4.3s on ARM64. Under this load, part of TTFT is time spent queued behind other requests.
- **Tail latency:** p99 TTFT stays within 21% of p90, but p99 ITL is 3-14x the median. Expect occasional pauses between tokens when many requests are in flight.
- **Slowest requests:** on ARM64, p99 end-to-end latency reaches 57s for LFM2.5-8B-A1B and 65s for Qwen3.8-4B-Distill, against 22s and 25s on x86_64.
- **Startup:** ARM64 takes longer to become ready on most models, up to 285s for Nemotron-3-Nano-4B.

## Test configuration

- 200 prompts per model from the ShareGPT conversation set, all sent at once (unlimited request rate)
- Up to 256 concurrent sequences, 8192-token maximum context length
- 90% of GPU memory allocated to vLLM
- One run per model and platform. Each model was started from a cold server with the GPU freed between models.
