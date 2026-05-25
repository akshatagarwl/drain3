# drain3 Performance Autoresearch Loop

## Quick Start
```
/autoresearch Goal: Improve drain3 performance. Scope: /Users/akshat/workspace/drain3/src/*.rs. Metric: cargo bench --features zenbench output. Verify: cargo clippy --all-targets --all-features -- -D warnings && cargo test. Iterations: 10. After each iteration: run benchmarks with zenbench, update results.md, append JSON to progress.jsonl, regenerate progress_plot.html. Start now.
```

---

## Benchmarking Tools

### zenbench (recommended)
- **Interleaved execution**: Runs all benchmarks in shuffled order each round
- **Paired statistics**: Wilcoxon test on round-by-round differences  
- **Noise threshold**: Configurable significance gate (default ±2%)
- **Run**: `cargo bench --features zenbench`
- **Save baseline**: `cargo bench --features zenbench -- --save-baseline=main`
- **Compare**: `cargo bench --features zenbench -- --baseline=main`
- **CI mode**: `cargo bench --features zenbench -- --baseline=main --max-regression=5`

### divan (optional)
- **Attribute macros**: `#[divan::bench]`
- **Hardware TSC**: Uses rdtsc for precise timing
- **Thread contention**: Built-in `#[divan::bench(threads = N)]`

### criterion (fallback)
- `cargo bench` (default)
- More samples: `cargo bench -- --sample-size 200`

---

## Baseline (Iteration 0 - 2026-05-24)

| Benchmark | Value |
|-----------|-------|
| train_merge | ~410 µs |
| train_fanout | ~2.16 ms |
| match_into | ~2.45 ms |
| match_miss | ~70 µs |
| match_bigdict_hit | ~47 ms |
| concurrent_match/1t | ~2.5 ms |
| concurrent_match/2t | ~9.0 ms |
| concurrent_match/4t | ~43 ms |

---

## Per-Iteration Protocol

1. **Run benchmarks**: `cd /Users/akshat/workspace/drain3 && cargo bench --features zenbench 2>&1`

2. **Apply change** - Edit source files

3. **Verify**: `cargo clippy --all-targets --all-features -- -D warnings && cargo test`

4. **Update results.md** - append iteration with table

5. **Append to progress.jsonl** - full change documentation

6. **Regenerate progress_plot.html** - update progressData array

7. **Commit**: `git add -A && git commit -m "perf: iteration N - [desc]" && git push`

---

## Decision Criteria

- **KEPT**: >=3/8 benchmarks improved >2% with non-overlapping CI
- **REVERTED**: Any benchmark regressed >5%

---

## Final Report

Generate `REPORT.md` with executive summary, change log, and conclusions.
