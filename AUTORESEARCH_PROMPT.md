# drain3 Performance Autoresearch Loop

## Quick Start
In a new chat, type:
```
/autoresearch Goal: Significantly improve drain3 performance across all benchmarks. Scope: /Users/akshat/workspace/drain3/src/*.rs. Metric: cargo bench output - maximize throughput, minimize latency. Verify: cargo clippy --all-targets --all-features -- -D warnings && cargo test. Iterations: 10. After each iteration: run benchmarks, update /Users/akshat/workspace/drain3/results.md with table, append JSON to /Users/akshat/workspace/drain3/progress.jsonl, regenerate /Users/akshat/workspace/drain3/progress_plot.html with Plotly chart showing normalized throughput over iterations. Start now.
```

---

## Goal
Significantly improve drain3 throughput across all benchmarks with measurable progress visualization.

## Baseline (Iteration 0)
| Benchmark | Value |
|-----------|-------|
| train_merge | 779.65 µs |
| train_fanout | 4163.7 ns |
| match_miss | 134.47 µs |
| match_bigdict_hit | 90.532 ms |
| concurrent_match/1t | 4894.7 ns |

## Per-Iteration Protocol

1. **Run benchmarks**: `cd /Users/akshat/workspace/drain3 && cargo bench 2>&1 | grep -E "^Benchmarking|^ +time:"`

2. **Apply change** (or launch parallel explore agents for new ideas)

3. **Verify**: `cargo clippy --all-targets --all-features -- -D warnings && cargo test`

4. **Update results.md** - append iteration with table

5. **Append to progress.jsonl** - MUST include actual changes:
```json
{"iteration":N,"timestamp":"ISO8601","train_merge_us":X.X,"train_fanout_ns":X.X,"match_miss_us":X.X,"match_bigdict_hit_ms":X.X,"concurrent_1t_ns":X.X,"decision":"kept|reverted","changes":{"summary":"brief description","files_modified":["src/lib.rs","src/tree.rs"],"diff":"git diff --stat output or brief description of what changed"},"reasoning":"why this change was attempted"}
```

6. **Append detailed entry to results.md**:
```markdown
### Iteration N - YYYY-MM-DD HH:MM

**Changes Made:**
- File: `src/X.rs`
  - Before: [describe code before]
  - After: [describe code after]
  - Why: [rationale for change]

**Benchmark Results:**
| Benchmark | Baseline | Iteration N | Improvement |
|-----------|----------|-------------|-------------|
| train_merge | 779.65 µs | X µs | ±Y% |
...

**Kept/Reversion Decision:** [KEPT if >=2/5 metrics improved, REVERTED otherwise]
```

7. **Regenerate progress_plot.html** - MUST update visualization:
```html
<!DOCTYPE html>
<html><head><title>drain3 Performance</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script></head>
<body>
<h1>drain3 Performance Over Time</h1>
<div id="chart" style="width:100%;height:600px;"></div>
<script>
const data = [
  {"iteration":0,"train_merge_us":779.65,"train_fanout_ns":4163.7,"match_miss_us":134.47,"match_bigdict_hit_ms":90.532,"concurrent_1t_ns":4894.7,"decision":"baseline"},
  {"iteration":1,"train_merge_us":780.0,"train_fanout_ns":4160.0,"match_miss_us":128.0,"match_bigdict_hit_ms":87.0,"concurrent_1t_ns":5122.0,"decision":"kept"}
  // ADD NEW ENTRIES AFTER EACH ITERATION HERE
];
const baseline = {train_merge:779.65,train_fanout:4163.7,match_miss:134.47,match_bigdict_hit:90.532,concurrent_1t:4894.7};
const traces = [
  {name:'train_merge', y:data.map(d=>baseline.train_merge/d.train_merge_us)},
  {name:'train_fanout', y:data.map(d=>baseline.train_fanout/d.train_fanout_ns)},
  {name:'match_miss', y:data.map(d=>baseline.match_miss/d.match_miss_us)},
  {name:'match_bigdict_hit', y:data.map(d=>baseline.match_bigdict_hit/d.match_bigdict_hit_ms)},
  {name:'concurrent_1t', y:data.map(d=>baseline.concurrent_1t/d.concurrent_1t_ns)}
].map(t=>({...t, mode:'lines+markers', line:{shape:'spline'}}));

const colors = data.map(d=>d.decision==='kept'?'green':d.decision==='reverted'?'red':'blue');
Plotly.newPlot('chart', traces.map((t,i)=>({...t, marker:{color:colors}})), {
  title:'Normalized Throughput (higher=better, green=kept, red=reverted, blue=baseline)',
  xaxis:{title:'Iteration'}, yaxis:{title:'Normalized Performance (1.0=baseline)'},
  shapes:[{type:'line',x0:0,x1:data.length,y0:1,y1:1,line:{dash:'dot',color:'gray'}}]
});
</script></body></html>
```

7. **Commit**: `git add -A && git commit -m "perf: iteration N - [description]" && git push`

## Final Report Generation (after stopping)

When the loop stops, generate a final `REPORT.md` with:

```markdown
# drain3 Performance Optimization Report

## Executive Summary
- Total iterations completed: N
- Improvements kept: M
- Regressions reverted: R
- Best improvement per benchmark:
  - train_merge: X% (iteration Y)
  - train_fanout: X% (iteration Y)
  - match_miss: X% (iteration Y)
  - match_bigdict_hit: X% (iteration Y)
  - concurrent_1t: X% (iteration Y)

## All Changes Made (Chronological)

### Iteration 1 - YYYY-MM-DD
**Change:** [description]
**Files:** [list]
**Result:** [KEPT/REVERTED]
**Metrics:** [table]

### Iteration 2 - YYYY-MM-DD
...

## Performance Trend Chart
See: progress_plot.html

## Conclusions & Recommendations
- [Key findings]
- [What worked]
- [What didn't work]
- [Recommended next steps]
```

# drain3 Performance Autoresearch Loop

## Quick Start
In a new chat, type:
```
/autoresearch Goal: Significantly improve drain3 performance across all benchmarks. Scope: /Users/akshat/workspace/drain3/src/*.rs. Metric: cargo bench output - maximize throughput, minimize latency. Verify: cargo clippy --all-targets --all-features -- -D warnings && cargo test. Iterations: 10. After each iteration: run benchmarks, update /Users/akshat/workspace/drain3/results.md with detailed change log, append JSON to /Users/akshat/workspace/drain3/progress.jsonl (include changes.summary, changes.files_modified, changes.diff, reasoning fields), regenerate /Users/akshat/workspace/drain3/progress_plot.html with Plotly chart showing normalized throughput over iterations + change history panel. Start now.
```

---

## Goal
Significantly improve drain3 throughput across all benchmarks with measurable progress visualization and complete change history.

## Baseline (Iteration 0)
| Benchmark | Value |
|-----------|-------|
| train_merge | 779.65 µs |
| train_fanout | 4163.7 ns |
| match_miss | 134.47 µs |
| match_bigdict_hit | 90.532 ms |
| concurrent_match/1t | 4894.7 ns |

## Per-Iteration Protocol

1. **Run benchmarks**: `cd /Users/akshat/workspace/drain3 && cargo bench 2>&1 | grep -E "^Benchmarking|^ +time:"`

2. **Analyze & plan change** - Use explore agents or analyze directly

3. **Apply change** - Edit source files

4. **Verify**: `cargo clippy --all-targets --all-features -- -D warnings && cargo test`

5. **Append to progress.jsonl** - Include full change documentation:
```json
{"iteration":N,"timestamp":"ISO8601","train_merge_us":X.X,"train_fanout_ns":X.X,"match_miss_us":X.X,"match_bigdict_hit_ms":X.X,"concurrent_1t_ns":X.X,"decision":"kept|reverted","changes":{"summary":"brief description of what changed","files_modified":["src/lib.rs","src/tree.rs"],"diff":"git diff --stat or line-level description of actual changes","before_code":"code snippet before","after_code":"code snippet after"},"reasoning":"why this change was attempted, what hypothesis it tests"}
```

6. **Append detailed entry to results.md**:
```markdown
### Iteration N - YYYY-MM-DD HH:MM

**Hypothesis:** [why this change should help]

**Changes Made:**
- File: `src/X.rs`
  - Before: [code before change]
  - After: [code after change]
  - Lines changed: [number or specific lines]

**Benchmark Results:**
| Benchmark | Baseline | Iteration N | Improvement |
|-----------|----------|-------------|-------------|
| train_merge | 779.65 µs | X µs | ±Y% |
...

**Decision:** [KEPT if >=2/5 metrics improved, REVERTED otherwise]
**Lessons Learned:** [what this tells us about the codebase]
```

7. **Regenerate progress_plot.html** - Update with new iteration data (append to the `progressData` array in the script section)

8. **Commit**: `git add -A && git commit -m "perf: iteration N - [description]" && git push`