# drain3 Performance Autoresearch Loop

## Quick Start
```
/autoresearch Goal: Improve drain3 performance across all benchmarks. Scope: /Users/akshat/workspace/drain3/src/*.rs. Metric: cargo bench --features zenbench output. Verify: cargo clippy --all-targets --all-features -- -D warnings && cargo test. Iterations: 10. After each iteration: run zenbench, update results.md, append to progress.jsonl, regenerate progress_plot.html with Plotly chart. Start now.
```

---

## Benchmarking Tools Setup

### 1. zenbench (PRIMARY - recommended)
**Why**: Interleaved execution eliminates thermal drift. Paired Wilcoxon statistics detect real changes vs noise.

```toml
# Cargo.toml
[dev-dependencies]
zenbench = { version = "0.1", features = ["criterion-compat"] }
```

**Commands**:
```bash
# Run with zenbench (interleaved)
cargo bench --features zenbench

# Save baseline
cargo bench --features zenbench -- --save-baseline=before

# Compare (exits 1 if >5% regression)
cargo bench --features zenbench -- --baseline=before --max-regression=5

# Multi-pass for extra stability
cargo bench --features zenbench -- --best-of-passes=3
```

### 2. divan (ALTERNATIVE)
**Why**: Attribute macros, hardware TSC timer, thread contention built-in.

```toml
[dev-dependencies]
divan = "0.1"
```

**Commands**:
```bash
cargo bench --features divan
```

### 3. criterion (FALLBACK)
```bash
cargo bench
cargo bench -- --sample-size 200 --warm-up-time 5
```

---

## Baseline (Iteration 0)

```bash
cargo bench --features zenbench -- --save-baseline=main
```

Expected baseline values:
| Benchmark | Value |
|-----------|-------|
| train_merge | ~410 µs |
| train_fanout | ~2.16 ms |
| match_into | ~2.45 ms |
| match_miss | ~70 µs |
| match_bigdict_hit | ~47 ms |
| match_bigdict_miss | ~25 ms |
| concurrent_match/1t | ~2.5 ms |
| concurrent_match/2t | ~9.0 ms |
| concurrent_match/4t | ~43 ms |

---

## Per-Iteration Protocol

### Step 1: Apply optimization & verify
```bash
# Edit source files
cargo clippy --all-targets --all-features -- -D warnings && cargo test
```

### Step 2: Benchmark with zenbench
```bash
cargo bench --features zenbench 2>&1 | tee iter-N-zenbench.log
```

### Step 3: Compare vs previous
```bash
cargo bench --features zenbench -- --baseline=before-iter-N --max-regression=2
```

### Step 4: Update results.md
```markdown
### Iteration N - YYYY-MM-DD HH:MM

**Hypothesis:** [why this should help]

**Changes:**
- File: `src/X.rs`
  - Before: [code]
  - After: [code]

**zenbench Results:**
| Benchmark | Before | After | Change | Significant? |
|-----------|--------|-------|--------|-------------|
| train_merge | X µs | Y µs | ±Z% | Yes/No |
...

**Decision:** KEPT / REVERTED
```

### Step 5: Append to progress.jsonl
```json
{"iteration":N,"timestamp":"ISO8601","train_merge_us":X.X,"train_fanout_ms":X.X,"match_miss_us":X.X,"match_bigdict_hit_ms":X.X,"concurrent_4t_ms":X.X,"decision":"kept|reverted","changes":{"summary":"brief","files_modified":[],"diff":""},"reasoning":"hypothesis"}
```

### Step 6: Regenerate progress_plot.html (see template below)

### Step 7: Commit
```bash
git add -A && git commit -m "perf: iteration N - [desc]" && git push
```

---

## progress_plot.html Template

```html
<!DOCTYPE html>
<html>
<head>
    <title>drain3 Performance - Autoresearch Progress</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: system-ui; max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }
        h1 { color: #333; }
        h2 { color: #555; border-bottom: 2px solid #eee; padding-bottom: 8px; }
        .metric-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        .metric-name { font-size: 12px; color: #666; text-transform: uppercase; }
        .metric-value { font-size: 24px; font-weight: 600; color: #333; }
        .improved { color: #22c55e; }
        .regressed { color: #ef4444; }
        table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; }
        th, td { border: 1px solid #e0e0e0; padding: 10px; text-align: right; }
        th { background: #4a90d9; color: white; font-weight: 600; }
        tr:nth-child(even) { background: #f9f9f9; }
    </style>
</head>
<body>
    <h1>drain3 Performance Autoresearch</h1>
    <div class="metric-cards" id="cards"></div>
    <div id="chart" style="width:100%;height:500px;"></div>
    <div id="table"></div>

    <script>
    // UPDATE THIS DATA AFTER EACH ITERATION
    const progressData = [
        {
            iteration: 0,
            timestamp: "2026-05-24T00:00:00Z",
            train_merge_us: 410.0,
            train_fanout_ms: 2.16,
            match_into_ms: 2.45,
            match_miss_us: 70.0,
            match_bigdict_hit_ms: 47.0,
            match_bigdict_miss_ms: 25.0,
            concurrent_1t_ms: 2.5,
            concurrent_2t_ms: 9.0,
            concurrent_4t_ms: 43.0,
            decision: "baseline"
        },
        // ADD ITERATION DATA HERE
        {
            iteration: 1,
            timestamp: "2026-05-26T00:00:00Z",
            train_merge_us: 408.0,
            train_fanout_ms: 2.15,
            match_into_ms: 2.41,
            match_miss_us: 68.0,
            match_bigdict_hit_ms: 46.0,
            match_bigdict_miss_ms: 24.5,
            concurrent_1t_ms: 2.5,
            concurrent_2t_ms: 7.5,
            concurrent_4t_ms: 28.0,
            decision: "kept"
        }
    ];

    const baseline = {
        train_merge_us: 410.0,
        train_fanout_ms: 2.16,
        match_into_ms: 2.45,
        match_miss_us: 70.0,
        match_bigdict_hit_ms: 47.0,
        match_bigdict_miss_ms: 25.0,
        concurrent_1t_ms: 2.5,
        concurrent_2t_ms: 9.0,
        concurrent_4t_ms: 43.0
    };

    // Render metric cards (latest kept iteration)
    const lastKept = [...progressData].reverse().find(d => d.decision === 'kept') || progressData[0];
    document.getElementById('cards').innerHTML = Object.entries(baseline).map(([key, base]) => {
        const current = lastKept[key];
        const change = ((base - current) / base) * 100;
        const name = key.replace(/_us|_ms/g, m => m === '_us' ? ' (µs)' : ' (ms)').replace('_', ' ').replace('concurrent ', 'concurrent/').replace(' 1t', '/1t').replace(' 2t', '/2t').replace(' 4t', '/4t');
        return `<div class="metric-card">
            <div class="metric-name">${name}</div>
            <div class="metric-value">${current.toFixed(2)}</div>
            <div class="${change >= 0 ? 'improved' : 'regressed'}">${change >= 0 ? '↓' : '↑'} ${Math.abs(change).toFixed(1)}%</div>
        </div>`;
    }).join('');

    // Render chart (normalized - higher is better)
    const traces = Object.keys(baseline).map(key => ({
        name: key.replace(/_us|_ms/g, m => m === '_us' ? ' (µs)' : ' (ms)').replace('_', ' '),
        x: progressData.map(d => d.iteration),
        y: progressData.map(d => baseline[key] / d[key]),
        mode: 'lines+markers',
        line: { shape: 'spline' }
    }));

    const colors = progressData.map(d =>
        d.decision === 'kept' ? '#22c55e' :
        d.decision === 'reverted' ? '#ef4444' : '#4a90d9'
    );

    Plotly.newPlot('chart', traces.map((t, i) => ({
        ...t,
        marker: { color: colors, size: 10 }
    })), {
        title: 'Normalized Performance (higher = faster) - Green=Kept, Red=Reverted, Blue=Baseline',
        xaxis: { title: 'Iteration', dtick: 1 },
        yaxis: { title: 'Normalized (1.0 = baseline)', rangemode: 'tozero' },
        shapes: [{ type: 'line', x0: 0, x1: progressData.length, y0: 1, y1: 1, line: { dash: 'dot', color: '#999' }}]
    }, {responsive: true});

    // Render results table
    document.getElementById('table').innerHTML = `
        <h2>Benchmark Results</h2>
        <table>
            <tr>
                <th>Iter</th>
                <th>train_merge</th>
                <th>train_fanout</th>
                <th>match_miss</th>
                <th>match_bigdict_hit</th>
                <th>concurrent/1t</th>
                <th>concurrent/4t</th>
                <th>Decision</th>
            </tr>
            ${progressData.map(d => `
                <tr>
                    <td>${d.iteration}</td>
                    <td>${d.train_merge_us.toFixed(1)} µs</td>
                    <td>${d.train_fanout_ms.toFixed(2)} ms</td>
                    <td>${d.match_miss_us.toFixed(1)} µs</td>
                    <td>${d.match_bigdict_hit_ms.toFixed(1)} ms</td>
                    <td>${d.concurrent_1t_ms.toFixed(2)} ms</td>
                    <td>${d.concurrent_4t_ms.toFixed(1)} ms</td>
                    <td class="${d.decision === 'kept' ? 'improved' : d.decision === 'reverted' ? 'regressed' : ''}">${d.decision}</td>
                </tr>
            `).join('')}
        </table>
    `;
    </script>
</body>
</html>
```

---

## Decision Criteria

**KEPT** if:
- >=3/9 benchmarks improved >2% (zenbench noise threshold)
- No benchmark regressed >5%

**REVERTED** otherwise.

---

## Final Report Template

```markdown
# drain3 Performance Optimization Report

## Executive Summary
- Total iterations: N
- Improvements kept: M
- Regressions reverted: R

## Best Improvements (vs iteration 0 baseline)
| Benchmark | Before | After | Improvement | Iteration |
|-----------|--------|-------|-------------|-----------|
| concurrent_4t | 43 ms | X ms | -X% | N |
| concurrent_2t | 9 ms | X ms | -X% | N |
| match_miss | 70 µs | X µs | -X% | N |

## Performance Over Time
See: progress_plot.html

## All Changes (Chronological)
### Iteration 1 - YYYY-MM-DD
**Change:** [description]
**Result:** KEPT/REVERTED

...

## Conclusions
- What worked: [list]
- What didn't: [list]
- Recommended next steps: [list]
```
