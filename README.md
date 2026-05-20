# drain3

[![docs.rs](https://docs.rs/drain3/badge.svg)](https://docs.rs/drain3)
[![crates.io](https://img.shields.io/crates/v/drain3.svg)](https://crates.io/crates.io)
![License](https://img.shields.io/crates/l/drain3.svg)

Fast log template extraction via fixed-depth prefix trees.

Parses log lines into tokens, clusters by prefix tree, replaces variables with `<*>` placeholders. Extracts parameters from previously unseen lines.

## Usage

```rust
let samples = vec![
    "connection from 10.0.0.1 timeout after 5000ms",
    "connection from 10.0.0.2 timeout after 3000ms",
];
let matcher = drain3::train(&samples, Config::default())?;
let (id, args, matched) = matcher.match_line("connection from 192.168.1.1 timeout after 42ms");
assert!(matched);
assert_eq!(args, vec!["192.168.1.1", "42ms"]);
```

## Performance

- **First/last token prefilter** — 3–5× speedup on typical workloads
- **has_param_first quick rejection** — instant rejection of non-matching first tokens
- **Anchor checks** — 2-comparison candidate rejection

## License

Apache-2.0