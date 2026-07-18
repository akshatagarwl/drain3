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

## Masking

Collapse high-cardinality values (numbers, IPs, UUIDs, hashes) into stable
placeholders *before* tokenization so they don't spawn distinct clusters. Rules
run in order — list specific patterns before generic ones.

```rust
use drain3::{Config, MaskingInstruction};

let cfg = Config::builder()
    .masking(vec![
        MaskingInstruction { pattern: r"\b\d{1,3}(\.\d{1,3}){3}\b".into(), mask: "<IP>".into() },
        MaskingInstruction { pattern: r"\d+".into(), mask: "<NUM>".into() },
    ])
    .build();
// "user 1001 from 10.0.0.1" and "user 2002 from 10.0.0.2" now share one
// template: "user <NUM> from <IP>" (the values are part of the template,
// not extracted as <*> params).
```

An invalid pattern surfaces as `Error::InvalidMaskingRegex` from `train`
(or panics from `Matcher::new`, which does not validate).

## Performance

- **First/last token prefilter** — 3–5× speedup on typical workloads
- **has_param_first quick rejection** — instant rejection of non-matching first tokens
- **Anchor checks** — 2-comparison candidate rejection

## License

Apache-2.0