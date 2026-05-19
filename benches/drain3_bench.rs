use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use drain3::{train, Config};
use std::sync::Arc;
use std::thread;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }

    fn gen_range(&mut self, max: u64) -> u64 {
        self.next() % max
    }
}

fn bench_drain3(c: &mut Criterion) {
    const N_LINES: usize = 5000;
    let mut rng = Rng::new(99);

    let formatters: [fn(&mut Rng) -> String; 5] = [
        |rng| {
            format!(
                "svc auth user {} status ok latency {} region us-east-1 method GET",
                rng.gen_range(10_000),
                rng.gen_range(10_000)
            )
        },
        |rng| {
            format!(
                "svc billing user {} amount {} currency USD region eu-west-1 method POST",
                rng.gen_range(10_000),
                rng.gen_range(10_000)
            )
        },
        |rng| {
            format!(
                "svc gateway request {} upstream {} duration 42 region ap-south-1 method PUT",
                rng.gen_range(10_000),
                rng.gen_range(10_000)
            )
        },
        |rng| {
            format!(
                "svc storage bucket {} object {} size 1024 region us-west-2 method DELETE",
                rng.gen_range(10_000),
                rng.gen_range(10_000)
            )
        },
        |rng| {
            format!(
                "svc scheduler job {} worker {} priority high region us-central-1 method PATCH",
                rng.gen_range(10_000),
                rng.gen_range(10_000)
            )
        },
    ];
    let mut merge_lines = Vec::with_capacity(N_LINES);
    for i in 0..N_LINES {
        merge_lines.push(formatters[i % formatters.len()](&mut rng));
    }
    let train_merge = &merge_lines[..N_LINES / 10];

    rng = Rng::new(99);
    let mut fanout_lines = Vec::with_capacity(N_LINES);
    for i in 0..N_LINES {
        let a = b'a' + ((i / 676) % 26) as u8;
        let b = b'a' + ((i / 26) % 26) as u8;
        let c = b'a' + (i % 26) as u8;
        fanout_lines.push(format!(
            "host{}{}{} svc request id {} status ok",
            a as char,
            b as char,
            c as char,
            rng.gen_range(10_000)
        ));
    }

    {
        let mut group = c.benchmark_group("drain3");
        group.throughput(Throughput::Elements(train_merge.len() as u64));
        group.bench_function("train_merge", |b| {
            b.iter(|| train(black_box(train_merge), Config::default()).unwrap());
        });
        group.finish();
    }

    {
        let mut group = c.benchmark_group("drain3");
        group.throughput(Throughput::Elements(fanout_lines.len() as u64));
        group.bench_function("train_fanout", |b| {
            b.iter(|| train(black_box(&fanout_lines), Config::default()).unwrap());
        });
        group.finish();
    }

    let matcher = train(train_merge, Config::default()).unwrap();

    {
        let mut group = c.benchmark_group("drain3");
        group.throughput(Throughput::Elements(merge_lines.len() as u64));
        group.bench_function("match_into", |b| {
            let mut scratch = Vec::with_capacity(16);
            b.iter(|| {
                let mut matched = 0usize;
                for line in &merge_lines {
                    let (_, ok) = matcher.match_into(line, &mut scratch);
                    if ok {
                        matched += 1;
                    }
                }
                black_box(matched)
            });
        });
        group.finish();
    }

    let miss_lines: Vec<String> = merge_lines
        .iter()
        .map(|l| format!("zzz-unknown {}", l))
        .collect();
    {
        let mut group = c.benchmark_group("drain3");
        group.throughput(Throughput::Elements(miss_lines.len() as u64));
        group.bench_function("match_miss", |b| {
            let mut scratch = Vec::with_capacity(16);
            b.iter(|| {
                let mut matched = 0usize;
                for line in &miss_lines {
                    let (_, ok) = matcher.match_into(line, &mut scratch);
                    if ok {
                        matched += 1;
                    }
                }
                black_box(matched)
            });
        });
        group.finish();
    }

    rng = Rng::new(99);
    let mut big_lines = Vec::with_capacity(30_000);
    for i in 0..30_000 {
        let tc = 6 + i % 8;
        let a = b'a' + ((i / 676) % 26) as u8;
        let b = b'a' + ((i / 26) % 26) as u8;
        let c = b'a' + (i % 26) as u8;
        let host = format!("host{}{}{}", a as char, b as char, c as char);
        let mut line = host;
        for t in 1..tc {
            line.push(' ');
            match t % 4 {
                0 => line.push_str(&format!("req-{}", rng.gen_range(10_000))),
                1 => line.push_str("status"),
                2 => line.push_str("ok"),
                3 => line.push_str(&format!("code-{}", rng.gen_range(1000))),
                _ => unreachable!(),
            }
        }
        big_lines.push(line);
    }

    let matcher_big = train(&big_lines, Config::default()).unwrap();

    {
        let mut group = c.benchmark_group("drain3");
        group.throughput(Throughput::Elements(big_lines.len() as u64));
        group.bench_function("match_bigdict_hit", |b| {
            let mut scratch = Vec::with_capacity(16);
            b.iter(|| {
                let mut matched = 0usize;
                for line in &big_lines {
                    let (_, ok) = matcher_big.match_into(line, &mut scratch);
                    if ok {
                        matched += 1;
                    }
                }
                black_box(matched)
            });
        });
        group.finish();
    }

    let big_miss: Vec<String> = big_lines
        .iter()
        .map(|l| format!("zzzzz-unknown {}", l))
        .collect();
    {
        let mut group = c.benchmark_group("drain3");
        group.throughput(Throughput::Elements(big_miss.len() as u64));
        group.bench_function("match_bigdict_miss", |b| {
            let mut scratch = Vec::with_capacity(16);
            b.iter(|| {
                let mut matched = 0usize;
                for line in &big_miss {
                    let (_, ok) = matcher_big.match_into(line, &mut scratch);
                    if ok {
                        matched += 1;
                    }
                }
                black_box(matched)
            });
        });
        group.finish();
    }

    {
        let matcher_arc = Arc::new(train(train_merge, Config::default()).unwrap());
        let lines_arc = Arc::new(merge_lines.clone());
        let mut group = c.benchmark_group("concurrent_match");
        for n_threads in [1usize, 2, 4] {
            group.throughput(Throughput::Elements((n_threads * 1000) as u64));
            group.bench_function(format!("{}t", n_threads), |b| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n_threads)
                        .map(|_| {
                            let m = matcher_arc.clone();
                            let lines = lines_arc.clone();
                            thread::spawn(move || {
                                let mut matched = 0usize;
                                for line in lines.iter() {
                                    if m.find(line).is_some() {
                                        matched += 1;
                                    }
                                }
                                black_box(matched)
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_drain3);
criterion_main!(benches);