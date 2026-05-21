use drain3::{train, Config, Error};
use std::collections::HashMap;

fn render_template_placeholders(t: &drain3::Template, param_str: &str) -> String {
    let mut out: Vec<String> = Vec::with_capacity(t.token_count());
    let mut dense_idx = 0;
    for i in 0..t.token_count() {
        if t.is_param(i) {
            out.push(param_str.to_string());
        } else {
            out.push(t.tokens()[dense_idx].to_string());
            dense_idx += 1;
        }
    }
    out.join(" ")
}

#[test]
fn logpai_sshd_scenario() {
    let samples: Vec<String> = vec![
        "Dec 10 07:07:38 LabSZ sshd[24206]: input_userauth_request: invalid user test9 [preauth]"
            .into(),
        "Dec 10 07:08:28 LabSZ sshd[24208]: input_userauth_request: invalid user webmaster [preauth]"
            .into(),
        "Dec 10 09:12:32 LabSZ sshd[24490]: Failed password for invalid user ftpuser from 0.0.0.0 port 62891 ssh2"
            .into(),
        "Dec 10 09:12:35 LabSZ sshd[24492]: Failed password for invalid user pi from 0.0.0.0 port 49289 ssh2"
            .into(),
        "Dec 10 09:12:44 LabSZ sshd[24501]: Failed password for invalid user ftpuser from 0.0.0.0 port 60836 ssh2"
            .into(),
        "Dec 10 07:28:03 LabSZ sshd[24245]: input_userauth_request: invalid user pgadmin [preauth]"
            .into(),
    ];
    let cfg = Config::builder().similarity_threshold(0.4).build();
    let m = train(&samples, cfg.clone()).unwrap();
    let mut want: HashMap<String, usize> = HashMap::new();
    want.insert(
        "Dec 10 <*> LabSZ <*> input_userauth_request: invalid user <*> [preauth]".into(),
        3,
    );
    want.insert(
        "Dec 10 <*> LabSZ <*> Failed password for invalid user <*> from 0.0.0.0 port <*> ssh2"
            .into(),
        3,
    );
    let mut got: HashMap<String, usize> = HashMap::new();
    let mut total = 0;
    for tmpl in m.templates() {
        let key = render_template_placeholders(tmpl, cfg.param_string.as_ref());
        *got.entry(key).or_insert(0) += tmpl.count();
        total += tmpl.count();
    }
    assert_eq!(got, want, "templates mismatch");
    assert_eq!(total, samples.len(), "total count mismatch");
}

#[test]
fn logpai_sshd_scenario_high_sim() {
    let samples: Vec<String> = vec![
        "Dec 10 07:07:38 LabSZ sshd[24206]: input_userauth_request: invalid user test9 [preauth]"
            .into(),
        "Dec 10 07:08:28 LabSZ sshd[24208]: input_userauth_request: invalid user webmaster [preauth]"
            .into(),
        "Dec 10 09:12:32 LabSZ sshd[24490]: Failed password for invalid user ftpuser from 0.0.0.0 port 62891 ssh2"
            .into(),
        "Dec 10 09:12:35 LabSZ sshd[24492]: Failed password for invalid user pi from 0.0.0.0 port 49289 ssh2"
            .into(),
        "Dec 10 09:12:44 LabSZ sshd[24501]: Failed password for invalid user ftpuser from 0.0.0.0 port 60836 ssh2"
            .into(),
        "Dec 10 07:28:03 LabSZ sshd[24245]: input_userauth_request: invalid user pgadmin [preauth]"
            .into(),
    ];
    let cfg = Config::builder().similarity_threshold(0.75).build();
    let m = train(&samples, cfg.clone()).unwrap();
    let mut want: HashMap<String, usize> = HashMap::new();
    want.insert(samples[0].clone(), 1);
    want.insert(samples[1].clone(), 1);
    want.insert(
        "Dec 10 <*> LabSZ <*> Failed password for invalid user <*> from 0.0.0.0 port <*> ssh2"
            .into(),
        3,
    );
    want.insert(samples[5].clone(), 1);
    let mut got: HashMap<String, usize> = HashMap::new();
    let mut total = 0;
    for tmpl in m.templates() {
        let key = render_template_placeholders(tmpl, cfg.param_string.as_ref());
        *got.entry(key).or_insert(0) += tmpl.count();
        total += tmpl.count();
    }
    assert_eq!(got, want, "templates mismatch");
    assert_eq!(total, samples.len(), "total count mismatch");
}

#[test]
fn logpai_short_message() {
    let m = train(
        &["hello".into(), "hello".into(), "otherword".into()],
        Config::default(),
    )
    .unwrap();
    let mut got: HashMap<String, usize> = HashMap::new();
    for tmpl in m.templates() {
        let key = render_template_placeholders(tmpl, "<*>");
        *got.entry(key).or_insert(0) += tmpl.count();
    }
    let mut want: HashMap<String, usize> = HashMap::new();
    want.insert("hello".into(), 2);
    want.insert("otherword".into(), 1);
    assert_eq!(got, want, "templates mismatch");
}

#[test]
fn logpai_match_only() {
    let m = train(
        &[
            "aa aa aa".into(),
            "aa aa bb".into(),
            "aa aa cc".into(),
            "xx yy zz".into(),
        ],
        Config::default(),
    )
    .unwrap();
    let cases: Vec<(&str, usize)> = vec![
        ("aa aa tt", 1),
        ("xx yy zz", 2),
        ("xx yy rr", 0),
        ("nothing", 0),
    ];
    for (line, want) in cases {
        let id = m.match_id(line);
        if want == 0 {
            assert!(
                id.is_none(),
                "Match({line:?}): got id={id:?}, want no match"
            );
        } else {
            assert_eq!(
                id,
                Some(want),
                "Match({line:?}): got id={id:?}, want id={want}"
            );
        }
    }
}

#[test]
fn deterministic_templates() {
    let samples: Vec<String> = vec![
        "svc 1 INFO user 10".into(),
        "svc 2 INFO user 20".into(),
        "svc 3 ERROR user 30".into(),
        "svc 4 ERROR user 40".into(),
    ];
    let m1 = train(&samples, Config::default()).unwrap();
    let m2 = train(&samples, Config::default()).unwrap();
    assert_eq!(
        m1.templates(),
        m2.templates(),
        "templates are not deterministic"
    );
}

#[test]
fn train_handles_empty_input() {
    let m = train(&[], Config::default()).unwrap();
    assert!(m.templates().is_empty(), "expected no templates");
    assert!(m.match_id("anything").is_none(), "expected no match");
}

#[test]
fn zero_thresholds_are_valid() {
    let cfg = Config::builder()
        .similarity_threshold(0.0)
        .match_threshold(0.0)
        .build();
    let m = train(&["A B C".into(), "A B D".into()], cfg).unwrap();
    assert!(
        m.match_id("A X Y").is_some(),
        "expected match with 0.0 match threshold"
    );
    assert_eq!(
        m.templates().len(),
        1,
        "expected 1 template with 0.0 similarity"
    );
}

#[test]
fn max_clusters() {
    let lines: Vec<String> = vec![
        "alpha X Y".into(),
        "bravo X Y".into(),
        "charlie X Y".into(),
        "delta X Y".into(),
        "echo X Y".into(),
    ];
    let cfg = Config::builder().max_clusters(2).build();
    let result = train(&lines, cfg);
    assert!(result.is_err(), "train with max_clusters=2 should fail");
    let err = match result {
        Ok(_) => panic!("expected error"),
        Err(e) => e,
    };
    assert!(
        matches!(err, Error::MaxClustersReached { .. }),
        "expected MaxClustersReached"
    );
    let cfg = Config::builder().max_clusters(0).build();
    let full = train(&lines, cfg).unwrap();
    assert!(
        full.templates().len() > 2,
        "expected uncapped training to produce more than 2 templates: {}",
        full.templates().len()
    );
}

#[test]
fn train_validation() {
    let cfg = Config::builder().depth(2).build();
    assert!(
        train(&["a b c".into()], cfg).is_err(),
        "expected error for invalid depth"
    );
}

#[test]
fn zero_value_config_is_rejected() {
    let zero_cfg = Config::builder()
        .depth(0)
        .similarity_threshold(0.0)
        .match_threshold(0.0)
        .max_children(0)
        .max_tokens(0)
        .max_bytes(0)
        .max_clusters(0)
        .param_string(String::new().into())
        .parametrize_numeric_tokens(false)
        .extra_delimiters(vec![])
        .enable_match_prefilter(false)
        .build();
    assert!(
        train(&["a b c".into()], zero_cfg).is_err(),
        "expected error for zero-value Config"
    );
}

#[test]
fn extra_delimiters() {
    let cfg = Config::builder().extra_delimiters(vec!["=".into()]).build();
    let m = train(&["k=v a=1".into(), "k=v a=2".into()], cfg).unwrap();
    let (id, args, ok) = m.match_line("k=v a=7");
    assert!(ok, "expected match");
    assert_eq!(id, 1, "expected template id 1, got {id}");
    assert_eq!(args, vec!["7"], "unexpected args: {args:?}");
}

#[test]
fn match_into() {
    let samples: Vec<String> = vec![
        "service 1 level INFO user 10 action 5".into(),
        "service 2 level INFO user 20 action 5".into(),
        "service 3 level INFO user 30 action 5".into(),
    ];
    let m = train(&samples, Config::default()).unwrap();
    let line = "service 99 level INFO user 777 action 5";
    let (id_a, args_a, ok_a) = m.match_line(line);
    let mut scratch: Vec<String> = Vec::with_capacity(8);
    let (id_b, ok_b) = m.match_into(line, &mut scratch);
    assert_eq!(id_a, id_b, "MatchInto id mismatch");
    assert_eq!(ok_a, ok_b, "MatchInto ok mismatch");
    assert_eq!(args_a, scratch, "MatchInto args mismatch");
    assert!(!scratch.is_empty(), "expected extracted params");
    scratch.clear();
    let (_, ok_miss) = m.match_into("short unmatched", &mut scratch);
    assert!(!ok_miss, "expected no match");
    assert!(
        scratch.is_empty(),
        "expected empty args on miss, got {scratch:?}"
    );
}

#[test]
fn config_and_templates_are_copied() {
    let cfg = Config::builder().extra_delimiters(vec!["=".into()]).build();
    let m = train(&["k=v a=1".into(), "k=v a=2".into()], cfg).unwrap();
    let read_cfg = m.config();
    assert_eq!(
        read_cfg.extra_delimiters[0], "=",
        "config getter leaked mutable slice"
    );
    let templates = m.templates();
    assert_eq!(
        templates[0].tokens()[0],
        m.templates()[0].tokens()[0],
        "templates getter leaked mutable data"
    );
}

#[test]
fn concurrent_find_is_sync_safe() {
    use std::sync::Arc;
    use std::thread;
    let m = train(
        &["alpha 123".into(), "beta 456".into(), "gamma 789".into()],
        Config::default(),
    )
    .unwrap();
    let m = Arc::new(m);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let m = Arc::clone(&m);
            thread::spawn(move || {
                for _ in 0..1000 {
                    m.find("alpha 999");
                    m.find("beta 888");
                    m.find("gamma 777");
                    m.find("delta 666");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("thread panicked");
    }
}
