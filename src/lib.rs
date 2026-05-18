//! drain3 — fast log template extraction via fixed-depth prefix trees.
//!
//! Rust port of [logpai/Drain3](https://github.com/logpai/Drain3). Splits log lines into tokens,
//! clusters them by a prefix tree keyed on token count, and replaces
//! variable tokens with a param placeholder (`<*>` by default).
//!
//! # Example
#![forbid(unsafe_code)]
//! ```
//! use drain3::Config;
//!
//! # fn main() -> Result<(), drain3::Error> {
//! let samples: Vec<String> = vec![
//!     "connection from 10.0.0.1 timeout after 5000ms".into(),
//!     "connection from 10.0.0.2 timeout after 3000ms".into(),
//!     "connection from 10.0.0.3 timeout after 1000ms".into(),
//! ];
//! let matcher = drain3::train(&samples, Config::default())?;
//! let (id, args, ok) = matcher.match_line("connection from 192.168.1.1 timeout after 42ms");
//! assert!(ok);
//! assert_eq!(args, vec!["192.168.1.1", "42ms"]);
//! # Ok(())
//! # }
//! ```
use snafu::Snafu;
use std::cell::RefCell;
use std::collections::HashMap;
mod dict;
mod prefilter;

/// Errors that can occur during training or template reconstruction.
#[derive(Debug, Snafu)]
pub enum Error {
    /// Tree depth is below the minimum of 3.
    #[snafu(display("depth must be >= 3, got {got}"))]
    InvalidDepth { got: usize },
    /// Similarity threshold is outside [0, 1].
    #[snafu(display("similarity threshold must be in [0, 1], got {got}"))]
    InvalidSimilarityThreshold { got: f64 },
    /// Match threshold is outside [0, 1].
    #[snafu(display("match threshold must be in [0, 1], got {got}"))]
    InvalidMatchThreshold { got: f64 },
    /// Max children is below the minimum of 2.
    #[snafu(display("max children must be >= 2, got {got}"))]
    InvalidMaxChildren { got: usize },
    /// Max tokens must be >= 1.
    #[snafu(display("max tokens must be >= 1, got {got}"))]
    InvalidMaxTokens { got: usize },
    /// Max bytes must be >= 1.
    #[snafu(display("max bytes must be >= 1, got {got}"))]
    InvalidMaxBytes { got: usize },
    /// Param string was empty.
    #[snafu(display("param string must not be empty"))]
    EmptyParamString,
    /// Template id must be > 0.
    #[snafu(display("template id must be > 0, got {id}"))]
    InvalidTemplateId { id: usize },
    /// Duplicate template id encountered.
    #[snafu(display("duplicate template id {id}"))]
    DuplicateTemplateId { id: usize },
    /// Template count must be > 0.
    #[snafu(display("template {id} count must be > 0"))]
    ZeroCountTemplate { id: usize },
}

/// Sentinel for tokens that are not present in the dictionary.
pub(crate) const UNKNOWN_TOKEN_ID: TokenId = TokenId(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TokenId(pub(crate) u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ClusterId(pub(crate) usize);

/// Controls training and matching behavior.
#[derive(Debug, Clone, PartialEq, bon::Builder)]
pub struct Config {
    #[builder(default = 4)]
    depth: usize,
    #[builder(default = 0.5)]
    similarity_threshold: f64,
    #[builder(default = 1.0)]
    match_threshold: f64,
    #[builder(default = 100)]
    max_children: usize,
    #[builder(default = 64)]
    max_tokens: usize,
    #[builder(default = 1024)]
    max_bytes: usize,
    #[builder(default = 0)]
    max_clusters: usize,
    #[builder(default = "<*>".to_string())]
    param_string: String,
    #[builder(default = true)]
    parametrize_numeric_tokens: bool,
    #[builder(default)]
    extra_delimiters: Vec<String>,
    #[builder(default = true)]
    enable_match_prefilter: bool,
}

impl Config {
    /// Prefix tree depth. Must be >= 3.
    pub fn depth(&self) -> usize {
        self.depth
    }
    /// Fraction of tokens that must match for a line to join a cluster during training.
    pub fn similarity_threshold(&self) -> f64 {
        self.similarity_threshold
    }
    /// Fraction of tokens that must match for a line to be considered a match.
    pub fn match_threshold(&self) -> f64 {
        self.match_threshold
    }
    /// Max children per inner node. One slot is reserved for the param catch-all.
    pub fn max_children(&self) -> usize {
        self.max_children
    }
    /// Max tokens per line. Lines with more tokens are skipped.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    /// Max bytes per line. Lines longer than this are skipped.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
    /// Max clusters. 0 = unlimited.
    pub fn max_clusters(&self) -> usize {
        self.max_clusters
    }
    /// Placeholder string for param tokens.
    pub fn param_string(&self) -> &str {
        &self.param_string
    }
    /// Whether numeric tokens are automatically parameterized during tree insertion.
    pub fn parametrize_numeric_tokens(&self) -> bool {
        self.parametrize_numeric_tokens
    }
    /// Additional delimiter strings to replace with spaces before tokenization.
    pub fn extra_delimiters(&self) -> &[String] {
        &self.extra_delimiters
    }
    /// Enable the first/last token prefilter optimization for matching.
    pub fn enable_match_prefilter(&self) -> bool {
        self.enable_match_prefilter
    }

    fn validate(&self) -> Result<(), Error> {
        if self.depth < 3 {
            return Err(Error::InvalidDepth { got: self.depth });
        }
        if !(0.0..=1.0).contains(&self.similarity_threshold) {
            return Err(Error::InvalidSimilarityThreshold {
                got: self.similarity_threshold,
            });
        }
        if !(0.0..=1.0).contains(&self.match_threshold) {
            return Err(Error::InvalidMatchThreshold {
                got: self.match_threshold,
            });
        }
        if self.max_children < 2 {
            return Err(Error::InvalidMaxChildren {
                got: self.max_children,
            });
        }
        if self.max_tokens < 1 {
            return Err(Error::InvalidMaxTokens {
                got: self.max_tokens,
            });
        }
        if self.max_bytes < 1 {
            return Err(Error::InvalidMaxBytes {
                got: self.max_bytes,
            });
        }
        if self.param_string.is_empty() {
            return Err(Error::EmptyParamString);
        }
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: 4,
            similarity_threshold: 0.5,
            match_threshold: 1.0,
            max_children: 100,
            max_tokens: 64,
            max_bytes: 1024,
            max_clusters: 0,
            param_string: "<*>".to_string(),
            parametrize_numeric_tokens: true,
            extra_delimiters: Vec::new(),
            enable_match_prefilter: true,
        }
    }
}

/// A trained log template.
#[derive(Debug, Clone, PartialEq)]
pub struct Template {
    id: usize,
    tokens: Vec<String>,
    params: Vec<bool>,
    token_count: usize,
    count: usize,
}

impl Template {
    /// Cluster id.
    pub fn id(&self) -> usize {
        self.id
    }
    /// Dense token list: only non-param tokens, in order.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
    /// Whether position `idx` is a param placeholder.
    ///
    /// # Panics
    /// Panics if `idx` is out of bounds (>= `token_count`).
    pub fn is_param(&self, idx: usize) -> bool {
        self.params[idx]
    }
    /// Total number of positions (len(tokens) + param_count).
    pub fn token_count(&self) -> usize {
        self.token_count
    }
    /// Number of matching log lines.
    pub fn count(&self) -> usize {
        self.count
    }
}
pub(crate) struct Cluster {
    id: ClusterId,
    count: usize,
    param_count: usize,
    token_str: Vec<String>,
    token_ids: Vec<TokenId>,
    /// Indices of non-param tokens, for fast scoring.
    non_param_idx: Vec<usize>,
    anchor0: Option<usize>,
    anchor1: Option<usize>,
}
impl Cluster {
    fn new(
        id: ClusterId,
        token_str: Vec<String>,
        token_ids: Vec<TokenId>,
        param_id: TokenId,
    ) -> Self {
        let mut s = Self {
            id,
            count: 1,
            param_count: 0,
            token_str,
            token_ids,
            non_param_idx: Vec::new(),
            anchor0: None,
            anchor1: None,
        };
        s.rebuild_non_param_idx(param_id);
        s
    }
    fn rebuild_non_param_idx(&mut self, param_id: TokenId) {
        self.non_param_idx.clear();
        self.param_count = 0;
        for (i, &tid) in self.token_ids.iter().enumerate() {
            if tid == param_id {
                self.param_count += 1;
            } else {
                self.non_param_idx.push(i);
            }
        }
        self.anchor0 = self.non_param_idx.first().copied();
        // anchor1 = last only when distinct from anchor0 (i.e., >= 2 non-param tokens)
        self.anchor1 = if self.non_param_idx.len() >= 2 {
            self.non_param_idx.last().copied()
        } else {
            None
        };
    }
    fn extract_args_into(&self, line_tokens: &[String], param_id: TokenId, dst: &mut Vec<String>) {
        if self.token_ids.is_empty() || line_tokens.is_empty() || self.param_count == 0 {
            return;
        }
        let limit = self.token_ids.len().min(line_tokens.len());
        dst.clear();
        for (i, tok) in line_tokens.iter().enumerate().take(limit) {
            if self.token_ids[i] == param_id {
                dst.push(tok.clone());
            }
        }
    }
    fn to_template(&self, param_id: TokenId) -> Template {
        let token_count = self.token_ids.len();
        let mut params = vec![false; token_count];
        let mut dense = Vec::with_capacity(token_count - self.param_count);
        for (i, &tid) in self.token_ids.iter().enumerate() {
            if tid == param_id {
                params[i] = true;
            } else {
                dense.push(self.token_str[i].clone());
            }
        }
        Template {
            id: self.id.0,
            tokens: dense,
            params,
            token_count,
            count: self.count,
        }
    }
}
struct Node {
    children: HashMap<TokenId, usize>,
    cluster_ids: Vec<ClusterId>,
}
impl Node {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            cluster_ids: Vec::new(),
        }
    }
}
/// A trained DRAIN matcher. Holds the prefix tree, token dictionary, and
/// precomputed indices for fast line matching.
///
/// Create via [`train`], [`train`], or [`matcher_from_templates`].
pub struct Matcher {
    cfg: Config,
    templates: Vec<Template>,
    nodes: Vec<Node>,
    root_by_len: Vec<Option<usize>>,
    clusters: Vec<Option<Cluster>>,
    dict_ids: HashMap<String, TokenId>,
    dict_next_id: TokenId,
    param_id: TokenId,
    next_cluster: ClusterId,
    match_needed: Vec<usize>,
    prefilter_buckets: Vec<prefilter::PrefilterBucket>,
    dict_frozen: Option<dict::FrozenDict>,
    has_param_first: bool,
    scratch_tok: RefCell<Vec<String>>,
}
impl Matcher {
    /// Create a new unfinalized matcher with the given config.
    ///
    /// The matcher is not ready for matching until `finalize_training` is
    /// called. Prefer the crate-level constructors [`train`] or
    /// [`matcher_from_templates`] for typical use.
    pub fn new(cfg: Config) -> Self {
        let mut m = Self {
            cfg: cfg.clone(),
            templates: Vec::new(),
            nodes: Vec::new(),
            root_by_len: Vec::new(),
            clusters: vec![None], // 0 is sentinel
            dict_ids: HashMap::new(),
            dict_next_id: TokenId(1),
            param_id: TokenId(0),
            next_cluster: ClusterId(1),
            match_needed: Vec::new(),
            prefilter_buckets: Vec::new(),
            dict_frozen: None,
            has_param_first: false,
            scratch_tok: RefCell::new(Vec::new()),
        };
        m.param_id = m.intern_token(cfg.param_string());
        m
    }
    fn freeze_dict(&mut self) {
        let entries: Vec<(String, TokenId)> =
            self.dict_ids.iter().map(|(k, v)| (k.clone(), *v)).collect();
        self.dict_frozen = Some(dict::FrozenDict::new(entries));
        let mut scratch = self.scratch_tok.borrow_mut();
        if scratch.capacity() < self.cfg.max_tokens() {
            *scratch = Vec::with_capacity(self.cfg.max_tokens());
        }
        self.has_param_first = false;
        for c in self.clusters.iter().filter_map(|c| c.as_ref()) {
            if !c.token_ids.is_empty() && c.token_ids[0] == self.param_id {
                self.has_param_first = true;
                break;
            }
        }
    }
    fn resolve_token_id(&self, token: &str) -> TokenId {
        if let Some(ref dict) = self.dict_frozen {
            dict.lookup(token).unwrap_or(self.param_id)
        } else {
            self.dict_ids.get(token).copied().unwrap_or(self.param_id)
        }
    }
    fn intern_token(&mut self, token: &str) -> TokenId {
        if let Some(&id) = self.dict_ids.get(token) {
            return id;
        }
        let id = self.dict_next_id;
        self.dict_next_id = TokenId(self.dict_next_id.0 + 1);
        self.dict_ids.insert(token.to_string(), id);
        id
    }
    fn intern_token_ids(&mut self, tokens: &[String], dst: &mut Vec<TokenId>) {
        dst.clear();
        dst.reserve(tokens.len());
        for t in tokens {
            dst.push(self.intern_token(t));
        }
    }
    fn required_score(&self, token_count: usize, sim_th: f64) -> usize {
        if sim_th == self.cfg.match_threshold() && token_count < self.match_needed.len() {
            return self.match_needed[token_count];
        }
        (sim_th * token_count as f64).ceil() as usize
    }
    fn rebuild_match_needed(&mut self) {
        self.match_needed.resize(self.root_by_len.len(), 0);
        for tc in 0..self.match_needed.len() {
            self.match_needed[tc] = (self.cfg.match_threshold() * tc as f64).ceil() as usize;
        }
    }
    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Match a line. Returns `(template_id, extracted_args, matched)`.
    pub fn match_line(&self, line: &str) -> (usize, Vec<String>, bool) {
        let mut dst = Vec::new();
        let (id, ok) = self.match_into(line, &mut dst);
        (id, dst, ok)
    }
    /// Like `match_line`, but writes extracted args into `dst`.
    pub fn match_into(&self, line: &str, dst: &mut Vec<String>) -> (usize, bool) {
        let (cluster, tokens) = self.find_match(line);
        if let Some(c) = cluster {
            let id = c.id.0;
            c.extract_args_into(&tokens, self.param_id, dst);
            return (id, true);
        }
        (0, false)
    }
    /// Match a line, returning only the template id.
    pub fn match_id(&self, line: &str) -> Option<usize> {
        self.find_match(line).0.map(|c| c.id.0)
    }
    /// Return a deep copy of the matcher's config.
    pub fn config(&self) -> Config {
        self.cfg.clone()
    }
    /// Return trained templates sorted by descending count.
    ///
    /// This is a deep clone — mutations do not affect the matcher.
    pub fn templates(&self) -> Vec<Template> {
        self.templates.clone()
    }
    /// Template by cluster id.
    pub fn template_for_id(&self, id: usize) -> Option<Template> {
        let c = self.clusters.get(id)?.as_ref()?;
        Some(c.to_template(self.param_id))
    }
    // ------------------------------------------------------------------
    // Internal matching
    // ------------------------------------------------------------------
    fn tokenize_input(&self, content: &str) -> Option<Vec<String>> {
        if content.len() > self.cfg.max_bytes() {
            return None;
        }
        if self.cfg.extra_delimiters().is_empty() {
            let mut scratch = self.scratch_tok.borrow_mut();
            scratch.clear();
            let count = tokenize_whitespace_count(content, &mut scratch, self.cfg.max_tokens());
            if count == 0 || count > self.cfg.max_tokens() {
                return None;
            }
            Some(scratch.clone())
        } else {
            let t = tokenize(content, self.cfg.extra_delimiters(), self.cfg.max_tokens());
            if t.is_empty() || t.len() > self.cfg.max_tokens() {
                return None;
            }
            Some(t)
        }
    }
    fn find_match(&self, line: &str) -> (Option<&Cluster>, Vec<String>) {
        // Quick rejection: if no cluster has a param at position 0 and there
        // are no extra delimiters, an unknown first token means no match.
        if !self.has_param_first && self.cfg.extra_delimiters().is_empty() {
            if let Some(ref dict) = self.dict_frozen {
                if dict
                    .lookup(&line[..line.find(' ').unwrap_or(line.len())])
                    .is_none()
                {
                    return (None, Vec::new());
                }
            }
        }
        let Some(tokens) = self.tokenize_input(line) else {
            return (None, Vec::new());
        };
        let tc = tokens.len();
        if tc >= self.root_by_len.len() || self.root_by_len[tc].is_none() {
            return (None, tokens);
        }
        if self.cfg.enable_match_prefilter() && tc < self.prefilter_buckets.len() {
            let mut candidates = Vec::new();
            if let Some(ids) = prefilter::prefilter_candidates_compact(
                &self.prefilter_buckets,
                &self.dict_ids,
                &tokens,
                &mut candidates,
            ) {
                let cluster =
                    self.fast_match_strings(ids, &tokens, self.cfg.match_threshold(), true);
                return (cluster, tokens);
            }
            return (None, tokens);
        }
        let cluster = self.tree_search_with_threshold(&tokens, self.cfg.match_threshold(), true);
        (cluster, tokens)
    }
    fn tree_search_with_threshold(
        &self,
        tokens: &[String],
        threshold: f64,
        include_params: bool,
    ) -> Option<&Cluster> {
        let tc = tokens.len();
        if tc >= self.root_by_len.len() {
            return None;
        }
        let root_idx = self.root_by_len[tc]?;
        if tc == 0 {
            return self.nodes[root_idx]
                .cluster_ids
                .first()
                .and_then(|&id| self.clusters.get(id.0).and_then(|c| c.as_ref()));
        }
        let max_depth = self.cfg.depth().saturating_sub(2);
        let mut cur_depth = 1;
        let mut cur_idx = root_idx;
        // Use a stack-allocated buffer for small token counts, Vec for larger.
        if tc <= 128 {
            let mut ids = [UNKNOWN_TOKEN_ID; 128];
            for (i, tok) in tokens.iter().enumerate() {
                ids[i] = self.resolve_token_id(tok);
            }
            for &tid in ids[..tc].iter() {
                if cur_depth >= max_depth || cur_depth == tc {
                    break;
                }
                let next = self.nodes[cur_idx]
                    .children
                    .get(&tid)
                    .copied()
                    .or_else(|| self.nodes[cur_idx].children.get(&self.param_id).copied());
                cur_idx = next?;
                cur_depth += 1;
            }
        } else {
            let ids: Vec<TokenId> = tokens
                .iter()
                .map(|tok| self.resolve_token_id(tok))
                .collect();
            for &tid in ids.iter() {
                if cur_depth >= max_depth || cur_depth == tc {
                    break;
                }
                let next = self.nodes[cur_idx]
                    .children
                    .get(&tid)
                    .copied()
                    .or_else(|| self.nodes[cur_idx].children.get(&self.param_id).copied());
                cur_idx = next?;
                cur_depth += 1;
            }
        }
        self.fast_match_strings(
            &self.nodes[cur_idx].cluster_ids,
            tokens,
            threshold,
            include_params,
        )
    }
    fn fast_match_strings(
        &self,
        cluster_ids: &[ClusterId],
        tokens: &[String],
        threshold: f64,
        include_params: bool,
    ) -> Option<&Cluster> {
        let n_tokens = tokens.len();
        let needed = self.required_score(n_tokens, threshold);

        // Fast path: at threshold=1.0 with include_params, return the first
        // perfect match — every non-param token must match exactly.
        if include_params && threshold >= 1.0 {
            'next_candidate: for &cid in cluster_ids {
                let Some(c) = self.clusters.get(cid.0).and_then(|c| c.as_ref()) else {
                    continue;
                };
                if c.token_str.len() != n_tokens {
                    continue;
                }
                if let Some(a) = c.anchor0 {
                    if c.token_str[a] != tokens[a] {
                        continue;
                    }
                }
                if let Some(a) = c.anchor1 {
                    if c.token_str[a] != tokens[a] {
                        continue;
                    }
                }
                for &idx in &c.non_param_idx {
                    if Some(idx) == c.anchor0 || Some(idx) == c.anchor1 {
                        continue;
                    }
                    if c.token_str[idx] != tokens[idx] {
                        continue 'next_candidate;
                    }
                }
                return Some(c);
            }
            return None;
        }

        let mut max_score: isize = -1;
        let mut max_param_count: isize = -1;
        let mut max_cluster: Option<&Cluster> = None;
        for &cid in cluster_ids {
            let Some(c) = self.clusters.get(cid.0).and_then(|c| c.as_ref()) else {
                continue;
            };
            if c.token_str.len() != n_tokens {
                continue;
            }
            let param_count = c.param_count;
            let mut sim_tokens = if include_params { param_count } else { 0 };
            let np_idx = &c.non_param_idx;
            let mut remaining = np_idx.len();

            // Anchor checks: score first and last non-param positions first.
            if let Some(a) = c.anchor0 {
                if c.token_str[a] == tokens[a] {
                    sim_tokens += 1;
                }
                remaining -= 1;
                if sim_tokens + remaining < needed {
                    continue;
                }
            }
            if let Some(a) = c.anchor1 {
                if c.token_str[a] == tokens[a] {
                    sim_tokens += 1;
                }
                remaining -= 1;
                if sim_tokens + remaining < needed {
                    continue;
                }
            }

            for &idx in np_idx {
                if Some(idx) == c.anchor0 || Some(idx) == c.anchor1 {
                    continue;
                }
                if c.token_str[idx] == tokens[idx] {
                    sim_tokens += 1;
                }
                remaining -= 1;
                if sim_tokens + remaining < needed {
                    break;
                }
            }
            if sim_tokens as isize > max_score
                || (sim_tokens as isize == max_score && param_count as isize > max_param_count)
            {
                max_score = sim_tokens as isize;
                max_param_count = param_count as isize;
                max_cluster = Some(c);
            }
        }
        if max_score >= needed as isize {
            max_cluster
        } else {
            None
        }
    }
    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------
    fn create_cluster(&mut self, tokens: Vec<String>) -> ClusterId {
        let mut token_ids = Vec::new();
        self.intern_token_ids(&tokens, &mut token_ids);
        let id = self.next_cluster;
        self.next_cluster = ClusterId(self.next_cluster.0 + 1);
        let cl = Cluster::new(id, tokens, token_ids, self.param_id);
        if id.0 >= self.clusters.len() {
            self.clusters.resize_with(id.0 + 1, || None);
        }
        self.clusters[id.0] = Some(cl);
        self.add_seq_to_prefix_tree(id);
        id
    }
    /// Add a single log line to an unfinalized matcher.
    ///
    /// Returns the [`Template`] for the matched or newly created cluster.
    /// Returns `None` if the line is rejected (too long, too many tokens, or
    /// max clusters reached).
    ///
    /// # Panics
    /// Panics if called on a matcher that has already been finalized.
    pub fn add_log_message(&mut self, content: &str) -> Option<Template> {
        let tokens = self.tokenize_input(content)?;
        let tc = tokens.len();
        if tc >= self.root_by_len.len() {
            // First log with this token count: create cluster + tree path.
            let cid = self.create_cluster(tokens);
            return self.clusters[cid.0]
                .as_ref()
                .map(|c| c.to_template(self.param_id));
        }
        if let Some(c) =
            self.tree_search_with_threshold(&tokens, self.cfg.similarity_threshold(), false)
        {
            let cid = c.id;
            let mut changed = false;
            let cluster = self.clusters[cid.0]
                .as_mut()
                .expect("cluster was just found via tree search");
            for (i, tok) in tokens.iter().enumerate().take(cluster.token_str.len()) {
                if cluster.token_ids[i] == self.param_id {
                    continue;
                }
                if cluster.token_str[i] != *tok {
                    cluster.token_ids[i] = self.param_id;
                    cluster.token_str[i] = self.cfg.param_string().to_string();
                    cluster.param_count += 1;
                    changed = true;
                }
            }
            if changed {
                cluster.rebuild_non_param_idx(self.param_id);
            }
            cluster.count += 1;
            return Some(cluster.to_template(self.param_id));
        }
        // No match — create new cluster
        if self.cfg.max_clusters() > 0 && self.next_cluster.0 > self.cfg.max_clusters() {
            return None;
        }
        let cid = self.create_cluster(tokens);
        self.clusters[cid.0]
            .as_ref()
            .map(|c| c.to_template(self.param_id))
    }
    /// Read-only lookup: find the best matching template for a line without
    /// mutating the matcher.
    ///
    /// Returns `None` if no cluster matches.
    pub fn find(&self, content: &str) -> Option<Template> {
        let (cluster, _) = self.find_match(content);
        cluster.map(|c| c.to_template(self.param_id))
    }
    fn add_seq_to_prefix_tree(&mut self, cluster_id: ClusterId) {
        let cluster = self.clusters[cluster_id.0]
            .as_ref()
            .expect("cluster exists by construction");
        let tc = cluster.token_ids.len();
        if tc >= self.root_by_len.len() {
            self.root_by_len.resize_with(tc + 1, || None);
        }
        if self.root_by_len[tc].is_none() {
            let idx = self.nodes.len();
            self.nodes.push(Node::new());
            self.root_by_len[tc] = Some(idx);
        }
        let mut cur_idx = self.root_by_len[tc].expect("root was just created if missing");
        if tc == 0 {
            self.nodes[cur_idx].cluster_ids.push(cluster_id);
            return;
        }
        for (i, &token_id) in cluster.token_ids.iter().enumerate() {
            let cur_depth = i + 1;
            if cur_depth >= self.cfg.depth() - 2 || cur_depth >= tc {
                self.nodes[cur_idx].cluster_ids.push(cluster_id);
                break;
            }
            let key = {
                let node = &self.nodes[cur_idx];
                if node.children.contains_key(&token_id) {
                    token_id
                } else if self.cfg.parametrize_numeric_tokens()
                    && has_numbers(&cluster.token_str[i])
                {
                    self.param_id
                } else {
                    let specific_count = node.children.len();
                    let has_wild = node.children.contains_key(&self.param_id);
                    let available = self.cfg.max_children() - 1;
                    if specific_count < available
                        || (!has_wild && specific_count < self.cfg.max_children() - 1)
                    {
                        token_id
                    } else {
                        self.param_id
                    }
                }
            };
            if !self.nodes[cur_idx].children.contains_key(&key) {
                let new_idx = self.nodes.len();
                self.nodes.push(Node::new());
                self.nodes[cur_idx].children.insert(key, new_idx);
            }
            cur_idx = self.nodes[cur_idx].children[&key];
        }
    }
    fn sync_templates_from_clusters(&mut self) {
        let mut out: Vec<Template> = Vec::with_capacity(self.clusters.len().saturating_sub(1));
        for id in 1..self.clusters.len() {
            let Some(c) = self.clusters[id].as_ref() else {
                continue;
            };
            out.push(c.to_template(self.param_id));
        }
        out.sort_by_key(|b| std::cmp::Reverse(b.count()));
        self.templates = out;
    }
    fn finalize_training(&mut self) {
        self.sync_templates_from_clusters();
        self.rebuild_match_needed();
        self.prefilter_buckets = prefilter::rebuild_match_prefilter(&self.clusters, self.param_id);
        self.freeze_dict();
    }
}
/// Train a matcher with the provided config.
pub fn train(samples: &[String], cfg: Config) -> Result<Matcher, Error> {
    cfg.validate()?;
    let mut m = Matcher::new(cfg);
    for sample in samples {
        m.add_log_message(sample);
    }
    m.finalize_training();
    Ok(m)
}
/// Rebuild a matcher from pre-existing templates.
pub fn matcher_from_templates(cfg: Config, templates: &[Template]) -> Result<Matcher, Error> {
    cfg.validate()?;
    let mut m = Matcher::new(cfg);
    if templates.is_empty() {
        m.finalize_training();
        return Ok(m);
    }
    let mut sorted: Vec<_> = templates.to_vec();
    sorted.sort_by_key(|t| t.id());
    let mut seen = std::collections::HashSet::new();
    let mut max_id = 0;
    for t in &sorted {
        if t.id() == 0 {
            return Err(Error::InvalidTemplateId { id: t.id() });
        }
        if !seen.insert(t.id()) {
            return Err(Error::DuplicateTemplateId { id: t.id() });
        }
        if t.count() == 0 {
            return Err(Error::ZeroCountTemplate { id: t.id() });
        }
        if t.id() > max_id {
            max_id = t.id();
        }
    }
    m.clusters.resize_with(max_id + 1, || None);
    m.next_cluster = ClusterId(max_id + 1);
    for t in &sorted {
        let mut full = vec![String::new(); t.token_count()];
        let mut dense_idx = 0;
        for (i, slot) in full.iter_mut().enumerate().take(t.token_count()) {
            if t.is_param(i) {
                *slot = m.cfg.param_string().to_string();
            } else {
                *slot = t.tokens()[dense_idx].clone();
                dense_idx += 1;
            }
        }
        let mut token_ids = Vec::new();
        m.intern_token_ids(&full, &mut token_ids);
        let cl = Cluster::new(ClusterId(t.id()), full, token_ids, m.param_id);
        m.clusters[t.id()] = Some(cl);
    }
    for id in 1..m.clusters.len() {
        if m.clusters[id].is_some() {
            m.add_seq_to_prefix_tree(ClusterId(id));
        }
    }
    m.finalize_training();
    Ok(m)
}
/// Single-pass tokenizer that splits on individual space characters.
/// Does **not** trim and **does** preserve empty tokens, matching Go's
/// `tokenizeWhitespaceCount` so that consecutive spaces increase the token
/// count (and therefore change the tree bucket).
fn tokenize_whitespace_count(content: &str, dst: &mut Vec<String>, max_tokens: usize) -> usize {
    if content.is_empty() || max_tokens == 0 {
        return 0;
    }
    dst.clear();
    let bytes = content.as_bytes();
    let mut start = 0;
    let mut count = 1;
    for i in 0..bytes.len() {
        if bytes[i] != b' ' {
            continue;
        }
        dst.push(
            std::str::from_utf8(&bytes[start..i])
                .expect("ascii space split preserves utf8")
                .to_string(),
        );
        start = i + 1;
        if count >= max_tokens {
            return count + 1; // signal overflow
        }
        count += 1;
    }
    dst.push(
        std::str::from_utf8(&bytes[start..])
            .expect("ascii space split preserves utf8")
            .to_string(),
    );
    count
}

/// Tokenizer used when `extra_delimiters` are configured. Matches Go's
/// `tokenize`: trim, replace delimiters with spaces, then collapse runs of
/// spaces and skip empty fields (like `strings.Fields`).
fn tokenize(content: &str, extra_delimiters: &[String], max_tokens: usize) -> Vec<String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut s = trimmed.to_string();
    for d in extra_delimiters {
        if !d.is_empty() {
            s = s.replace(d, " ");
        }
    }
    s.split(' ')
        .filter(|t| !t.is_empty())
        .take(max_tokens)
        .map(|t| t.to_string())
        .collect()
}
fn has_numbers(s: &str) -> bool {
    s.bytes().any(|b| b.is_ascii_digit())
}

/// Precomputed recipe for rendering a template with supplied parameter values.
#[derive(Debug, Clone)]
pub struct RenderPlan {
    head: Vec<u8>,
    segments: Vec<RenderSegment>,
    max_size: usize,
}

#[derive(Debug, Clone)]
struct RenderSegment {
    arg_idx: usize,
    tail: Vec<u8>,
}

impl RenderPlan {
    /// Build a render plan for `t`.
    ///
    /// `max_arg_len` is optional. When provided, it is called once for each
    /// parameter position and included in `max_size`.
    pub fn new(t: &Template, max_arg_len: Option<&dyn Fn(usize) -> usize>) -> Self {
        let mut head: Vec<u8> = Vec::new();
        let mut segments: Vec<RenderSegment> = Vec::new();
        let mut arg_idx = 0usize;
        let mut tok_idx = 0usize;
        let mut cur: Vec<u8> = Vec::new();

        for i in 0..t.token_count() {
            if i > 0 {
                cur.push(b' ');
            }
            if t.is_param(i) {
                if let Some(last) = segments.last_mut() {
                    last.tail = cur;
                } else {
                    head = cur;
                }
                segments.push(RenderSegment {
                    arg_idx,
                    tail: Vec::new(),
                });
                cur = Vec::new();
                arg_idx += 1;
            } else {
                cur.extend_from_slice(t.tokens()[tok_idx].as_bytes());
                tok_idx += 1;
            }
        }
        if let Some(last) = segments.last_mut() {
            last.tail = cur;
        } else {
            head = cur;
        }

        let mut max_size = head.len();
        for seg in &segments {
            max_size += seg.tail.len();
            if let Some(f) = max_arg_len {
                max_size += f(seg.arg_idx);
            }
        }
        Self {
            head,
            segments,
            max_size,
        }
    }

    /// Upper-bound size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Render into `dst`. Missing argument positions render as empty strings.
    pub fn append(&self, dst: &mut Vec<u8>, args: Option<&[&str]>) {
        dst.extend_from_slice(&self.head);
        for seg in &self.segments {
            if let Some(s) = args.and_then(|a| a.get(seg.arg_idx)) {
                dst.extend_from_slice(s.as_bytes());
            }
            dst.extend_from_slice(&seg.tail);
        }
    }
}

/// Deterministically sample lines as fixed-size blocks at regular strides
/// with random jitter inside each stride window.
///
/// The target sample count is `frac * len(lines)`, but the actual returned
/// count is rounded up to the nearest multiple of `block_size` (capped by the
/// input length) because entire blocks are appended per stride.
///
/// Uses a seeded rng derived from the input length so the result is
/// deterministic — same input produces the same sample across runs.
pub fn stride_sample(lines: &[String], frac: f64, block_size: usize) -> Vec<String> {
    let total = lines.len();
    if total == 0 {
        return Vec::new();
    }
    let sample_n = (total as f64 * frac) as usize;
    if sample_n == 0 {
        return Vec::new();
    }
    let num_blocks = (sample_n / block_size).max(1);
    let stride = (total / num_blocks).max(block_size);
    let mut rng = fastrand::Rng::with_seed(total as u64);
    let mut out: Vec<String> = Vec::with_capacity(sample_n);
    let mut start = 0usize;
    while start < total && out.len() < sample_n {
        let max_offset = stride.saturating_sub(block_size).max(1);
        let offset = start + (rng.u32(..max_offset as u32) as usize);
        if offset >= total {
            break;
        }
        let end = (offset + block_size).min(total);
        out.extend(lines[offset..end].iter().cloned());
        start += stride;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Expand a dense Template into its full token sequence, inserting
    /// `param_str` at each position marked in `params`.
    fn render_template_placeholders(t: &Template, param_str: &str) -> String {
        let mut out: Vec<String> = Vec::with_capacity(t.token_count());
        let mut dense_idx = 0;
        for i in 0..t.token_count() {
            if t.is_param(i) {
                out.push(param_str.to_string());
            } else {
                out.push(t.tokens()[dense_idx].clone());
                dense_idx += 1;
            }
        }
        out.join(" ")
    }
    // -----------------------------------------------------------------
    // Reference behaviour: scenarios ported from logpai/Drain3 test_drain.py
    // -----------------------------------------------------------------
    #[test]
    fn logpai_sshd_scenario() {
        let samples: Vec<String> = vec![
            "Dec 10 07:07:38 LabSZ sshd[24206]: input_userauth_request: invalid user test9 [preauth]".into(),
            "Dec 10 07:08:28 LabSZ sshd[24208]: input_userauth_request: invalid user webmaster [preauth]".into(),
            "Dec 10 09:12:32 LabSZ sshd[24490]: Failed password for invalid user ftpuser from 0.0.0.0 port 62891 ssh2".into(),
            "Dec 10 09:12:35 LabSZ sshd[24492]: Failed password for invalid user pi from 0.0.0.0 port 49289 ssh2".into(),
            "Dec 10 09:12:44 LabSZ sshd[24501]: Failed password for invalid user ftpuser from 0.0.0.0 port 60836 ssh2".into(),
            "Dec 10 07:28:03 LabSZ sshd[24245]: input_userauth_request: invalid user pgadmin [preauth]".into(),
        ];
        let cfg = Config::builder()
            .similarity_threshold(0.4) // logpai default
            .build();
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
            let key = render_template_placeholders(&tmpl, cfg.param_string());
            *got.entry(key).or_insert(0) += tmpl.count();
            total += tmpl.count();
        }
        assert_eq!(got, want, "templates mismatch");
        assert_eq!(total, samples.len(), "total count mismatch");
    }
    #[test]
    fn logpai_sshd_scenario_high_sim() {
        let samples: Vec<String> = vec![
            "Dec 10 07:07:38 LabSZ sshd[24206]: input_userauth_request: invalid user test9 [preauth]".into(),
            "Dec 10 07:08:28 LabSZ sshd[24208]: input_userauth_request: invalid user webmaster [preauth]".into(),
            "Dec 10 09:12:32 LabSZ sshd[24490]: Failed password for invalid user ftpuser from 0.0.0.0 port 62891 ssh2".into(),
            "Dec 10 09:12:35 LabSZ sshd[24492]: Failed password for invalid user pi from 0.0.0.0 port 49289 ssh2".into(),
            "Dec 10 09:12:44 LabSZ sshd[24501]: Failed password for invalid user ftpuser from 0.0.0.0 port 60836 ssh2".into(),
            "Dec 10 07:28:03 LabSZ sshd[24245]: input_userauth_request: invalid user pgadmin [preauth]".into(),
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
            let key = render_template_placeholders(&tmpl, cfg.param_string());
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
            let key = render_template_placeholders(&tmpl, "<*>");
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
            ("aa aa tt", 1), // wildcard absorbs tt
            ("xx yy zz", 2), // exact
            ("xx yy rr", 0), // literal mismatch
            ("nothing", 0),  // unknown token count
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
    // -----------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------
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
        // match_threshold=0.0 accepts any tree-routable candidate.
        assert!(
            m.match_id("A X Y").is_some(),
            "expected match with 0.0 match threshold"
        );
        // similarity_threshold=0.0 merges aggressively: one template.
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
        let capped = train(&lines, cfg.clone()).unwrap();
        assert!(
            capped.templates().len() <= 2,
            "expected at most 2 templates, got {}",
            capped.templates().len()
        );
        let cfg = Config::builder().max_clusters(0).build();
        let full = train(&lines, cfg).unwrap();
        assert!(
            full.templates().len() > capped.templates().len(),
            "expected uncapped training to produce more templates: uncapped={} capped={}",
            full.templates().len(),
            capped.templates().len()
        );
    }
    // -----------------------------------------------------------------
    // Config validation
    // -----------------------------------------------------------------
    #[test]
    fn train_validation() {
        let cfg = Config {
            depth: 2,
            ..Config::default()
        };
        assert!(
            train(&["a b c".into()], cfg).is_err(),
            "expected error for invalid depth"
        );
    }
    #[test]
    fn zero_value_config_is_rejected() {
        let zero_cfg = Config {
            depth: 0,
            similarity_threshold: 0.0,
            match_threshold: 0.0,
            max_children: 0,
            max_tokens: 0,
            max_bytes: 0,
            max_clusters: 0,
            param_string: String::new(),
            parametrize_numeric_tokens: false,
            extra_delimiters: Vec::new(),
            enable_match_prefilter: false,
        };
        assert!(
            train(&["a b c".into()], zero_cfg).is_err(),
            "expected error for zero-value Config"
        );
    }
    // -----------------------------------------------------------------
    // Features
    // -----------------------------------------------------------------
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
            read_cfg.extra_delimiters()[0],
            "=",
            "config getter leaked mutable slice"
        );
        let templates = m.templates();
        assert_eq!(
            templates[0].tokens()[0],
            m.templates()[0].tokens()[0],
            "templates getter leaked mutable data"
        );
    }
}
