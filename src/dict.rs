use crate::TokenId;

/// Fast read-only token-to-ID lookup table.
///
/// Uses a sorted flat array with binary search.  Faster than `HashMap` for
/// read-only access because there is no hashing overhead and the layout is
/// cache-friendly.
#[derive(Debug, Clone)]
pub struct FrozenDict {
    keys: Vec<String>,
    vals: Vec<TokenId>,
}

impl FrozenDict {
    /// Build a frozen dictionary from (key, value) pairs.
    pub fn new(mut entries: Vec<(String, TokenId)>) -> Self {
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut keys = Vec::with_capacity(entries.len());
        let mut vals = Vec::with_capacity(entries.len());
        for (k, v) in entries {
            keys.push(k);
            vals.push(v);
        }
        Self { keys, vals }
    }

    pub fn lookup(&self, key: &str) -> Option<TokenId> {
        self.keys
            .binary_search_by(|probe| probe.as_str().cmp(key))
            .ok()
            .map(|idx| self.vals[idx])
    }
}
