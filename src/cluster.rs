use std::sync::Arc;

use string_interner::backend::BucketBackend;
use string_interner::StringInterner;

use crate::{ClusterId, Template, TokenId};

pub(crate) struct Cluster {
    pub id: ClusterId,
    pub count: usize,
    pub param_count: usize,
    pub token_str: Vec<Arc<str>>,
    pub token_ids: Vec<TokenId>,
    pub non_param_idx: Vec<usize>,
    pub param_positions: Vec<usize>,
    pub anchor0: Option<usize>,
    pub anchor1: Option<usize>,
}

impl Cluster {
    pub fn new(
        id: ClusterId,
        token_str: Vec<Arc<str>>,
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
            param_positions: Vec::new(),
            anchor0: None,
            anchor1: None,
        };
        s.rebuild_indices(param_id);
        s
    }

    pub fn rebuild_indices(&mut self, param_id: TokenId) {
        self.non_param_idx.clear();
        self.param_positions.clear();
        self.param_count = 0;
        for (i, &tid) in self.token_ids.iter().enumerate() {
            if tid == param_id {
                self.param_count += 1;
                self.param_positions.push(i);
            } else {
                self.non_param_idx.push(i);
            }
        }
        self.anchor0 = self.non_param_idx.first().copied();
        self.anchor1 = if self.non_param_idx.len() >= 2 {
            self.non_param_idx.last().copied()
        } else {
            None
        };
    }

    pub fn to_template(
        &self,
        interner: &StringInterner<BucketBackend<usize>>,
        param_id: TokenId,
    ) -> Template {
        let token_count = self.token_ids.len();
        let mut params = vec![false; token_count];
        let mut dense = Vec::with_capacity(token_count - self.param_count);
        for (i, &tid) in self.token_ids.iter().enumerate() {
            if tid == param_id {
                params[i] = true;
            } else {
                dense.push(Arc::from(interner.resolve(usize::from(tid)).unwrap()));
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

pub(crate) struct Node {
    pub children: std::collections::HashMap<TokenId, usize>,
    pub cluster_ids: Vec<ClusterId>,
}

impl Node {
    pub fn new() -> Self {
        Self {
            children: std::collections::HashMap::new(),
            cluster_ids: Vec::new(),
        }
    }
}
