//! 记忆系统：长期记忆条目持久化（JSON），UI 可增删查，Agent 可写入。
//!
//! 每条记忆有两种投递方式：
//! - system：直接拼进系统提示词，模型每轮都能看到（有上限，按最新优先）；
//! - on_demand：只把「类型 + 标题 + id」索引放进系统提示词，
//!   模型需要细节时调用 MemoryGuide 工具读取（与 ToolGuide 的渐进式加载同思路）。

use anyhow::Result;
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

pub const MEMORY_MODE_SYSTEM: &str = "system";
pub const MEMORY_MODE_ON_DEMAND: &str = "on_demand";

fn default_memory_mode() -> String {
    MEMORY_MODE_SYSTEM.into()
}

fn normalize_mode(mode: &str) -> String {
    match mode.trim() {
        MEMORY_MODE_ON_DEMAND => MEMORY_MODE_ON_DEMAND.into(),
        _ => MEMORY_MODE_SYSTEM.into(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub memory_type: String, // knowledge / preference / gotcha / narrative / workflow ...
    pub title: String,
    pub content: String,
    pub created_at: String,
    /// system = 直接注入系统提示词；on_demand = 仅注入索引，经 MemoryGuide 读取。
    /// 旧数据没有该字段时按 system 处理（保持原有行为）。
    #[serde(default = "default_memory_mode")]
    pub mode: String,
}

/// 一轮注入的完整结果：系统直连文本 + 按需索引 + 统计。
#[derive(Debug, Clone, Default, Serialize)]
pub struct MemoryInjection {
    pub system_text: String,
    pub on_demand_index: String,
    pub system_count: usize,
    pub on_demand_count: usize,
    pub system_chars: usize,
    pub index_chars: usize,
    pub total_count: usize,
    pub limit: usize,
}

#[derive(Clone, Default)]
pub struct MemoryStore {
    entries: Arc<RwLock<Vec<MemoryEntry>>>,
    path: PathBuf,
}

impl MemoryStore {
    pub fn init(&mut self, path: PathBuf) {
        self.path = path.clone();
        if let Ok(s) = std::fs::read_to_string(&path) {
            if let Ok(list) = serde_json::from_str::<Vec<MemoryEntry>>(&s) {
                self.entries = Arc::new(RwLock::new(list));
            }
        }
    }

    pub async fn list(&self) -> Vec<MemoryEntry> {
        self.entries.read().await.clone()
    }

    pub async fn add(&self, memory_type: &str, title: &str, content: &str, mode: &str) -> Result<MemoryEntry> {
        let entry = MemoryEntry {
            id: uuid::Uuid::new_v4().to_string(),
            memory_type: memory_type.to_string(),
            title: title.to_string(),
            content: content.to_string(),
            created_at: Local::now().to_rfc3339(),
            mode: normalize_mode(mode),
        };
        {
            let mut list = self.entries.write().await;
            list.insert(0, entry.clone());
        }
        self.persist().await?;
        Ok(entry)
    }

    pub async fn remove(&self, id: &str) -> Result<()> {
        {
            let mut list = self.entries.write().await;
            list.retain(|e| e.id != id);
        }
        self.persist().await?;
        Ok(())
    }

    /// 编辑已有记忆（原标题/内容/投递方式缺失则保持原值）
    pub async fn update(
        &self,
        id: &str,
        title: Option<&str>,
        content: Option<&str>,
        mode: Option<&str>,
    ) -> Result<bool> {
        let mut found = false;
        {
            let mut list = self.entries.write().await;
            if let Some(e) = list.iter_mut().find(|e| e.id == id) {
                if let Some(t) = title {
                    if !t.trim().is_empty() {
                        e.title = t.to_string();
                    }
                }
                if let Some(c) = content {
                    if !c.trim().is_empty() {
                        e.content = c.to_string();
                    }
                }
                if let Some(m) = mode {
                    e.mode = normalize_mode(m);
                }
                found = true;
            }
        }
        if found {
            self.persist().await?;
        }
        Ok(found)
    }

    pub async fn clear(&self) -> Result<()> {
        {
            let mut list = self.entries.write().await;
            list.clear();
        }
        self.persist().await?;
        Ok(())
    }

    /// 按当前投递方式组装注入内容。`limit` 只限制直连系统提示词的条数（最新优先）。
    pub async fn injection(&self, limit: usize) -> MemoryInjection {
        let list = self.entries.read().await;
        let limit = limit.min(200);
        let mut inj = MemoryInjection {
            limit,
            total_count: list.len(),
            ..Default::default()
        };
        let mut system_lines: Vec<String> = Vec::new();
        let mut index_lines: Vec<String> = Vec::new();
        for e in list.iter() {
            // 按需条目，以及超出直连上限的直连条目，都只给索引：
            // 超限时不能把记忆整个丢掉（0 = 全部按需加载也依赖这一分支）。
            if normalize_mode(&e.mode) == MEMORY_MODE_ON_DEMAND || inj.system_count >= limit {
                inj.on_demand_count += 1;
                index_lines.push(format!(
                    "- [{}] {}（id: {}）",
                    e.memory_type, e.title, e.id
                ));
            } else {
                inj.system_count += 1;
                inj.system_chars += e.title.chars().count() + e.content.chars().count();
                system_lines.push(format!("【{}】{}\n{}", e.memory_type, e.title, e.content));
            }
        }
        inj.system_text = system_lines.join("\n\n");
        inj.on_demand_index = index_lines.join("\n");
        inj.index_chars = inj.on_demand_index.chars().count();
        inj
    }

    /// 按需加载的记忆条目（MemoryGuide 的检索范围）
    pub async fn on_demand(&self) -> Vec<MemoryEntry> {
        self.entries
            .read()
            .await
            .iter()
            .filter(|e| normalize_mode(&e.mode) == MEMORY_MODE_ON_DEMAND)
            .cloned()
            .collect()
    }

    /// 读取单条（id 精确/前缀，或标题精确匹配）。找不到返回 None。
    pub async fn find(&self, key: &str) -> Option<MemoryEntry> {
        let key = key.trim();
        if key.is_empty() {
            return None;
        }
        let list = self.entries.read().await;
        list.iter()
            .find(|e| e.id == key || e.title == key)
            .or_else(|| list.iter().find(|e| e.id.starts_with(key)))
            .cloned()
    }

    /// 在全部记忆里做大小写不敏感的子串检索（标题 + 内容）。
    pub async fn search(&self, query: &str) -> Vec<MemoryEntry> {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return Vec::new();
        }
        self.entries
            .read()
            .await
            .iter()
            .filter(|e| {
                e.title.to_lowercase().contains(&q) || e.content.to_lowercase().contains(&q)
            })
            .cloned()
            .collect()
    }

    async fn persist(&self) -> Result<()> {
        if self.path.as_os_str().is_empty() {
            return Ok(());
        }
        if let Some(p) = self.path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let list = self.entries.read().await;
        let s = serde_json::to_string_pretty(&*list)?;
        crate::paths::atomic_write(&self.path, s)?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, mode: &str) -> MemoryEntry {
        MemoryEntry {
            id: id.into(),
            memory_type: "knowledge".into(),
            title: format!("title-{id}"),
            content: format!("content-{id}"),
            created_at: "2026-11-28T00:00:00+08:00".into(),
            mode: mode.into(),
        }
    }

    fn store(entries: Vec<MemoryEntry>) -> MemoryStore {
        let mut s = MemoryStore::default();
        s.entries = Arc::new(RwLock::new(entries));
        s
    }

    #[tokio::test]
    async fn injection_splits_modes_and_respects_limit() {
        let s = store(vec![
            entry("a", MEMORY_MODE_SYSTEM),
            entry("b", MEMORY_MODE_SYSTEM),
            entry("c", MEMORY_MODE_SYSTEM),
            entry("d", MEMORY_MODE_ON_DEMAND),
        ]);
        let inj = s.injection(2).await;
        assert_eq!(inj.system_count, 2);
        assert_eq!(inj.on_demand_count, 2); // 1 条按需 + 1 条超出直连上限
        assert_eq!(inj.total_count, 4);
        assert!(inj.system_text.contains("content-a"));
        assert!(inj.system_text.contains("content-b"));
        // 超限的直连记忆不被丢弃，而是降级为按需索引
        assert!(!inj.system_text.contains("content-c"));
        assert!(inj.on_demand_index.contains("id: c"));
        assert!(inj.on_demand_index.contains("id: d"));
    }

    #[tokio::test]
    async fn zero_limit_indexes_everything() {
        let s = store(vec![
            entry("a", MEMORY_MODE_SYSTEM),
            entry("b", MEMORY_MODE_SYSTEM),
        ]);
        let inj = s.injection(0).await;
        assert_eq!(inj.system_count, 0);
        assert_eq!(inj.on_demand_count, 2);
        assert!(inj.system_text.is_empty());
        assert!(inj.on_demand_index.contains("id: a"));
        assert!(inj.on_demand_index.contains("id: b"));
    }
}
