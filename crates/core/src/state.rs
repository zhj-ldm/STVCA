//! 共享状态：配置、主 Agent、事件广播、打断标志、语音客户端。

use crate::agents::MainAgent;
use crate::config::Config;
use crate::events::Event;
use crate::mcp::McpManager;
use crate::memory::MemoryStore;
use crate::scheduler::Scheduler;
use crate::sessions::SessionStore;
use crate::skills::SkillManager;
use crate::voice::VoiceClient;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::sync::Mutex;

pub struct CoreState {
    pub config: Arc<Mutex<Config>>,
    pub config_path: std::path::PathBuf,
    pub events: broadcast::Sender<(u64, Event)>,
    /// 事件序号（单调递增；SSE 用它做断线重放与前端去重）
    pub event_seq: Arc<std::sync::atomic::AtomicU64>,
    /// 最近事件环形缓冲：SSE 重连（Last-Event-ID）与 Lagged 补发都从这里取
    pub event_log: Arc<std::sync::Mutex<std::collections::VecDeque<(u64, Event)>>>,
    /// 主 agent 会话（&mut 由 Mutex 保护）
    pub main_agent: Arc<Mutex<Option<MainAgent>>>,
    /// 语音链路客户端
    pub voice: VoiceClient,
    pub skills: SkillManager,
    /// MCP 服务配置与连通性
    pub mcp: McpManager,
    /// 本地 Ollama 模型名缓存（用于 -nothink 后缀自动匹配，重建 Agent 时刷新）
    pub ollama_models_cache: Arc<tokio::sync::Mutex<Vec<String>>>,
    pub memory: MemoryStore,
    pub scheduler: Scheduler,
    /// 多会话存储（对话历史持久化）
    pub sessions: Arc<tokio::sync::Mutex<SessionStore>>,
    /// 主 agent 是否正在工作（只能通过 try_acquire_main 的 CAS 置位）
    pub main_busy: Arc<AtomicBool>,
    /// 主任务轮次代号：每次成功抢占 main_busy 时 +1。
    /// 打断按代号匹配，残留打断信号不会误杀下一个新任务。
    pub main_gen: Arc<AtomicU64>,
    /// 最近一次打断请求针对的轮次代号（0 = 无打断）
    pub interrupt_gen: Arc<AtomicU64>,
    /// 正在 TTS 播报
    pub speaking: Arc<AtomicBool>,
    /// 语音唤醒处理中（防 KWS 重复回调导致同一句话双发）
    pub voice_active: Arc<AtomicBool>,
    /// 是否已初始化 agent（API 配置就绪后）
    pub agents_ready: Arc<AtomicBool>,
    /// 语音对话目标会话（前端开启监听/切换会话时同步；None = 跟随 active）
    pub voice_session: Arc<Mutex<Option<String>>>,
    /// 当前主任务所属会话（权威来源；前端切换会话时据此恢复进行态/按钮）
    pub turn_session: Arc<Mutex<Option<String>>>,
    /// 子 Agent 任务管理（内存态）
    pub subagents: Arc<crate::subagents::SubagentManager>,
    /// 连续对话模式（多轮讨论）：进入后无需再喊唤醒词，直到用户/AI 结束
    pub continuous_active: Arc<AtomicBool>,
}

impl CoreState {
    pub fn new(config: Config, config_path: std::path::PathBuf) -> Self {
        let (tx, _) = broadcast::channel(1024);
        let mcp_path = config_path.with_file_name("mcp.json");
        Self {
            config: Arc::new(Mutex::new(config)),
            config_path,
            events: tx,
            event_seq: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            event_log: Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new())),
            main_agent: Arc::new(Mutex::new(None)),
            voice: VoiceClient::default(),
            skills: SkillManager::default(),
            mcp: McpManager::new(mcp_path),
            ollama_models_cache: Arc::new(Mutex::new(Vec::new())),
            memory: MemoryStore::default(),
            scheduler: Scheduler::default(),
            sessions: Arc::new(Mutex::new(SessionStore::new())),
            main_busy: Arc::new(AtomicBool::new(false)),
            main_gen: Arc::new(AtomicU64::new(0)),
            interrupt_gen: Arc::new(AtomicU64::new(0)),
            speaking: Arc::new(AtomicBool::new(false)),
            voice_active: Arc::new(AtomicBool::new(false)),
            agents_ready: Arc::new(AtomicBool::new(false)),
            voice_session: Arc::new(Mutex::new(None)),
            turn_session: Arc::new(Mutex::new(None)),
            subagents: Arc::new(crate::subagents::SubagentManager::new()),
            continuous_active: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn emit(&self, ev: Event) {
        let id = self.event_seq.fetch_add(1, Ordering::SeqCst) + 1;
        if let Ok(mut log) = self.event_log.lock() {
            log.push_back((id, ev.clone()));
            while log.len() > 512 {
                log.pop_front();
            }
        }
        let _ = self.events.send((id, ev));
    }

    pub fn set_main_status(&self, status: &str) {
        self.emit(Event::AgentStatus {
            agent: "main".into(),
            status: status.into(),
        });
    }

    /// 原子抢占主 Agent：成功返回本轮代号（调用方负责结束时释放 main_busy），
    /// 已有轮次在跑则返回 None。CAS 保证并发请求只有一个能进入。
    pub fn try_acquire_main(&self) -> Option<u64> {
        self.main_busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .ok()?;
        Some(self.main_gen.fetch_add(1, Ordering::SeqCst) + 1)
    }

    /// 请求打断当前轮次：记录当前轮次代号。空闲时不记录，
    /// 即使记录与抢占交错，代号不匹配的新轮次也不受影响。
    pub fn request_interrupt(&self) {
        if self.main_busy.load(Ordering::SeqCst) {
            self.interrupt_gen
                .store(self.main_gen.load(Ordering::SeqCst), Ordering::SeqCst);
        }
    }

    pub fn is_main_interrupted(&self, gen: u64) -> bool {
        gen != 0 && self.interrupt_gen.load(Ordering::SeqCst) == gen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_core() -> CoreState {
        CoreState::new(
            Config::default(),
            std::env::temp_dir().join("st_state_test_config.json"),
        )
    }

    #[test]
    fn acquire_is_exclusive() {
        let core = test_core();
        assert!(core.try_acquire_main().is_some());
        assert!(core.try_acquire_main().is_none());
        core.main_busy.store(false, Ordering::SeqCst);
        assert!(core.try_acquire_main().is_some());
    }

    #[test]
    fn interrupt_hits_only_its_own_generation() {
        let core = test_core();
        let g1 = core.try_acquire_main().unwrap();
        core.request_interrupt();
        assert!(core.is_main_interrupted(g1));
        // 下一轮拿到新代号：旧打断信号不得命中
        core.main_busy.store(false, Ordering::SeqCst);
        let g2 = core.try_acquire_main().unwrap();
        assert_ne!(g1, g2);
        assert!(!core.is_main_interrupted(g2));
    }

    #[test]
    fn idle_interrupt_is_not_recorded() {
        let core = test_core();
        core.request_interrupt(); // 空闲时点停止
        let g = core.try_acquire_main().unwrap();
        assert!(!core.is_main_interrupted(g));
    }
}
