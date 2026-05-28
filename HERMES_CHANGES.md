# Hermes Agent — emodark Fork 定制改动

本仓库是 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 的个人定制分支。

---

## 核心改动

### 1. 会话存储：SQLite → LMDB + BM25

`hermes_state.py` 完整重写，用 LMDB（内存映射 KV 存储）替代 SQLite 作为会话数据库。

**动机：**
- SQLite WAL 模式在 NFS/SMB 上不兼容，反复出现 `locking protocol` 崩溃
- SQLite 曾被多次损坏（root page 损坏），需要从 JSON 文件重建
- 多进程并发写入时 SQLite 写锁竞争严重

**架构：**
- LMDB 作为主存储：内存映射、零拷贝读、事务原子写
- BM25 倒排索引替代 SQLite FTS5：全文搜索、CJK 分词支持
- 模块级 `_lmdb_env_registry` 单例 + 引用计数：同一进程内多次 `SessionDB()` 不崩
- BM25 懒加载 + OOM 异常安全：索引文件过大时优雅降级（搜索返回空，CRUD 正常）
- 保留 `apply_wal_with_fallback`：`kanban_db.py` 仍使用 SQLite，不受影响
- 默认 DB 路径：`state.db` → `state.lmdb`
- Schema 版本：13 → 14

**新增文件：**
- `tools/migrate_state_to_lmdb.py` — SQLite → LMDB 流式迁移工具
- `hermes_bm25.py` — 纯 Python BM25 倒排索引（零外部依赖）

### 2. 记忆系统大重构

`agent/memory_manager.py` 及相关模块重写，建立分层记忆架构。

- 记忆系统规则化：`MEMORY.md` 定义写入/检索规范
- Prefetch 文件读取替代 hindsight recall 作为主力上下文注入
- 分层策略：宏观层（滚动文件）+ 近期层 + 翻书提示
- 场景标签系统：context 仅保留 scene 标签，控制检索噪声

### 3. 知识路由系统

`agent/calibration.md` + `cli.py` 中的知识路由 Pre-hook 机制。

- 5 层 skill 推荐系统
- 外部知识路由校准 + 本地 0.5B LLM 摘要
- 200 字符预算控制
- 场景/实体标签防假阳性修复

### 4. Hindsight BM25 本地索引集成

在 hindsight 记忆检索层集成 BM25 本地索引，作为向量检索的补充。

- 倒排索引 + BM25 排名（k1=1.2, b=0.75）
- CJK 单字分词
- JSON gzip 持久化

### 5. 飞书/Lark 增强

`gateway/run.py` + `cron/scheduler.py` 中的飞书专属补丁（`HERMES_LARK_*` 标记段）：

| 补丁 | 功能 |
|------|------|
| `HERMES_LARK_NORMALIZE` | 飞书消息标准化预处理 |
| `HERMES_LARK_START` | 消息开始回调，传递 anchor_id |
| `HERMES_LARK_COMPLETE` | 异步等待消息完成 |
| `HERMES_LARK_BACKGROUND_REVIEW` | 后台审阅回调 |
| `HERMES_LARK_INTERRUPT` | 中断消息处理，支持 abort |
| `HERMES_LARK_CRON_DELIVER` | Cron 结果推送到飞书 |

### 6. Web 服务认证增强

`hermes_cli/web_server.py` 中的 session token 认证改进：
- 多路径头部值比较（解决 uvicorn/Starlette 在不同 host/port 下头部传递不一致）
- 兼容 `Authorization: Bearer` 传统路径

### 7. Kanban 增强

- Zombie reaper：检测并回收崩溃 worker
- WAL PRAGMA 跳过：避免冗余 WAL 设置
- Post-commit page_count 不变性检查
- Grace period 容错

### 8. 其他小改动

- Codex 模型适配：`hermes_cli/codex_models.py` 中 codex 传输层配置
- Auth 模块：`hermes_cli/auth.py` 认证流程优化
- 技能中心：`hermes_cli/skills_hub.py` 扩展
- Docker 发布：`.github/workflows/docker-publish.yml`

---

## 分支策略

| 分支 | 说明 |
|------|------|
| `main` | 主线：上游 + 所有定制改动 |
| `main-local` | 仅跟踪上游 main，无定制改动 |

## 维护

```bash
# 拉取上游更新
git checkout main-local
git pull upstream main

# 合并到定制分支
git checkout main
git rebase main-local
```

---

*最后更新: 2026-05-28*
