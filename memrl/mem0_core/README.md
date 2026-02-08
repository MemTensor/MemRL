# memrl.mem0_core 模块说明

本模块是一个**与 MemOS / `memrl.service` 解耦的 mem0 集成层**，用来在本项目中实现基于 mem0 的记忆基线（baseline）。
它不依赖 `MemoryService`，后续所有基于 mem0 的实验都应通过这里暴露的接口完成。

目标场景：
- 在 BigCodeBench / LifelongBench 等 benchmark 中，对比：
  - 无记忆 baseline；
  - MemOS + RL（现有主方案）；
  - mem0 记忆（新 baseline，主要对比检索质量与记忆组织方式）。

本模块解决的问题：
- 提供一个**统一的、benchmark 无关的 mem0 封装**，避免在每个 benchmark 下都直接操作 mem0 SDK；
- 屏蔽 mem0 不同版本（v0.x / v1.x）在接口和返回格式上的差异；
- 提供**结构化日志钩子**，方便在实验中详细记录“写入了哪些经验、检索到了哪些记忆”，支持后续分析实验是否按预期运行。

---

## 组件概览

### 1. `Mem0Config`

文件：`memrl/mem0_core/config.py`

作用：描述本仓库内部使用 mem0 的最小配置。

关键字段：
- `mode: str`  
  - `"oss"`：使用开源 `mem0.Memory` 类（当前主要模式）；  
  - `"platform"`：预留给未来的托管/HTTP 集成，当前仍回退为 `"oss"`。
- `user_id: str`  
  - 逻辑上的记忆作用域（mem0 的 user_id），benchmark 可以用前缀 + domain / task 组合出不同用户。
- `api_key: Optional[str]` / `base_url: Optional[str]`  
  - 可选的鉴权信息；通常依旧建议通过环境变量设置（OPENAI_API_KEY/MEM0_API_KEY），这里只是透传。
- `extra_init_kwargs: Dict[str, Any]`  
  - 直接传给底层 `Memory` 构造函数，用于高级配置（自定义向量库、reranker 等）。

### 2. `Mem0Client`

文件：`memrl/mem0_core/client.py`

作用：对 `mem0` Python SDK 的轻量封装。

主要职责：
- 初始化底层 `Memory` 实例；
- 统一 `add` / `search` 的调用方式和返回格式：
  - 处理 mem0 v0.x 直接返回 `List[dict]` 与 v1.x 返回 `{"results": [...]}` 的差异；
  - 将结果统一转换为 `RetrievedMemory`。
- 输出 DEBUG 级别日志：
  - `add()`：记录 user_id、是否 infer、metadata key 等；
  - `search()`：记录 query 片段、limit、threshold、filters keys、返回条数。

上层一般不会直接使用 `Mem0Client`，而是通过 `Mem0Store` 间接调用。

#### 2.1 向量数据存储路径与并发策略

mem0 OSS 默认使用本地 Qdrant 作为向量存储，且会在 `mem0_dir` 下创建一个迁移用的
`migrations_qdrant`。如果多个进程同时使用同一个本地 Qdrant 目录，会触发：

> Storage folder ... is already accessed by another instance of Qdrant client

为了既保证 **train/val 共享记忆**，又允许 **不同任务/benchmark 并行运行**，`Mem0Client`
在初始化时做了两层路径控制：

1. 为 mem0 内部迁移/遥测单独设置 `MEM0_DIR`（按进程隔离）

   在 `_init_memory` 开头：

   - 若环境中尚未设置 `MEM0_DIR`，则自动设置为：

     ```text
     <repo_root>/.mem0/runtime_<pid>
     ```

     其中：
     - `<repo_root>` 为当前仓库根目录（`memrl/mem0_core` 往上两级）；
     - `<pid>` 为当前进程号。

   - 这样，每个进程自己的 `migrations_qdrant` 路径互不干扰，再同时跑多个进程时不会因为
     `/root/.mem0/migrations_qdrant` 被重复使用而报错。

   这一层只影响 mem0 内部的迁移/遥测 Qdrant，不影响我们手动指定的“主向量库”路径。

2. 为主向量库按 `user_id` 做目录划分（物理隔离，train/val 共享）

   在 mem0 的 `MemoryConfig` 上，我们会根据 `cfg.user_id` 构造一个稳定的 Qdrant 路径：

   ```text
   <repo_root>/.mem0/qdrant/user_<hash(user_id)>
   ```

   - `<repo_root>` 为当前仓库根目录；
   - `<hash(user_id)>` 是对 `user_id` 做 MD5 后取前 8 位，用于避免路径过长/包含特殊字符；
   - 同一个 `user_id`（例如：
     - BigCodeBench：`bcb_mem0_instruct_hard`；
     - LifelongBench db_bench：`llb_mem0_db_bench_seed42`；
     - LifelongBench os_interaction：`llb_mem0_os_interaction_seed42`）
     在任意进程中都会指向同一个向量库目录；
   - 不同 `user_id` 会拥有完全独立的本地 Qdrant 目录，从而在物理上隔离不同实验的记忆。

   对于 **同一个任务**：
   - LLB 里 `run_mem0(split='train')` 与 `run_mem0(split='val')` 使用相同的 `user_id`，
     因此它们共享同一个 `<repo_root>/.mem0/qdrant/user_<hash>`；
   - train 阶段写入的 `Experience` 会保存在这份库里，val 阶段在独立进程中检索时也能命中；
   - 这就是“train+val 共用物理记忆库”的实现方式。

3. 历史数据库（history.db）存放位置

   除了向量库，mem0 还会维护一个 `history.db`，用于记录调用历史等信息。为避免写入 root
   用户的 HOME 目录，本模块将它也固定到项目目录下：

   ```text
   <repo_root>/.mem0/history.db
   ```

   这样做的好处：
   - 所有 mem0 相关数据均位于当前仓库内，更易于打包、迁移、备份；
   - 不污染系统级的 `~/.mem0`；
   - 不同仓库之间的 mem0 状态自然隔离。

4. 与 BigCodeBench / LifelongBench 的关系

   - BigCodeBench mem0 模式（`memrl.bigcodebench_eval.runner.BCBRunner.run_mem0`）和
     LifelongBench mem0 模式（`memrl.lifelongbench_eval.runner.LLBRunner.run_mem0`）都统一通过
     `Mem0Client` / `Mem0Store` 初始化 mem0；
   - 因此它们共享同一套“按 `user_id` 分库”的存储逻辑：
     - BCB：不同 split/subset 使用不同 `user_id`，同一 run 的 train/val 在同一个进程内完成；
     - LLB：同一 task + seed 的 train/val 使用相同 `user_id`，在不同进程中共享一份向量库；
   - 如果想彻底清空本项目的 mem0 向量库和 history，可以安全地删除：

     ```bash
     rm -rf <repo_root>/.mem0
     ```

     下一次运行 mem0 模式时会自动重新创建。

### 3. `Experience` 与 `RetrievedMemory`

文件：`memrl/mem0_core/types.py`

#### `Experience`

统一描述“一次应当写入 mem0 的经验”：

- `benchmark: str`：来源 benchmark 名称，如 `"bigcodebench"`、`"lifelongbench-db"`；
- `task_id: str`：任务标识，如 BigCodeBench 的 `"BigCodeBench/37"` 或 LLB 的 sample_index；
- `phase: str`：`"train"` / `"val"` 等；
- `success: bool`：该次执行是否成功（用于后续分析）；
- `task_text: str`：面向人类/LLM 的任务描述文本；
- `trajectory: str`：对应的执行轨迹：
  - 对 BCB：可以是模型完整输出或“代码 + 评测结果”的组合；
  - 对 LLB：通常为 Session 的对话轨迹或 ground-truth 轨迹。
- `metadata: Dict[str, Any]`：额外信息，如 epoch、domain、reward 等。

Adaptors（如 BigCodeBench / LifelongBench 的 mem0 适配层）负责构造 `Experience`。

#### `RetrievedMemory`

对 mem0 返回的单条记忆的统一视图：

- `id: str`：mem0 内部的记忆 ID；
- `memory: str`：主要记忆内容（mem0 的 `memory` 字段）；
- `score: float`：相似度或相关性分数（mem0 的 `score` 字段）；
- `metadata: Dict[str, Any]`：mem0 侧存储的元数据（我们在写入时注入的 benchmark/task/outcome 等）；
- 以及可选字段 `user_id` / `agent_id` / `run_id`。

---

## 4. `Mem0Store`

文件：`memrl/mem0_core/store.py`

作用：以 `Experience` 为中心的高层 API，是 benchmark 使用 mem0 的主要入口。

构造函数：

```python
store = Mem0Store(
    cfg: Mem0Config,
    log_callback: Optional[Callable[[str, dict], None]] = None,
)
```

- `cfg`：mem0 配置；
- `log_callback`：可选的日志钩子，签名为 `(event_name: str, payload: dict) -> None`。
  - benchmark 可以把它实现为“写入 JSONL 文件”的函数，这样核心模块只负责发事件，不关心落盘细节。

### 4.1 写入：`add_experience`

```python
store.add_experience(exp: Experience, infer: bool = True) -> List[RetrievedMemory]
```

行为：
- 自动将 `Experience` 转换为两条消息：
  - `{"role": "user", "content": task_text}`
  - `{"role": "assistant", "content": trajectory}`
- 合并 metadata：
  - 固定字段：`benchmark` / `task_id` / `phase` / `success`；
  - 额外字段：来自 `exp.metadata`，但不会覆盖上述核心键；
- 调用 `Mem0Client.add(...)` 写入 mem0，返回 `List[RetrievedMemory]`。

日志：
- `logger.debug`：记录 benchmark、task_id、phase、success、metadata keys；
- 若传入 `log_callback`：
  - 触发事件名：`"mem0.add_experience"`；
  - payload 只包含精简信息：
    - `benchmark` / `task_id` / `phase` / `success`；
    - `meta`：额外元数据字段名列表；
  - **不写入完整文本**，避免日志过大。

### 4.2 检索：`search`

```python
store.search(
    query: str,
    *,
    limit: int,
    threshold: Optional[float] = None,
    extra_filters: Optional[Dict[str, object]] = None,
) -> List[RetrievedMemory]
```

行为：
- 将 `extra_filters` 透传给 mem0 的 `filters`（例如限定某个 benchmark、task_id 或其它标签）；
- 调用 `Mem0Client.search(...)`，统一返回 `List[RetrievedMemory]`。

日志：
- 若传入 `log_callback`：
  - 触发事件名：`"mem0.search"`；
  - payload 包含：
    - `query_preview`：query 的前几十个字符；
    - `limit` / `threshold` / `filters_keys`；
    - `returned`：返回条数；
    - `top_samples`：截断的若干样本预览（id / score / preview）。

benchmark 适配层可以直接把这些 payload 写成 JSONL，以便后续分析 mem0 检索是否合理。

---

## 5. `format_memories_for_llm`

文件：`memrl/mem0_core/formatting.py`

作用：将 `RetrievedMemory` 列表格式化成可注入 LLM 的上下文字符串。

接口：

```python
format_memories_for_llm(
    memories: Iterable[RetrievedMemory],
    *,
    budget_tokens: int | None,
    header: str = "# Retrieved Memories from mem0\n",
) -> str
```

特性：
- 若 `budget_tokens` 为正数，则按「4 字符 ≈ 1 token」粗略控制总字数；否则不做截断（只按条数控制）；
- 每条记忆会被渲染为一个小节：

  ```text
  # Retrieved Memories from mem0

  ## Example 1 [SUCCESS] (task_id=BigCodeBench/37)
  <内容截断后文本>

  ## Example 2 [FAILURE]
  ...
  ```

- `outcome` 会优先从 metadata 的 `outcome` 读取，否则根据 `success` / `outcome_success` 判断；
- `task_id` 同样来自 metadata。

这使得 BigCodeBench / LifelongBench 在使用 mem0 时，可以复用与 MemOS 记忆类似的上下文结构，便于对照实验。

---

## 使用示例（伪代码）

### BigCodeBench 侧（示意）

```python
from memrl.mem0_core import Mem0Config, Mem0Store, Experience, format_memories_for_llm

cfg = Mem0Config(
    mode="oss",
    user_id="bcb_mem0_hard",
)

def log_event(name: str, payload: dict) -> None:
    # 例如写入 results/bigcodebench_eval/.../mem0_log.jsonl
    ...

store = Mem0Store(cfg, log_callback=log_event)

# 训练阶段：写入经验
exp = Experience(
    benchmark="bigcodebench",
    task_id="BigCodeBench/37",
    phase="train",
    success=True,
    task_text=task_prompt,
    trajectory=generated_code_with_result,
    metadata={"entry_point": entry_point, "epoch": epoch_index},
)
store.add_experience(exp)

# 验证阶段：检索 + 格式化
mems = store.search(task_prompt, limit=5, threshold=0.2)
mem_ctx = format_memories_for_llm(mems, budget_tokens=2000)
# 将 mem_ctx 拼入 system prompt
```

### LifelongBench 侧（示意）

```python
cfg = Mem0Config(
    mode="oss",
    user_id="llb_db_mem0",
)
store = Mem0Store(cfg, log_callback=log_event)

# 写入经验：基于 entry + Session 生成 Experience
exp = Experience(
    benchmark="lifelongbench-db",
    task_id=str(sample_index),
    phase="train",
    success=session_success(session),
    task_text=entry["instruction"],
    trajectory=session_to_trajectory(session),
    metadata={"split": split_name, "epoch": epoch_index},
)
store.add_experience(exp)

# 检索：用 entry 的 instruction 作为 query
mems = store.search(entry["instruction"], limit=k, threshold=threshold)
mem_ctx = format_memories_for_llm(mems, budget_tokens=summary_tokens)
```

---

后续在 BigCodeBench / LifelongBench 的具体集成中：
- 只需要围绕 `Experience` / `Mem0Store` / `format_memories_for_llm` 做适配；
- 所有 mem0 相关的 SDK 调用、版本差异和基础日志，都统一收敛在 `memrl.mem0_core` 这一层。  

如果你在集成过程中发现 core 层不够用（例如需要额外的过滤字段或日志字段），可以在这里扩展接口，而不触碰 benchmark 具体逻辑。 
