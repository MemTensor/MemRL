## Self-RAG 核心模块（selfrag_core）

本目录实现了一个**基于主 LLM 的 Self-RAG 风格记忆基线**，目标是：

- 不依赖外部 Self-RAG 模型（如专门训练的 LLaMA 变体）；
- 完全复用当前项目已经在使用的 OpenAI 兼容 LLM 与 Embedding；
- 只在“是否检索 + 选哪些经验 + 如何打标签”这一层引入 Self-RAG 思想；
- 为 BigCodeBench / LifelongBench 提供一个介于 `memory`（MemOS）与 `mem0` 之间的中等复杂度基线。

### 核心概念

- `SelfRAGExperience`：一次任务 + 对应执行轨迹
  - 由上层 benchmark 适配层构造；
  - `task_text` 通常是任务指令 / Prompt；
  - `trajectory` 仅包含该任务的“执行轨迹”（代码 / SQL / 操作步骤 + 评测结果），不包含 memory_context；
  - metadata 中记录 benchmark、split、phase、成功与否等标签。

- `SelfRAGStore`：候选经验的向量检索层
  - 使用传入的 embed_fn（通常是 OpenAI embedding）将 Experience 文本编码为向量；
  - 简单实现为“内存列表 + JSONL 持久化 + 余弦相似度”，索引目录默认为：

    ```text
    <repo_root>/.selfrag/index/<user_id>/index.jsonl
    ```

  - 提供 `add_experience(exp)` 与 `search(query, top_k, filters)` 接口；
  - `search` 返回 `RetrievedCandidate` 列表（id/text/score/metadata）。

- `SelfRAGClient`：LLM 决策层
  - 不直接依赖具体 LLM Provider，而是注入一个 `generate_fn(messages) -> text` 回调；
  - 在 `retrieval_mode="adaptive"` 下：
    - 将当前任务 query + Top-K 候选 Experience 摘要打包成一个 Prompt；
    - 引导主 LLM 输出 JSON：

      ```json
      {
        "should_retrieve": true,
        "docs": [
          {
            "id": "selfrag-0",
            "task_id": "BigCodeBench/42",
            "selected": true,
            "relevance": 0.93,
            "support": 0.88,
            "utility": 5
          },
          ...
        ]
      }
      ```

    - 解析 JSON，得到 `SelfRAGDecision`：
      - `should_retrieve`: 是否需要注入记忆；
      - `docs[*]`: 对每条候选的打分与 selected 标记。
  - 在 `retrieval_mode="always"/"never"` 下，直接用 Top-K 或完全不检索；
  - 解析失败时，会根据配置决定是否回退为简单 Top-K 全选。

- `build_memory_context(candidates, decision)`：
  - 根据 `SelfRAGDecision` 中被选中的候选，构造最终注入给主 LLM 的 memory_context 文本；
  - 文本形式类似：

    ```text
    # Retrieved Experiences from Self-RAG

    ## Example 1 [SUCCESS] (task_id=...)
    [TASK]
    ...

    [TRAJECTORY]
    ...
    ```

### 与 mem0 的关系

- mem0_core：
  - 使用 mem0 提供的 Memory 类与向量存储；
  - 检索纯粹基于相似度 + 阈值，没有“是否检索”的显式决策。

- selfrag_core：
  - 使用本项目已有的 embedding Provider 实现简单向量检索；
  - 在此基础上，引入主 LLM 做 Self-RAG 风格决策：
    - 决定是否检索；
    - 为每条候选经验打标签（相关性/支持度/有用性）并决定 selected；
  - 只负责构造 memory_context，不直接生成代码/SQL，生成仍由主 LLM 在上层完成。

### 与 benchmark 的集成方式（概述）

- BigCodeBench：
  - 训练阶段：
    - 每个任务结束后，将该任务的 Prompt + 生成代码 + 评测结果打包为 SelfRAGExperience 存入 SelfRAGStore；
    - 下一个任务开始前，先用 SelfRAGStore + SelfRAGClient 选出若干“经验示例”，注入 system prompt；
  - 验证阶段：
    - 不再写入新经验，只做检索 + 决策 + 注入。

- LifelongBench（db_bench / os_interaction）：
  - 训练：
    - 每个 session 结束后，将任务指令 + 完整对话轨迹打包为 SelfRAGExperience；
  - 新 session 开始前：
    - 使用 SelfRAG 决策从历史 session 中找出类似任务的经验轨迹，作为“示例回放”注入 agent 的 system prompt。

后续实际实现时，BCB/LLB 的 runner 会在 `--mode selfrag` 下调用 selfrag_core 的这些接口，
并在 `results/*/selfrag/*` 目录下落盘对应的日志与 summary。  

