from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Mem0Config:
    """
    本仓库内部使用 mem0 的最小配置。

    只包含我们真正需要的字段，不尝试完整覆盖 mem0 自身的全部配置能力。
    """

    # 模式："oss" 使用开源 Memory 类；"platform" 预留给未来的托管/HTTP 集成
    mode: str = "oss"  # "oss" | "platform"

    # 记忆作用域标识：目前以 user_id 为主要分区键
    user_id: str = "mem0_default_user"

    # 可选：托管模式下的 API 鉴权信息
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # 透传给底层 Memory 构造函数的额外参数（例如自定义向量库等）
    extra_init_kwargs: Dict[str, Any] = field(default_factory=dict)

    def normalized_mode(self) -> str:
        """返回归一化后的 mode 值，遇到未知配置时回退为 'oss'。"""
        mode = (self.mode or "oss").strip().lower()
        if mode not in {"oss", "platform"}:
            return "oss"
        return mode

