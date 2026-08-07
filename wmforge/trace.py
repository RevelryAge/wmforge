# -*- coding: utf-8 -*-
"""
trace —— 盲水印追溯码（trace_id）生成
=======================================

紧凑 24 字节追溯码，恰好填满 BLIND_WM_DATA_LEN（24），配合 RS(64,24) 强纠错：

    {drawing_code[:15]}_{用户索引3位base36}_{分钟级base62 4字符}
    总长 = 15 + 1 + 3 + 1 + 4 = 24 字节

- 用户不存明文：按 user 名称排序分配 3 位 base36 唯一索引
  （用户数 ≤20000 < 36³=46656，超限抛 ValueError）；
  反查时把索引段映射回用户名。
- 时间戳分钟级 base62（基准 2026-01-01，62⁴≈1478 万分钟≈28 年，覆盖到 2054）。

本模块零框架依赖；用户索引映射由调用方注入（`user_index_map: dict`），
可来自任何用户体系（如 Django / 自建表）。

示例：``"F13241420_1a2_3k9x"``
"""

import hashlib
import string
from datetime import datetime
from typing import Optional

# 3 位 base36 空间 46656 ≥ 20000 用户上限；超限必须抛错防止 trace_id 静默变长
_USER_INDEX_MAX = 36**3
_B36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

_B62_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
# 基准 2026-01-01 00:00（本地时区，与生成侧一致），避免 4 位 base62 从 1970 起算 28 年不够
_TS_EPOCH_MINUTES = int(datetime(2026, 1, 1).timestamp()) // 60


def _b36(i: int, width: int) -> str:
    s = ""
    while i:
        i, r = divmod(i, 36)
        s = _B36_ALPHABET[r] + s
    return s.rjust(width, "0")


def build_user_index_map(
    users: list,
    *,
    max_users: int = _USER_INDEX_MAX,
) -> dict:
    """用户名列表 → {user: 3位base36索引}（按名称排序，保证历史索引稳定）。

    Args:
        users: 全部用户名（含禁用用户，保证删除用户后索引不漂移）。
        max_users: 索引空间上限（默认 36³=46656）。
    Raises:
        ValueError: 用户数超上限（trace_id 会静默变长，必须拒绝）。
    """
    if len(users) > max_users:
        raise ValueError(
            f"用户数 {len(users)} 超过 3 位 base36 索引上限 {max_users}，请扩展 trace_id 布局"
        )
    return {u: _b36(i, 3) for i, u in enumerate(sorted(users))}


def user_index_b36(user: str, user_index_map: dict) -> str:
    """用户名 → 3 位 base36 唯一索引。

    Args:
        user: 登录名。
        user_index_map: {user: index}，由 build_user_index_map 或外部注入。
    Returns:
        3 位 base36 索引；用户不在映射表中时返回确定性 fallback
        （sha256 截断取模），保证嵌入流程不中断、反查按字符串精确匹配不受影响。
    """
    if user in user_index_map:
        return user_index_map[user]
    digest = hashlib.sha256(user.encode("utf-8")).hexdigest()
    fallback = int(digest[:8], 16) % _USER_INDEX_MAX
    return _b36(fallback, 3)


def minute_base62(ts: Optional[datetime] = None) -> str:
    """分钟级时间戳（基准 2026-01-01）转 4 位 base62（0-9a-zA-Z，覆盖到 2054）"""
    minutes = int(((ts or datetime.now()).timestamp()) // 60) - _TS_EPOCH_MINUTES
    if minutes < 0:
        minutes = 0
    s = ""
    while minutes:
        minutes, r = divmod(minutes, 62)
        s = _B62_ALPHABET[r] + s
    return s.rjust(4, "0")[-4:]


def generate_trace_id(
    drawing_code: str,
    user: str,
    user_index_map: dict,
    ts: Optional[datetime] = None,
) -> str:
    """生成盲水印追溯码（紧凑 24 字节，配合 RS(64,24) 强纠错）。

    Args:
        drawing_code: 图纸编码（≤15 字符，超长截断；非法字符过滤）。
        user: 操作者登录名。
        user_index_map: {user: 3位base36索引}——由 build_user_index_map 或外部注入。
        ts: 时间戳（默认当前时间；测试可注入固定值）。
    Returns:
        24 字节 trace_id 字符串，如 ``"F13241420_1a2_3k9x"``。
    """
    _allowed = set(string.ascii_letters + string.digits + "_-.")
    clean_drawing = "".join(c for c in drawing_code if c in _allowed)[:15]
    user_idx = user_index_b36(user, user_index_map)
    ts_str = minute_base62(ts)
    return f"{clean_drawing}_{user_idx}_{ts_str}"
