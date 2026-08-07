# -*- coding: utf-8 -*-
"""
trace_id 生成回归测试

- 布局: {drawing_code[:15]}_{用户3位base36}_{分钟级base62 4位} = 24 字节
- 确定性: 同输入同输出（截图时间戳固定）
- 用户超限抛错（防 trace_id 静默变长）
- fallback 确定性（用户不在映射表时 sha256 截断，不中断嵌入）
"""
import unittest
from datetime import datetime

from wmforge.trace import (
    build_user_index_map,
    generate_trace_id,
    minute_base62,
    user_index_b36,
)


class TestTrace(unittest.TestCase):
    FIXED_TS = datetime(2026, 8, 7, 12, 30, 0)

    def test_layout_within_24(self):
        """布局 {code[:15]}_{3位}_{4位}，编码满 15 位时恰好 24 字节，否则更短"""
        m = build_user_index_map(["alice", "bob", "carol"])
        tid = generate_trace_id("F13241420", "bob", m, ts=self.FIXED_TS)
        self.assertLessEqual(len(tid), 24)
        self.assertTrue(tid.startswith("F13241420_"))

    def test_layout_full_15_code_is_24(self):
        """编码满 15 位 → 恰好 24 字节（配合 RS(64,24) 满载荷）"""
        m = build_user_index_map(["alice"])
        tid = generate_trace_id("F" * 15, "alice", m, ts=self.FIXED_TS)
        self.assertEqual(len(tid), 24)

    def test_deterministic(self):
        m = build_user_index_map(["alice", "bob", "carol"])
        a = generate_trace_id("F13241420", "bob", m, ts=self.FIXED_TS)
        b = generate_trace_id("F13241420", "bob", m, ts=self.FIXED_TS)
        self.assertEqual(a, b)

    def test_drawing_code_truncated_to_15(self):
        m = build_user_index_map(["alice"])
        tid = generate_trace_id("F" * 30, "alice", m, ts=self.FIXED_TS)
        self.assertTrue(tid.startswith("F" * 15))

    def test_illegal_chars_filtered(self):
        """非 ASCII 字母数字（中文/符号/空白）全部过滤，只留 [A-Za-z0-9_-.]"""
        m = build_user_index_map(["alice"])
        tid = generate_trace_id("F13241420/特殊字符!@#  ", "alice", m, ts=self.FIXED_TS)
        self.assertTrue(tid.startswith("F13241420_"))
        self.assertEqual(len(tid), 9 + 1 + 3 + 1 + 4)

    def test_user_index_stable_by_name(self):
        m = build_user_index_map(["carol", "alice", "bob"])
        # 按名称排序: alice=000, bob=001, carol=002
        self.assertEqual(user_index_b36("alice", m), "000")
        self.assertEqual(user_index_b36("bob", m), "001")
        self.assertEqual(user_index_b36("carol", m), "002")

    def test_user_index_exceed_max_raises(self):
        with self.assertRaises(ValueError):
            build_user_index_map([f"u{i}" for i in range(36**3 + 1)])

    def test_user_index_fallback_deterministic(self):
        m = build_user_index_map(["alice"])
        a = user_index_b36("not_in_map", m)
        b = user_index_b36("not_in_map", m)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)

    def test_minute_base62_within_28y(self):
        # 基准 2026-01-01；2054 年内 4 位 base62 足够
        tid = generate_trace_id("X", "alice", build_user_index_map(["alice"]), ts=self.FIXED_TS)
        self.assertLessEqual(len(tid), 24)


if __name__ == "__main__":
    unittest.main()
