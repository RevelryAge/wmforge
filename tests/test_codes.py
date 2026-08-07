# -*- coding: utf-8 -*-
"""
RS(64,24) 编解码回归测试

对齐 blind-watermark 0.4.4 的 RS 纠错行为：
- 20 字节错误可纠正（含全部 40 字节校验位同时错的最坏情形）
- 21 字节错误必须拒绝（超纠错能力，防静默坏码）
"""
import unittest

from wmforge.codes import (
    BLIND_WM_DATA_LEN,
    BLIND_WM_ECC_BYTES,
    RSError,
    rs_decode,
    rs_encode,
)


class TestRS(unittest.TestCase):
    def _code(self):
        data = b"F13241420_01q_3k9x_ab12z"  # 恰好 24 字节 trace_id
        return rs_encode(data, nsym=BLIND_WM_ECC_BYTES)

    def test_roundtrip_clean(self):
        code = self._code()
        self.assertEqual(
            rs_decode(code, nsym=BLIND_WM_ECC_BYTES),
            b"F13241420_01q_3k9x_ab12z",
        )

    def test_correct_20_bytes_errors(self):
        """20 字节错误（含最坏情形: 全部校验位错）→ 可纠正"""
        code = bytearray(self._code())
        for i in range(20):
            code[BLIND_WM_DATA_LEN + i] ^= 0xFF
        decoded = rs_decode(bytes(code), nsym=BLIND_WM_ECC_BYTES)
        self.assertEqual(decoded, b"F13241420_01q_3k9x_ab12z")

    def test_reject_21_bytes_errors(self):
        """21 字节错误（超纠错能力）→ 必须拒绝，防静默坏码"""
        code = bytearray(self._code())
        for i in range(21):
            code[i] ^= 0xFF
        with self.assertRaises(RSError):
            rs_decode(bytes(code), nsym=BLIND_WM_ECC_BYTES)

    def test_data_length_must_be_24(self):
        """数据长度必须恰好 24（RS(64,24) 系统码）"""
        with self.assertRaises(ValueError):
            rs_encode(b"short", nsym=BLIND_WM_ECC_BYTES)


if __name__ == "__main__":
    unittest.main()
