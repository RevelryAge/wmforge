# -*- coding: utf-8 -*-
"""
core 向量内核回归测试（数值逐位对齐 blind-watermark 0.4.4）

覆盖三条硬约束（任一破坏则老图追溯失效）：
1. roundtrip: 向量嵌入 → 向量提取 必须 0 误码
2. 交叉兼容: 库嵌入图 → 向量提取 = 库提取（逐位一致且 100% 还原）
3. 向量嵌入图 → 库提取 同样 100% 还原
"""
import os
import tempfile
import unittest

import cv2
import numpy as np

from wmforge.core import BlindWmVector

try:
    from blind_watermark import WaterMark

    HAS_LIB = True
except ImportError:
    HAS_LIB = False


class TestBlindWmVector(unittest.TestCase):
    def setUp(self):
        # 1600×1200: LL 800×600 → 块 30000, 每 bit ~58 块投票(已验证可靠)。
        # 注意不要用更小的图: 块数太少时投票数不足(每 bit <32 块时 wm_avg
        # 出现大量中间值, roundtrip 进入临界区)——生产 2400 长边是 ~124 块/bit。
        rng = np.random.RandomState(42)
        self.img = rng.randint(0, 256, (1600, 1200, 3), dtype=np.uint8)
        self.img[:800, :600] = 255  # 模拟图纸白底
        self.wm_bit = rng.randint(0, 2, 512).astype(np.uint8)

    def _lib_extract(self, png, wm_shape=512):
        wm = WaterMark(password_wm=1, password_img=1)
        tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tf.close()
        try:
            cv2.imwrite(tf.name, png)
            bits = wm.extract(filename=tf.name, wm_shape=wm_shape, mode="bit")
        finally:
            os.remove(tf.name)
        return np.asarray(bits, dtype=np.uint8)

    def _lib_embed(self):
        wm = WaterMark(password_wm=1, password_img=1)
        tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tf.close()
        try:
            cv2.imwrite(tf.name, self.img)
            wm.read_img(tf.name)
            wm.read_wm(self.wm_bit.copy(), mode="bit")
            return wm.embed()
        finally:
            os.remove(tf.name)

    def test_vector_roundtrip(self):
        """向量嵌入 → 向量提取 0 误码"""
        vwm = BlindWmVector(password_wm=1, password_img=1)
        vwm.read_img_arr(self.img)
        vwm.read_wm(self.wm_bit.copy())
        emb = vwm.embed()
        bits = vwm.extract_bits(emb, wm_shape=512)
        self.assertTrue(np.array_equal(bits, self.wm_bit))

    @unittest.skipUnless(HAS_LIB, "blind_watermark 0.4.4 not installed")
    def test_lib_embedded_extractable_by_vector(self):
        """老图兼容: 库嵌入图 → 向量提取 100% 还原"""
        emb = self._lib_embed()
        bits = BlindWmVector(password_wm=1, password_img=1).extract_bits(
            emb, wm_shape=512
        )
        self.assertTrue(np.array_equal(bits, self.wm_bit))

    @unittest.skipUnless(HAS_LIB, "blind_watermark 0.4.4 not installed")
    def test_vector_embedded_extractable_by_lib(self):
        """向量嵌入图 → 库提取 100% 还原"""
        vwm = BlindWmVector(password_wm=1, password_img=1)
        vwm.read_img_arr(self.img)
        vwm.read_wm(self.wm_bit.copy())
        emb = vwm.embed()
        bits = self._lib_extract(emb)
        self.assertTrue(np.array_equal(bits, self.wm_bit))

    @unittest.skipUnless(HAS_LIB, "blind_watermark 0.4.4 not installed")
    def test_two_extractors_bit_identical(self):
        """两版提取器对同一输入逐位一致(数值等价性)"""
        vwm = BlindWmVector(password_wm=1, password_img=1)
        vwm.read_img_arr(self.img)
        vwm.read_wm(self.wm_bit.copy())
        emb = vwm.embed()
        a = self._lib_extract(emb)
        b = vwm.extract_bits(emb, wm_shape=512)
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
