# -*- coding: utf-8 -*-
"""
core —— 向量化盲水印内核（DWT+DCT+SVD）
=========================================

与 blind-watermark 0.4.4 的 WaterMarkCore 数值等价（逐块算法 → 批量 numpy），
消除 ~6.4 万次 Python 循环的 DWT+DCT+SVD 开销，嵌入/提取 7s → ~0.3s/页。

兼容性保证（不满足任何一条，老图追溯就会失效）：
1. shuffler 序列：np.random.RandomState(password_img).random((N,16)).argsort(axis=1)
   —— 与库 random_strategy1 完全一致（块内 16 个 DCT 系数的位置排列，
   块顺序固定为行优先，不做块级置换）。
2. DCT：4×4 DCT-II 正交基矩阵 C，Y = C·X·Cᵀ，与 cv2.dct 数值差 ~2.6e-07
   （实测），远小于量化裕量 d1/4 = 9。
3. SVD：np.linalg.svd 批量 (N,4,4) 与逐块 svd 结果完全一致（实测 0.0）。
4. 嵌入/提取公式与库逐位一致（见 embed_blocks / extract_blocks docstring）。
5. DWT/IDWT：pywt.dwt2/idwt2 'haar' 默认 mode='symmetric'（与库一致）。
6. 偶数补边 + YUV 转换 + alpha 处理：与库 read_img_arr 一致。

纯 numpy/cv2/pywt 实现，零框架依赖。
"""
import numpy as np
import cv2
import pywt

# ---------------------------------------------------------------------------
# 4×4 DCT-II 正交基矩阵（与 cv2.dct 的 ortho 归一化定义一致）
# ---------------------------------------------------------------------------
_DCT4 = np.zeros((4, 4), dtype=np.float64)
for _k in range(4):
    for _i in range(4):
        _a = np.sqrt(1 / 4) if _k == 0 else np.sqrt(2 / 4)
        _DCT4[_k, _i] = _a * np.cos(np.pi * (2 * _i + 1) * _k / 8)


def _batch_dct2(blocks: np.ndarray) -> np.ndarray:
    """(N,4,4) 批量 2D DCT-II，返回 float32（等价 cv2.dct 逐块，误差 ~1e-7）"""
    y = np.einsum("ij,njk->nik", _DCT4, blocks)
    y = np.einsum("nij,jk->nik", y, _DCT4.T)
    return y.astype(np.float32)


def _batch_idct2(blocks: np.ndarray) -> np.ndarray:
    """(N,4,4) 批量 2D IDCT-II，返回 float32"""
    y = np.einsum("ij,njk->nik", _DCT4.T, blocks)
    y = np.einsum("nij,jk->nik", y, _DCT4)
    return y.astype(np.float32)


def random_strategy1(seed: int, size: int, block_shape: int = 16) -> np.ndarray:
    """与库 bwm_core.random_strategy1 逐位一致：(size, block_shape) 逐行排列"""
    return (
        np.random.RandomState(seed)
        .random(size=(size, block_shape))
        .argsort(axis=1)
    )


def _blocks_from_ll(ll: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """LL (float32) → (N,4,4) 行优先块（与库 as_strided 的 block_index 顺序一致）"""
    return (
        ll[: gh * 4, : gw * 4]
        .reshape(gh, 4, gw, 4)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 4, 4)
        .copy()
    )


def _ll_from_blocks(blocks: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """(N,4,4) → (gh*4, gw*4) LL（_blocks_from_ll 的逆）"""
    return (
        blocks.reshape(gh, gw, 4, 4)
        .transpose(0, 2, 1, 3)
        .reshape(gh * 4, gw * 4)
    )


def embed_blocks(
    blocks: np.ndarray,
    wm_bit: np.ndarray,
    shuffler: np.ndarray,
    d1: float = 36.0,
    d2: float = 20.0,
) -> np.ndarray:
    """
    向量化 block_add_wm_slow：dct → 16 系数位置打乱 → svd → 双奇异值量化调制
    → 重建 → 逆打乱 → idct。

    与库逐位一致的嵌入公式：
        s[0] = (s[0] // d1 + 1/4 + 1/2·wm) * d1   （wm=0 → 余数 9；wm=1 → 余数 27）
        s[1] = (s[1] // d2 + 1/4 + 1/2·wm) * d2
    逆打乱语义（库 block_dct_flatten[shuffler] = copy）：
        z[shuffler[i]] = y[i]，即 y 的逆置换。
    """
    n = blocks.shape[0]
    wm_size = wm_bit.size
    dct_blocks = _batch_dct2(blocks)
    flat = dct_blocks.reshape(n, 16)
    shuffled = flat[np.arange(n)[:, None], shuffler].reshape(n, 4, 4)

    u, s, vh = np.linalg.svd(shuffled, full_matrices=False)
    bits = wm_bit[np.arange(n) % wm_size].astype(np.float64)
    s[:, 0] = (s[:, 0] // d1 + 0.25 + 0.5 * bits) * d1
    s[:, 1] = (s[:, 1] // d2 + 0.25 + 0.5 * bits) * d2

    recon = np.einsum("bij,bj,bjk->bik", u, s, vh).astype(np.float32)
    flat2 = recon.reshape(n, 16)
    unshuffled = np.empty_like(flat2)
    unshuffled[np.arange(n)[:, None], shuffler] = flat2
    return _batch_idct2(unshuffled.reshape(n, 4, 4))


def extract_blocks(
    blocks: np.ndarray,
    shuffler: np.ndarray,
    d1: float = 36.0,
    d2: float = 20.0,
) -> np.ndarray:
    """
    向量化 block_get_wm_slow：dct → 打乱 → svd → 双奇异值余数判定。

    与库逐位一致的提取公式：
        wm = (s0 % d1 > d1/2) * 3/4 + (s1 % d2 > d2/2) * 1/4
    返回 (N,) 0~1 的每块 bit 置信度。
    """
    n = blocks.shape[0]
    dct_blocks = _batch_dct2(blocks)
    flat = dct_blocks.reshape(n, 16)
    shuffled = flat[np.arange(n)[:, None], shuffler].reshape(n, 4, 4)

    _, s, _ = np.linalg.svd(shuffled, full_matrices=False)
    s0_bit = (s[:, 0] % d1 > d1 / 2).astype(np.float64)
    s1_bit = (s[:, 1] % d2 > d2 / 2).astype(np.float64)
    return (s0_bit * 3 + s1_bit * 1) / 4


def one_dim_kmeans(inputs: np.ndarray) -> np.ndarray:
    """照抄库 one_dim_kmeans：一维二分类（输入 512 个平均投票值 → 0/1）"""
    threshold = 0.0
    center = [float(inputs.min()), float(inputs.max())]
    e_tol = 10 ** (-6)
    for _ in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]
        if abs((center[0] + center[1]) / 2 - threshold) < e_tol:
            threshold = (center[0] + center[1]) / 2
            break
    return inputs > threshold


def _shuffle_wm_bit(wm_bit: np.ndarray, password_wm: int) -> np.ndarray:
    """与库 WaterMark.read_wm 的加密一致：RandomState(password_wm).shuffle 就地打乱"""
    out = wm_bit.copy()
    np.random.RandomState(password_wm).shuffle(out)
    return out


def _decrypt_wm_avg(wm_avg: np.ndarray, password_wm: int) -> np.ndarray:
    """与库 extract_decrypt 一致：z[wm_index[i]] = y[i]（y 的逆置换）"""
    wm_index = np.arange(wm_avg.size)
    np.random.RandomState(password_wm).shuffle(wm_index)
    out = np.empty_like(wm_avg)
    out[wm_index] = wm_avg
    return out


# ---------------------------------------------------------------------------
# 内核类：与库 WaterMarkCore 结构对齐（read_img_arr / embed / extract_raw / extract_avg）
# ---------------------------------------------------------------------------
class BlindWmVectorCore:
    def __init__(self, password_img: int = 1):
        self.block_shape = np.array([4, 4])
        self.password_img = password_img
        self.d1, self.d2 = 36, 20
        self.img = None
        self.img_YUV = None
        self.img_shape = None
        self.ca_shape = None
        self.ca_block_shape = None
        self.part_shape = None
        self.ca = [None, None, None]
        self.hvd = [None, None, None]
        self.wm_bit = None
        self.wm_size = 0
        self.block_num = 0

    def read_img_arr(self, img: np.ndarray):
        """与库 read_img_arr 一致：alpha 剥离 + BGR→YUV + 偶数补边 + dwt2"""
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]
        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        self.img_YUV = cv2.copyMakeBorder(
            cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
            0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]
        self.ca_block_shape = (
            self.ca_shape[0] // 4, self.ca_shape[1] // 4, 4, 4,
        )
        self.part_shape = (self.ca_block_shape[0] * 4, self.ca_block_shape[1] * 4)
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        for c in range(3):
            self.ca[c], self.hvd[c] = pywt.dwt2(self.img_YUV[:, :, c], "haar")

    def read_wm(self, wm_bit: np.ndarray):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size

    def _blocks(self, channel: int) -> np.ndarray:
        gh, gw = self.ca_block_shape[:2]
        return _blocks_from_ll(self.ca[channel], gh, gw)

    def embed(self) -> np.ndarray:
        """向量化 embed：三通道批量嵌入 → idwt2 → YUV2BGR → clip。返回 BGR float32"""
        gh, gw = self.ca_block_shape[:2]
        shuffler = random_strategy1(self.password_img, self.block_num, 16)
        embed_ca = [ca.copy() for ca in self.ca]
        embed_yuv = [None, None, None]

        for c in range(3):
            out = embed_blocks(self._blocks(c), self.wm_bit, shuffler, self.d1, self.d2)
            ca_part = _ll_from_blocks(out, gh, gw)
            embed_ca[c][: self.part_shape[0], : self.part_shape[1]] = ca_part
            embed_yuv[c] = pywt.idwt2((embed_ca[c], self.hvd[c]), "haar")

        embed_img_yuv = np.stack(embed_yuv, axis=2)
        embed_img_yuv = embed_img_yuv[: self.img_shape[0], : self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_yuv, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, 0, 255)

        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        return embed_img

    def extract_raw(self, img: np.ndarray) -> np.ndarray:
        """向量化 extract_raw：返回 (3, block_num) 每通道每块的 bit 置信度"""
        self.read_img_arr(img)
        shuffler = random_strategy1(self.password_img, self.block_num, 16)
        wm_block_bit = np.zeros((3, self.block_num))
        for c in range(3):
            wm_block_bit[c] = extract_blocks(self._blocks(c), shuffler, self.d1, self.d2)
        return wm_block_bit

    def extract_avg(self, wm_block_bit: np.ndarray) -> np.ndarray:
        """与库 extract_avg 等价：wm_avg[i] = wm_block_bit[:, i::wm_size].mean()
        （逐块位置平均 + 三通道平均，一次 bincount 完成）"""
        n = wm_block_bit.shape[1]
        idx = np.arange(n) % self.wm_size
        means = wm_block_bit.mean(axis=0)
        wm_avg = np.bincount(idx, weights=means, minlength=self.wm_size)
        counts = np.bincount(idx, minlength=self.wm_size)
        return wm_avg / np.maximum(counts, 1)


class BlindWmVector:
    """WaterMark 0.4.4 兼容接口（免临时文件，直接收 numpy BGR 数组）：

    wm = BlindWmVector(password_wm=1, password_img=1)
    wm.read_img_arr(img_bgr) / wm.read_wm(wm_bits)
    embed_img = wm.embed()                          # 嵌入 → BGR ndarray
    bits = wm.extract_bits(img_bgr, wm_shape=512)   # 提取 → 解密后 0/1 (512,)
    """

    def __init__(self, password_wm: int = 1, password_img: int = 1):
        self.core = BlindWmVectorCore(password_img=password_img)
        self.password_wm = password_wm

    def read_img_arr(self, img: np.ndarray):
        self.core.read_img_arr(img)

    def read_wm(self, wm_bits: np.ndarray):
        self.wm_bit = _shuffle_wm_bit(np.asarray(wm_bits), self.password_wm)
        self.wm_size = self.wm_bit.size
        self.core.read_wm(self.wm_bit)

    def embed(self) -> np.ndarray:
        return self.core.embed()

    def extract_bits(self, img: np.ndarray, wm_shape) -> np.ndarray:
        """等价库 mode='bit'：extract_raw → extract_avg → kmeans → 解密"""
        self.core.wm_size = np.array(wm_shape).prod()
        wm_block_bit = self.core.extract_raw(img)
        wm_avg = self.core.extract_avg(wm_block_bit)
        bits = one_dim_kmeans(wm_avg)
        return _decrypt_wm_avg(bits, self.password_wm)
