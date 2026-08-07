# -*- coding: utf-8 -*-
"""
codes —— 盲水印码字编解码与定长参数
=====================================

包含：
- RS(64, k) over GF(256) 纠错码（纯 Python、零外部依赖）
- 盲水印定长参数常量（embed/extract 两端必须一致）

Reed-Solomon 背景：
DWT-DCT-SVD 盲水印在真实图像上存在"固定弱位"，512 bit 裸提取最坏
~1 bit 错误。RS(64,24) 提供 40 字节校验、可纠正最多 20 字节错误
（误码容忍 31%），抗截图压缩/轻度翻拍，且码字总长仍 64 字节、
不稀释每 bit 投票数。

实现要点：
- GF(2^8)：本原多项式 x^8+x^4+x^3+x^2+1 = 0x11D，生成元 α=2
- 编码：系统式，码字多项式 C(x) = D(x)·x^8 + (D(x)·x^8 mod g(x))
- 解码：BM 算法求错误定位多项式 → Chien search 定位 →
  GF(256) 高斯消元直接解 syndrome 方程求错误幅值
  （避免 Forney 公式在"最高位优先"多项式列表约定下的求导/求值歧义）
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 盲水印定长参数（embed/extract 两端必须一致）
# ---------------------------------------------------------------------------
# 码字长度 BLIND_WM_LENGTH = 64 字节 = 512 bit（wm_shape，每 bit 分到的
# DWT block 数决定错误率——wm_shape 越大每 bit block 越少、错误率越高）。
BLIND_WM_LENGTH = 64
# 数据段 24 字节 + RS 校验 40 字节 = RS(64,24) 纠 20 字节错（误码容忍 31%）。
BLIND_WM_DATA_LEN = 24
BLIND_WM_ECC_BYTES = 40
# 冗余倍数（3 倍冗余会稀释 block 平均，得不偿失，默认 1）。
BLIND_WM_REDUNDANCY = 1
# 白底压暗幅度（255→255-headroom）：为 DCT-SVD 的 DC 偏移留余量防截断。
# 历史值 18 会让 97% 白底的工程图纸全局发灰；10 视觉影响仅 -4%。
BLIND_WM_WHITE_HEADROOM = 10
# 嵌入基准：按最长边等比缩放到该值再嵌入，嵌完放大回原尺寸。
# 提取端必须按同一规则缩放回该基准尺寸再提取（"尺寸一致即可 0 误码"）。
BLIND_WM_MAX_EDGE = 2400

# ---------------------------------------------------------------------------
# GF(2^8) 基础
# ---------------------------------------------------------------------------
_PRIM = 0x11D  # 本原多项式 x^8+x^4+x^3+x^2+1，生成元 α=2
_N = 64         # 码长（字节，固定 512 bit）
_FCR = 0        # 生成多项式第一个根 α^0
_DEFAULT_NSYM = BLIND_WM_ECC_BYTES  # 默认校验符号数（RS(64,24)）


def _build_gf():
    exp = [0] * 512
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= _PRIM
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


_GF_EXP, _GF_LOG = _build_gf()


def _gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_inverse(a):
    return _GF_EXP[255 - _GF_LOG[a]]


def _gf_pow(a, n):
    return _GF_EXP[(_GF_LOG[a] * n) % 255] if a != 0 else 0


def _gf_poly_eval(poly, x):
    y = poly[0]
    for c in poly[1:]:
        y = _gf_mul(y, x) ^ c
    return y


def _gf_poly_scale(poly, x):
    return [_gf_mul(c, x) for c in poly]


def _gf_poly_add(a, b):
    a, b = list(a), list(b)
    if len(a) < len(b):
        a, b = b, a
    b = [0] * (len(a) - len(b)) + b
    return [x ^ y for x, y in zip(a, b)]


def _gf_poly_mul(a, b):
    r = [0] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            r[i + j] ^= _gf_mul(ca, cb)
    return r


def _gf_poly_trim(poly):
    while poly and poly[0] == 0:
        poly.pop(0)
    return poly or [0]


def rs_generator_poly(nsym: int = _DEFAULT_NSYM):
    g = [1]
    for i in range(nsym):
        g = _gf_poly_mul(g, [1, _GF_EXP[i + _FCR]])
    return g


class RSError(Exception):
    """RS 解码失败（错误数超出纠错能力）"""
    pass


def rs_encode(data: bytes, nsym: int = _DEFAULT_NSYM) -> bytes:
    """RS 编码：k 字节数据 -> 64 字节码字（前 k 字节为原始数据）"""
    k = _N - nsym
    if len(data) != k:
        raise ValueError(f"RS data length must be {k}, got {len(data)}")
    gen = rs_generator_poly(nsym)
    msg = list(data) + [0] * nsym
    for i in range(k):
        coef = msg[i]
        if coef != 0:
            for j in range(1, len(gen)):
                msg[i + j] ^= _gf_mul(gen[j], coef)
    return bytes(data) + bytes(msg[k:])


def _syndromes(code: list, nsym: int = _DEFAULT_NSYM) -> list:
    return [_gf_poly_eval(code, _GF_EXP[i + _FCR]) for i in range(nsym)]


def _berlekamp_massey(synd: list, nsym: int = _DEFAULT_NSYM) -> list:
    err_loc = [1]
    old_loc = [1]
    for i in range(len(synd)):
        K = i
        delta = synd[K]
        for j in range(1, len(err_loc)):
            delta ^= _gf_mul(err_loc[-(j + 1)], synd[K - j])
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = _gf_poly_scale(old_loc, delta)
                old_loc = _gf_poly_scale(err_loc, _gf_inverse(delta))
                err_loc = new_loc
            err_loc = _gf_poly_add(err_loc, _gf_poly_scale(old_loc, delta))
    err_loc = _gf_poly_trim(err_loc)
    if (len(err_loc) - 1) * 2 > nsym:
        raise RSError(f"Too many errors ({len(err_loc) - 1}) to correct")
    return err_loc


def _chien_search(err_loc: list, nmess: int) -> list:
    """遍历码字内所有位置 m（0=首字节），其系数为 x^(nmess-1-m)、
    定位根为 α^{-(nmess-1-m)}。根指数可能超出 [0, nmess)，必须按
    255 周期映射，不能只在 range(nmess) 内搜索（历史 bug 点）。"""
    errs = len(err_loc) - 1
    err_pos = []
    for m in range(nmess):
        i = (255 - (nmess - 1 - m)) % 255
        if _gf_poly_eval(err_loc, _GF_EXP[i]) == 0:
            err_pos.append(m)
    if len(err_pos) != errs:
        raise RSError("Could not locate errors")
    return err_pos


def _errata_values(msg: list, synd: list, err_pos: list) -> list:
    """由 syndrome 方程 S_i = Σ_k e_k·X_k^(i+FCR) 直接解错误幅值 e_k。

    用 GF(256) 高斯消元解 errs×errs 线性方程组（注意矩阵行=方程、列=变量，
    与直觉相反易写反——历史 bug 点），避开 Forney 公式在"最高位优先"
    多项式列表约定下求导/求值的坑。err_pos 为码字起始索引，返回一一对应幅值。
    """
    coef_pos = [len(msg) - 1 - p for p in err_pos]
    errs = len(coef_pos)
    if errs == 0:
        return []
    A = []
    for i in range(errs):  # 第 i 行 = 第 i 条 syndrome 方程
        row = []
        for k in range(errs):  # 第 k 列 = 第 k 个错误位置
            row.append(_gf_pow(_GF_EXP[coef_pos[k]], i + _FCR))
        A.append(row)
    b = list(synd[:errs])
    for col in range(errs):
        piv = next((r for r in range(col, errs) if A[r][col] != 0), None)
        if piv is None:
            raise RSError("Singular system in error magnitude solve")
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        inv = _gf_inverse(A[col][col])
        for j in range(col, errs):
            A[col][j] = _gf_mul(A[col][j], inv)
        b[col] = _gf_mul(b[col], inv)
        for r in range(errs):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                for j in range(col, errs):
                    A[r][j] ^= _gf_mul(A[col][j], f)
                b[r] ^= _gf_mul(b[col], f)
    return b


def rs_decode(code: bytes, nsym: int = _DEFAULT_NSYM) -> bytes:
    """RS 解码：64 字节码字 -> k 字节数据；错误超出纠错能力抛 RSError"""
    if len(code) != _N:
        raise ValueError(f"RS code length must be {_N}, got {len(code)}")
    msg = list(code)
    synd = _syndromes(msg, nsym)
    if not any(synd):
        return bytes(msg[: _N - nsym])
    err_loc = _berlekamp_massey(synd, nsym)
    err_pos = _chien_search(err_loc, len(msg))
    err_val = _errata_values(msg, synd, err_pos)
    for pos, val in zip(err_pos, err_val):
        msg[pos] ^= val
    # 修正后复核：仍有非零 syndrome 说明超出纠错能力，拒绝解码
    if any(_syndromes(msg, nsym)):
        raise RSError("Decoding failed after correction")
    return bytes(msg[: _N - nsym])
