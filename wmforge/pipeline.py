# -*- coding: utf-8 -*-
"""
pipeline —— 单图盲水印完整管线（embed / extract）
====================================================

embed(image, trace_id) -> JPEG bytes
    白底压暗 → 统一缩放回 2400 长边基准（大图降采样 / 小图放大）→
    RS 编码 → 向量内核嵌入 → 放大回原尺寸 → JPEG q92 输出。
    任何异常 fail-open：返回原输入（不阻塞业务主流程）。

extract(png_bytes, ...) -> trace_id | None
    可选 prealign（缩放回 2400 基准，与嵌入端同规则）→ 向量内核提取 →
    packbits → RS 解码 → 还原字符串。失败返回 None。

设计契约：
- 提取端尺寸必须与嵌入端一致（或经 prealign 缩回基准）才能 0 误码，
  见 extract(..., prealign=True)。
- 与 blind-watermark 0.4.4 数值逐位兼容；本管线不依赖旧库，
  旧库仅作可选 extras（legacy）用于交叉兼容回归测试。
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .codes import (
    BLIND_WM_DATA_LEN,
    BLIND_WM_ECC_BYTES,
    BLIND_WM_LENGTH,
    BLIND_WM_MAX_EDGE,
    BLIND_WM_REDUNDANCY,
    BLIND_WM_WHITE_HEADROOM,
    RSError,
    rs_decode,
    rs_encode,
)
from .core import BlindWmVector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _as_bgr_ndarray(image) -> np.ndarray:
    """bytes → BGR uint8 ndarray；已是 ndarray 则校验后原样返回。失败返回 None。"""
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.dtype == np.uint8:
            return image
        return None
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None else None


def _imencode_jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return enc.tobytes() if ok else b""


def _aligned_dim(dim: int, scale: float, block: int = 16) -> int:
    """缩放尺寸并对齐到 block 倍数（JPEG 4:2:0 色度采样最小单元 16×16）。

    嵌入基准与提取 prealign 必须使用同一套取整逻辑，两端尺寸一致才能 0 误码。
    非 16 倍数尺寸（如 1092×2400）在 JPEG 编码时色度下采样网格错位，
    会在全部水印 bit 上散布误差，即使 roundtrip 也超 RS 纠错能力。
    """
    n = round(dim * scale)
    n = max(block, (n + block // 2) // block * block)
    return n


def _normalize_to_base(img: np.ndarray) -> tuple:
    """按最长边等比缩放到 BLIND_WM_MAX_EDGE 基准（统一缩放，非仅降采样）。

    返回 (缩放图, scale, orig_shape)：
    - 长边 > 2400：LANCZOS4 降采样到 2400（大图块数降 ~9-18 倍，嵌入提速）；
    - 长边 < 2400：LANCZOS4 放大到 2400（小图也保证基准一致）；
    - 长边 == 2400：不动（scale=1.0）。

    尺寸经 _aligned_dim 对齐到 16 倍数，与 prealign_image 取整逻辑完全一致，
    保证 embed/extract 两端基准恒一致（"尺寸一致即可 0 误码"）。
    注意：早期版本只在长边 > 2400 时降采样、小图直嵌（scale=1.0），
    而提取端 prealign 总是缩到 2400 —— 小图两端基准错位必然提取失败。
    """
    orig_h, orig_w = img.shape[:2]
    long_edge = max(orig_h, orig_w)
    scale = BLIND_WM_MAX_EDGE / long_edge if long_edge != BLIND_WM_MAX_EDGE else 1.0
    target_w, target_h = _aligned_dim(orig_w, scale), _aligned_dim(orig_h, scale)
    if (target_w, target_h) != (orig_w, orig_h):
        # 统一走 _aligned_dim（对齐 16），覆盖 2400×1092 这类长边已 2400
        # 但短边非 16 倍数的边缘情况；实际缩放比供 _restore_to_orig 还原
        img = cv2.resize(
            img,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )
        scale = target_w / orig_w
    return img, scale, (orig_h, orig_w)


def _restore_to_orig(img: np.ndarray, scale: float, orig_shape: tuple) -> np.ndarray:
    """嵌入完成放大回原尺寸（CUBIC 视觉等效、更快）"""
    if scale != 1.0:
        return cv2.resize(
            img,
            (orig_shape[1], orig_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return img


def _darken_white(img: np.ndarray) -> np.ndarray:
    """白底压暗：>237 的高光像素降 BLIND_WM_WHITE_HEADROOM 级，防 DC 偏移截断丢 bit"""
    bright_mask = img > 237
    if bright_mask.any():
        img = np.clip(
            img.astype(np.int16) - BLIND_WM_WHITE_HEADROOM * bright_mask.astype(np.int16),
            0, 255,
        ).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# 嵌入
# ---------------------------------------------------------------------------


def embed(
    image,
    trace_id: str,
    password_wm: int = 1,
    password_img: int = 1,
) -> bytes:
    """盲水印嵌入：DWT+DCT+SVD 频域写入 trace_id。

    Args:
        image: BGR uint8 ndarray（推荐，跳过编解码），或 JPEG/PNG 字节流。
        trace_id: 追溯码（见 trace.generate_trace_id）。
        password_wm / password_img: 水印位加密/块打乱种子（embed/extract 必须一致）。
    Returns:
        JPEG bytes（fail-open：任何异常返回原输入编码，不抛错）。
    """
    is_ndarray = isinstance(image, np.ndarray)
    try:
        img = _as_bgr_ndarray(image)
        if img is None:
            # 无法解码：返回原输入（bytes 原样，ndarray 转 JPEG）
            return _imencode_jpeg(image) if is_ndarray else image

        # 统一缩放到基准尺寸再嵌入（大图降采样提速，小图放大保证两端基准一致）
        img, scale, orig_shape = _normalize_to_base(img)

        # 白底压暗，防 DCT-SVD DC 偏移在 idct 回像素空间时被 clip 截断
        img = _darken_white(img)

        # 数据 → RS(64,24) 码字 → 512 bit
        wm_data = trace_id.encode("utf-8")[:BLIND_WM_DATA_LEN].ljust(
            BLIND_WM_DATA_LEN, b"\x00"
        )
        wm_codeword = rs_encode(wm_data, nsym=BLIND_WM_ECC_BYTES)
        wm_bits = np.unpackbits(np.frombuffer(wm_codeword, dtype=np.uint8))

        vwm = BlindWmVector(password_wm=password_wm, password_img=password_img)
        vwm.read_img_arr(img)
        vwm.read_wm(np.tile(wm_bits, BLIND_WM_REDUNDANCY))
        embed_img = vwm.embed()

        embed_img = _restore_to_orig(embed_img, scale, orig_shape)

        result = _imencode_jpeg(embed_img, quality=92)
        if result and len(result) > 100:
            return result
        logger.warning("[wmforge] embed produced invalid output, fail-open to original")
    except Exception as e:
        logger.warning(f"[wmforge] embed failed for trace_id={trace_id}: {e}, fail-open")

    # fail-open：返回原输入编码
    if is_ndarray:
        return _imencode_jpeg(image)
    return image


# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------


def prealign_image(png_bytes: bytes, target_edge: int = BLIND_WM_MAX_EDGE) -> bytes:
    """提取前预对齐：把图片按最长边等比缩放到嵌入基准尺寸。

    嵌入端统一缩放到最长边 2400 后嵌入、再放大回原尺寸；
    提取端必须按同一规则缩放回该基准尺寸再提取（"尺寸一致即可 0 误码"）。

    使用 PIL Image.LANCZOS 实现——与生产 trace_extractor._prealign 完全一致。
    cv2 INTER_LANCZOS4 与 PIL LANCZOS 的 kernel 实现不同，实测对部分图纸
    （含大面积平坦/低能量块）cv2 链路错误更多，PIL 更稳。
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as pil_img:
            w, h = pil_img.size
            long_edge = max(w, h)
            if long_edge == target_edge:
                return png_bytes
            scale = target_edge / long_edge
            nw, nh = _aligned_dim(w, scale), _aligned_dim(h, scale)
            if (nw, nh) == (w, h):
                return png_bytes
            resized = pil_img.resize((nw, nh), Image.LANCZOS)
            buf = io.BytesIO()
            resized.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return png_bytes


# 兼容别名（函数原名与 extract() 的参数名冲突，外部调用 prefer prealign_image）
prealign = prealign_image


def extract(
    image_bytes: bytes,
    length: int = BLIND_WM_LENGTH,
    password_wm: int = 1,
    password_img: int = 1,
    prealign: bool = False,
) -> Optional[str]:
    """从疑似泄露图片中提取盲水印 trace_id。

    Args:
        image_bytes: 疑似泄露的图片字节流。
        length: 提取字节长度（必须与 embed 端 BLIND_WM_LENGTH 一致，默认 64）。
        password_wm / password_img: 与嵌入端一致的种子。
        prealign: 是否先缩放回嵌入基准尺寸（截图与原图尺寸不同时必须 True）。
    Returns:
        trace_id 字符串；提取失败返回 None（调用方展示友好提示）。
    """
    try:
        if prealign:
            image_bytes = prealign_image(image_bytes)
        img = _as_bgr_ndarray(image_bytes)
        if img is None:
            return None

        vwm = BlindWmVector(password_wm=password_wm, password_img=password_img)
        bits = vwm.extract_bits(img, wm_shape=length * 8 * BLIND_WM_REDUNDANCY)
        if bits is None:
            return None

        wm_bytes = np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()
        try:
            wm_data = rs_decode(wm_bytes, nsym=BLIND_WM_ECC_BYTES)
        except RSError:
            logger.warning("[wmforge] RS decode failed (too many bit errors), returning None")
            return None
        cleaned = wm_data.rstrip(b"\x00")
        if cleaned:
            return cleaned.decode("utf-8", errors="replace")
        return None
    except Exception as e:
        logger.warning(f"[wmforge] extract failed: {e}")
        return None
