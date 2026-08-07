# -*- coding: utf-8 -*-
"""
wmforge —— 生产级盲水印管线
============================

DWT+DCT+SVD 频域盲水印的完整工程化实现（与 blind-watermark 0.4.4
的 WaterMarkCore 数值逐位兼容，但为批量 numpy 向量化内核，快 2-3x）：

- core.BlindWmVector   向量化嵌入/提取内核（批量 SVD + einsum DCT）
- codes.rs_encode/decode  RS(64,24) 强纠错（可纠 20 字节，误码容忍 31%）
- trace.generate_trace_id 紧凑追溯码生成（24 字节：图纸编码+用户索引+时间戳）
- pipeline.embed/extract  单图完整管线（降采样基准嵌入 + prealign 提取）
- cli                    命令行工具

零框架依赖，可嵌入任意 Python 服务。
"""
from . import codes, core, pipeline, trace

__version__ = "0.1.0"

__all__ = ["core", "codes", "pipeline", "trace", "cli", "__version__"]
