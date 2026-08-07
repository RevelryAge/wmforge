# -*- coding: utf-8 -*-
"""
cli —— wmforge 命令行工具
==========================

    wmforge embed  input.jpg  trace_id  -o output.jpg
    wmforge extract leak.png [--prealign] [--json]

嵌入：读取输入图 → 写入盲水印 → 输出 JPEG（质量 92）。
提取：读取疑似泄露图 → 尝试还原 trace_id → 打印（支持 --json 结构化输出）。
"""
import argparse
import json
import sys

from . import __version__
from .pipeline import embed, extract


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wmforge",
        description="生产级盲水印管线：DWT+DCT+SVD 向量内核 + RS(64,24) 强纠错 + trace_id",
    )
    p.add_argument("--version", action="version", version=f"wmforge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("embed", help="嵌入盲水印")
    pe.add_argument("input", help="输入图片路径（JPEG/PNG）")
    pe.add_argument("trace_id", help="24 字节追溯码（见 wmforge.trace.generate_trace_id）")
    pe.add_argument("-o", "--output", default=None, help="输出 JPEG 路径（默认 input 同目录 *_wm.jpg）")
    pe.add_argument("--pw", type=int, default=1, help="水印位加密种子（默认 1）")
    pe.add_argument("--pi", type=int, default=1, help="块打乱种子（默认 1）")

    px = sub.add_parser("extract", help="提取盲水印")
    px.add_argument("input", help="疑似泄露图片路径")
    px.add_argument("--pw", type=int, default=1, help="水印位加密种子（默认 1）")
    px.add_argument("--pi", type=int, default=1, help="块打乱种子（默认 1）")
    px.add_argument("--prealign", action="store_true", help="先缩放回嵌入基准尺寸（截图/缩放图必用）")
    px.add_argument("--json", action="store_true", help="JSON 结构化输出")
    return p


def _cmd_embed(args: argparse.Namespace) -> int:
    with open(args.input, "rb") as f:
        data = f.read()
    out = embed(data, args.trace_id, password_wm=args.pw, password_img=args.pi)
    out_path = args.output or (args.input.rsplit(".", 1)[0] + "_wm.jpg")
    with open(out_path, "wb") as f:
        f.write(out)
    print(f"[wmforge] embedded -> {out_path} ({len(out)} bytes)")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    with open(args.input, "rb") as f:
        data = f.read()
    tid = extract(
        data,
        password_wm=args.pw,
        password_img=args.pi,
        prealign=args.prealign,
    )
    if args.json:
        print(json.dumps({"trace_id": tid, "ok": tid is not None}, ensure_ascii=False))
    else:
        print(tid if tid else "[wmforge] no watermark recovered")
    return 0 if tid else 1


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "embed":
        return _cmd_embed(args)
    return _cmd_extract(args)


if __name__ == "__main__":
    sys.exit(main())
