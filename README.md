# wmforge

生产级盲水印（DWT+DCT+SVD）完整管线：向量化内核 + RS(64,24) 强纠错 + 追溯码生成 + 单图嵌入/提取 CLI。

零框架依赖（numpy / opencv / PyWavelets），可嵌入任意 Python 服务。

## 特性

- **向量化内核**：批量 SVD + einsum 4×4 DCT-II，替代逐块 Python 循环，嵌入/提取快 2-3x
- **与 blind-watermark 0.4.4 数值逐位兼容**（同种子同结果），老图交叉兼容
- **RS(64,24) 强纠错**：可纠正最多 20 字节错误（误码容忍 31%），抗截图/压缩/轻度翻拍
- **紧凑追溯码**：24 字节 `trace_id`（图纸编码+用户索引+分钟级时间戳），用户不存明文
- **完整单图管线**：白底压暗、2400 长边基准降采样嵌入、提取端 prealign、fail-open
- **CLI**：`wmforge embed / extract`

## 安装

```bash
pip install wmforge          # 核心（numpy/opencv-headless/Pillow/PyWavelets）
pip install wmforge[legacy]  # 可选：blind-watermark 0.4.4（仅交叉兼容回归测试用）
```

## 快速开始

```python
from wmforge.pipeline import embed, extract
from wmforge.trace import build_user_index_map, generate_trace_id

# 用户索引映射由调用方注入（本库零框架依赖）
user_map = build_user_index_map(["alice", "bob", "carol"])
tid = generate_trace_id("F13241420", "bob", user_map)
# -> "F13241420_01q_3k9x"

wm_jpg = embed(img_bgr, tid)               # 嵌入 → JPEG bytes（fail-open）
trace_id = extract(leak_bytes, prealign=True)  # 提取 → trace_id | None
```

命令行：

```bash
wmforge embed  input.jpg  F13241420_01q_3k9x  -o output.jpg
wmforge extract leak.png --prealign
wmforge extract leak.png --json
```

## 兼容性

- 嵌入端统一缩放到最长边 2400 基准后再嵌入（大图降采样提速、**小图放大保证两端基准一致**）；提取端必须 `prealign=True`（或输入图与嵌入端尺寸一致），否则误码率上升
- 缩放尺寸统一对齐到 **16 的倍数**（JPEG 4:2:0 色度采样最小单元），embed 与 prealign 共用同一套取整逻辑 → 尺寸一致即 0 误码（roundtrip 硬保证）
- prealign 用 **PIL Image.LANCZOS** 实现（cv2 LANCZOS4 对低能量块图错误更多）
- 白底图纸默认压暗 10 级防 DC 偏移截断；`BLIND_WM_WHITE_HEADROOM` 可调
- 种子参数（`password_wm` / `password_img`）两端必须一致

## 鲁棒性边界（重要）

| 场景 | 结果 |
|---|---|
| roundtrip（尺寸一致） | **0 误码**（硬保证） |
| 单独等比缩放（任意比例，prealign 回基准） | **0 误码** |
| 单独 JPEG 重压缩（q ≥ 50） | **0 误码** |
| 缩放 + JPEG 组合（如 55% + q60） | **临界**：真实图纸 40-64 字节错误，超 RS(64,24) 可纠 20 字节，**可能失败** |

说明：
- 缩放+压缩组合的破坏来自插值混叠 + 量化误差叠加，错误均匀散布无法靠 RS 一次纠完；这是 DWT+DCT+SVD 盲水印的物理极限，**同类方案在该场景同样失败**（已实测对照）
- 平铺纯白/低能量 DCT 块的图最脆弱；适量扫描噪声（≥2%）/复杂内容可显著改善（seed=7 实测 2% 噪声下错误仅 2 byte）
- 实际部署建议：嵌入端保证输入为高分辨率渲染图（≥2400 长边），泄露检测以 roundtrip/单独缩放/单独重压缩为基准场景

## 参数

| 常量 | 值 | 含义 |
|---|---|---|
| `BLIND_WM_LENGTH` | 64 | 码字长度（字节） |
| `BLIND_WM_DATA_LEN` | 24 | 数据段（trace_id 长度） |
| `BLIND_WM_ECC_BYTES` | 40 | RS 校验（纠 20 字节） |
| `BLIND_WM_MAX_EDGE` | 2400 | 嵌入基准最长边 |
| `BLIND_WM_WHITE_HEADROOM` | 10 | 白底压暗幅度 |

## 许可

MIT
