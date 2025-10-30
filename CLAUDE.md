# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DCVC_accelerate 是 DCVC-RT (Deep Contextual Video Compression - Real Time) 的官方实现,这是一个实时神经视频编解码器,发表于 CVPR 2025。该项目在单个仓库中包含了整个 DCVC 系列模型,旨在实现高压缩率、低延迟和广泛的多功能性。

**核心特性:**
- 100+ FPS 1080p 编解码速度
- 支持 4K 实时编码
- 单模型支持宽比特率范围
- 码率控制支持
- 统一的 YUV 和 RGB 编码

## 架构概览

### 主要组件

1. **模型架构** (`src/models/`)
   - `video_model.py`: DMC (P帧) 视频压缩模型,使用隐式时序建模
   - `image_model.py`: DMCI (I帧) 图像压缩模型
   - `common_model.py`: 基础压缩模型类,包含熵编码/解码逻辑
   - `entropy_models.py`: 熵模型实现

2. **核心层** (`src/layers/`)
   - `layers.py`: 基础网络层 (DepthConvBlock, ResidualBlock 等)
   - `cuda_inference.py`: CUDA 优化的推理实现
   - `extensions/inference/`: CUDA 内核扩展 (C++/CUDA)

3. **工具模块** (`src/utils/`)
   - `video_reader.py`: 视频读取器 (YUV420, PNG)
   - `stream_helper.py`: 比特流读写辅助函数
   - `metrics.py`: PSNR, MS-SSIM 计算
   - `transforms.py`: 色彩空间转换 (RGB↔YCbCr, YUV420↔YUV444)

4. **比特流处理** (`src/cpp/`)
   - C++ 扩展用于算术编码和比特流操作
   - 使用 pybind11 绑定到 Python

### 双模型系统

系统使用两个独立的模型:
- **I帧模型** (DMCI): 用于编码关键帧,可独立解码
- **P帧模型** (DMC): 用于编码预测帧,依赖参考帧

关键创新:
- **隐式时序建模**: 消除复杂的显式运动模块
- **单低分辨率潜变量**: 避免渐进下采样,显著加速
- **特征适配器重置**: 通过 `reset_interval` 参数周期性重置以防止误差累积

### 量化参数 (QP) 系统

- I帧和P帧使用独立的QP
- P帧的QP会根据帧类型自动偏移 (`qp_shift = [0, 8, 4]`)
- 支持 64 个量化级别 (0-63)
- `force_zero_thres`: 控制量化阈值,影响压缩率和质量

## 开发命令

### 环境设置

```bash
# 创建 conda 环境
conda create -n dcvc python=3.12
conda activate dcvc

# 安装 PyTorch (CUDA 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装依赖
pip install -r requirements.txt
```

### 构建项目

**必须按顺序构建两个扩展:**

```bash
# 1. 构建 C++ 比特流扩展
cd ./src/cpp/
pip install .

# 2. 构建 CUDA 推理扩展
cd ../layers/extensions/inference/
pip install .
```

**验证 CUDA 扩展加载:**
- 如果 CUDA 内核未成功加载,会输出: `cannot import cuda implementation for inference, fallback to pytorch.`
- 这会导致性能下降,但不会影响功能

### 测试模型

**基本测试命令:**
```bash
python test_video.py \
    --model_path_i ./checkpoints/cvpr2025_image.pth.tar \
    --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
    --rate_num 4 \
    --test_config ./dataset_config_example_yuv420.json \
    --cuda 1 -w 1 \
    --write_stream 1 \
    --force_zero_thres 0.12 \
    --output_path output.json \
    --force_intra_period -1 \
    --reset_interval 64 \
    --verbose 1
```

**关键参数说明:**
- `--rate_num`: 测试的码率点数量 (2-64),默认4
- `--qp_i/--qp_p`: 自定义I帧/P帧的QP值列表
- `--force_intra_period`:
  - `-1`: 仅第一帧为I帧 (推荐用于测试)
  - `32/96`: 固定周期的I帧
  - `1`: 强制全I帧编码
- `--reset_interval`: 特征适配器重置间隔,默认32帧
- `--write_stream 1`: 写入比特流到 `out_bin/`
- `--check_existing 1`: 跳过已存在的测试结果
- `--verbose`:
  - `0`: 不测量速度
  - `1`: 序列级速度测量
  - `2`: 帧级速度测量
- `-w`: 并行worker数量,建议设置为GPU数量

**测试脚本:**
```bash
bash run_test.sh  # 使用预配置参数运行测试
```

### CPU 性能调优

算术编码在 CPU 上运行,需要确保 CPU 运行在高性能模式:

```bash
# 检查 CPU 频率
grep -E '^model name|^cpu MHz' /proc/cpuinfo

# 设置最大频率
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 恢复默认频率
echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## 数据集配置

### YUV420 格式 (推荐)

数据集配置示例: `dataset_config_example_yuv420.json`

目录结构:
```
/media/data/
├── UVG/
│   └── Beauty_1920x1080_120fps_420_8bit_YUV.yuv
├── MCL-JCV/
│   └── videoSRC01_1920x1080_30.yuv
└── HEVC_B/
    └── BasketballDrive_1920x1080_50.yuv
```

配置格式:
```json
{
  "root_path": "/media/data/",
  "test_classes": {
    "UVG": {
      "test": 1,
      "base_path": "UVG",
      "src_type": "yuv420",
      "sequences": {
        "Beauty_1920x1080_120fps_420_8bit_YUV.yuv": {
          "width": 1920,
          "height": 1080,
          "frames": 600,
          "intra_period": -1
        }
      }
    }
  }
}
```

### RGB 格式

参考 `DCVC-family/DCVC-FM/` 获取 RGB 测试指南

## 测试条件指南

参考 `test_conditions.md` 了解标准化测试条件:

**编码设置建议:**
- **不要裁剪**源序列,使用填充 (padding)
- 编码帧数: 至少 96 帧
- Intra period: 32, 96, 或 -1 (仅首帧为I帧)
- 避免不合理的小 intra period (如 10, 12)

**传统编解码器对比:**
- 优先比较 YUV420 内容
- 使用 B 帧而非 P 帧
- 使用分层 QP 设置
- 10-bit 内部位深度
- 推荐使用 HM-16.25, VTM-17.0, ECM-5.0

**质量指标:**
- PSNR 和 MS-SSIM
- YUV420: `PSNR_avg = (6*PSNR_y + PSNR_u + PSNR_v) / 8`

## DCVC 系列模型

仓库包含完整的 DCVC 系列实现,位于 `DCVC-family/`:
- **DCVC** (NeurIPS 2021): 基础模型
- **DCVC-TCM** (IEEE TMM): 时序上下文挖掘
- **DCVC-HEM** (ACM MM 2022): 混合空时熵建模
- **DCVC-DC** (CVPR 2023): 多样化上下文
- **DCVC-FM** (CVPR 2024): 特征调制
- **DCVC-RT** (CVPR 2025): 实时压缩 (主项目)
- **EVC** (ICLR 2023): 可扩展编码器

每个子项目都有独立的 README 和预训练模型。

## 代码规范

1. **颜色空间处理:**
   - 主项目优化用于 YUV420 格式
   - 内部使用 YCbCr 444 表示
   - 转换使用 BT.709 色彩空间

2. **模型推理:**
   - 使用 `torch.float16` 进行推理
   - CUDA 扩展自动融合操作以提升性能
   - 支持自动回退到 PyTorch 实现

3. **多进程处理:**
   - 使用 `spawn` 启动方式
   - 每个进程初始化独立的模型副本
   - GPU 通过 `CUDA_VISIBLE_DEVICES` 分配

4. **比特流格式:**
   - 使用 SPS (Sequence Parameter Set) 头部
   - NAL 类型: NAL_SPS, NAL_I, NAL_P
   - 支持两个熵编码器用于大分辨率 (>720p)

## 重要注意事项

1. **模型检查点:**
   - 下载预训练模型到 `./checkpoints/` 目录
   - 需要两个模型: `cvpr2025_image.pth.tar` 和 `cvpr2025_video.pth.tar`

2. **分辨率支持:**
   - 支持任意原始分辨率
   - 自动填充至 16 的倍数
   - 重建视频会裁剪回原始尺寸
   - 失真计算在原始分辨率进行

3. **性能测量:**
   - `test_time` 包含 I/O, 编解码, 失真计算
   - 实际编解码时间单独测量 (`avg_frame_encoding_time`, `avg_frame_decoding_time`)
   - 跳过前 10 帧作为预热
   - 确保 `time.time()` 在测试平台有足够精度

4. **输出文件:**
   - 比特流: `out_bin/{dataset}/{sequence}_q{qp}.bin`
   - 指标: `out_bin/{dataset}/{sequence}_q{qp}.json`
   - 结果汇总: 通过 `--output_path` 指定

## 许可证

Microsoft 开源项目,遵循 MIT 许可证。

## 引用

如果使用本代码,请引用 DCVC-RT 论文:
```bibtex
@inproceedings{jia2025towards,
  title={Towards Practical Real-Time Neural Video Compression},
  author={Jia, Zhaoyang and Li, Bin and Li, Jiahao and Xie, Wenxuan and Qi, Linfeng and Li, Houqiang and Lu, Yan},
  booktitle={CVPR},
  year={2025}
}
```
