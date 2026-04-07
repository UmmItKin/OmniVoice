# OmniVoice 本地語音克隆指南 (Arch Linux)

## 概述

OmniVoice 是一個多語言零樣本文字轉語音 (TTS) 模型，支援聲音克隆和聲音設計。本指南說明如何在 Arch Linux 上安裝並使用 AMD GPU 運行。

---

## 硬體需求

- **GPU**: AMD Radeon RX 6000 系列 (Navi 23 或更新)
- **記憶體**: 建議 16GB+
- **儲存空間**: 約 5GB (模型 + 依賴)

---

## 安裝步驟

### 1. 安裝 ROCm (AMD GPU 驅動)

```bash
# 使用 yay 或 paru 安裝
yay -S rocm-hip-sdk rocm-opencl-runtime rocminfo

# 驗證安裝
/opt/rocm/bin/rocminfo
```

應該看到類似輸出：
```
Agent 2
  Name:                    gfx1032
  Marketing Name:          AMD Radeon RX 6650 XT
```

### 2. 建立虛擬環境並安裝 PyTorch

```bash
# 建立專案目錄
cd OmniVoice
mkdir -p .venv

# 使用 uv 建立虛擬環境
uv venv .venv
source .venv/bin/activate

# 安裝 PyTorch ROCm 版本
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install torchaudio --index-url https://download.pytorch.org/whl/rocm7.2

# 驗證 GPU 可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 3. 安裝 OmniVoice

```bash
# 安裝 OmniVoice 及相關依賴
uv pip install -e .
uv pip install soundfile
uv pip install torchcodec  # 注意：ROCm 上可能無法使用
```

---

## 使用方法

### 基本語音克隆

```python
#!/usr/bin/env python3
import soundfile as sf
from omnivoice import OmniVoice
import torch

# 載入模型
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cpu",  # ROCm GPU 可用 "cuda"
    dtype=torch.float32
)

# 產生語音 (提供 ref_text 跳過 Whisper ASR)
audio = model.generate(
    text="要轉換的文字",
    ref_audio="你的錄音.wav",
    ref_text="錄音的內容文字",
    num_step=32,  # 步數越多品質越好
)

# 儲存輸出
sf.write("output.wav", audio[0].squeeze().numpy(), 24000)
```

### 使用 Voice Design (不需要參考音頻)

```python
audio = model.generate(
    text="Hello, this is a test.",
    instruct="male, british accent, middle-aged",
    num_step=32,
)
```

### 可用的 Instruct 選項

| 類別 | 選項 |
|------|------|
| 性別 | male, female |
| 年齡 | child, teenager, young adult, middle-aged, elderly |
| 音調 | very low pitch, low pitch, moderate pitch, high pitch, very high pitch |
| 口音 | american accent, british accent, australian accent, indian accent, chinese accent, japanese accent, korean accent |
| 風格 | whisper |

---

## 常見問題

### 1. ROCm segmentation fault

如果遇到 GPU 崩潰，使用 CPU 模式：
```python
device_map="cpu"
dtype=torch.float32
```

### 2. TorchCodec 錯誤

在 ROCm 上 TorchCodec 不相容。使用 soundfile 代替 torchaudio.save：
```python
import soundfile as sf
sf.write("output.wav", audio[0].squeeze().numpy(), 24000)
```

### 3. 加速下載模型

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### 4. 提升品質

- 增加 `num_step`：32 或 64
- 使用 `guidance_scale=2.0`
- 確保 `ref_text` 與實際錄音內容匹配

---

## 參考連結

- [OmniVoice GitHub](https://github.com/k2-fsa/OmniVoice)
- [OmniVoice HuggingFace](https://huggingface.co/k2-fsa/OmniVoice)
- [PyTorch ROCm 安裝](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.1.5/how-to/3rd-party/pytorch-install.html)

---

##  license

Apache 2.0