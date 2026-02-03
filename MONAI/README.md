# MONAI 설치 및 사용 가이드

## 개요

**MONAI (Medical Open Network for AI)**는 의료 영상 딥러닝을 위한 PyTorch 기반 오픈소스 프레임워크입니다. NVIDIA와 King's College London이 공동 개발했습니다.

| 특징 | 설명 |
|------|------|
| 의료 영상 특화 | CT, MRI, X-ray 등 의료 영상 처리에 최적화 |
| 풍부한 모델 | UNet, ViT, SwinUNETR 등 다양한 네트워크 |
| 전처리 파이프라인 | 의료 영상 전용 Transform 제공 |
| 손실 함수 | Dice Loss, Focal Loss 등 세그멘테이션 특화 |
| 데이터 포맷 | NIfTI, DICOM 등 의료 표준 지원 |

---

## 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | 8GB VRAM | 24GB VRAM |
| RAM | 16GB | 32GB+ |
| Python | 3.8+ | 3.10 |
| PyTorch | 1.9+ | 2.0+ |
| CUDA | 11.0+ | 12.4+ |

---

## 설치

### 1. Conda 환경 생성

```bash
conda create -n monai python=3.10 -y
conda activate monai
```

### 2. PyTorch 설치

```bash
# RTX 5090 등 최신 GPU (sm_120)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# RTX 4090 이하 (일반)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. MONAI 설치

```bash
# 전체 기능 설치 (권장)
pip install 'monai[all]'

# 또는 기본 설치
pip install monai
```

### 4. 설치 확인

```bash
python -c "import monai; monai.config.print_config()"
```

---

## 디렉토리 구조

```
~/projects/monai/
├── data/
│   └── Task09_Spleen/          # 샘플 데이터셋
│       ├── imagesTr/           # 학습용 CT 이미지 (.nii.gz)
│       ├── labelsTr/           # 학습용 라벨 (.nii.gz)
│       └── imagesTs/           # 테스트용 이미지
├── spleen_unet.pth             # 학습된 모델
├── spleen_visualization.png    # 데이터 시각화
└── spleen_prediction.png       # 예측 결과 시각화
```

---

## 빠른 시작

### 1. GPU 동작 확인

```python
import torch
from monai.networks.nets import UNet

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 3D UNet 테스트
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64),
    strides=(2, 2),
).to("cuda")

x = torch.randn(1, 1, 64, 64, 64).to("cuda")
output = model(x)
print(f"Input: {x.shape} → Output: {output.shape}")
print("✅ MONAI 3D UNet test passed!")
```

### 2. 샘플 데이터 다운로드

```python
import os
from monai.apps import download_and_extract

root_dir = os.path.expanduser("~/projects/monai/data")
os.makedirs(root_dir, exist_ok=True)

# Spleen CT 데이터셋 다운로드 (~1.5GB)
resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
download_and_extract(resource, output_dir=root_dir)

print("✅ 데이터 다운로드 완료!")
```

---

## 데이터 시각화

```python
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import LoadImage

# 데이터 경로
data_dir = os.path.expanduser("~/projects/monai/data/Task09_Spleen")
images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

print(f"📊 CT 이미지 수: {len(images)}")
print(f"📊 라벨 수: {len(labels)}")

# 첫 번째 이미지 로드
loader = LoadImage(image_only=True)
image = loader(images[0])
label = loader(labels[0])

print(f"📐 이미지 shape: {image.shape}")
print(f"📐 라벨 shape: {label.shape}")

# 중간 슬라이스 시각화
slice_idx = image.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image[:, :, slice_idx].T, cmap="gray", origin="lower")
axes[0].set_title("CT Image")
axes[0].axis("off")

axes[1].imshow(label[:, :, slice_idx].T, cmap="jet", origin="lower")
axes[1].set_title("Spleen Label")
axes[1].axis("off")

axes[2].imshow(image[:, :, slice_idx].T, cmap="gray", origin="lower")
axes[2].imshow(label[:, :, slice_idx].T, cmap="jet", alpha=0.5, origin="lower")
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("spleen_visualization.png", dpi=150)
plt.show()
```

---

## 세그멘테이션 학습

### 전체 학습 코드

```python
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, CropForegroundd,
    Resized, ToTensord
)
from monai.data import Dataset, DataLoader
import os
import glob

# 데이터 준비
data_dir = os.path.expanduser("~/projects/monai/data/Task09_Spleen")
images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))[:5]
labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))[:5]

data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

# 전처리 파이프라인
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Resized(keys=["image", "label"], spatial_size=(96, 96, 48)),
    ToTensord(keys=["image", "label"]),
])

# 데이터 로더
dataset = Dataset(data=data_dicts, transform=transforms)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# 모델 설정
device = torch.device("cuda")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
).to(device)

loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습
print("🚀 학습 시작...")
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for batch in loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/10, Loss: {epoch_loss/len(loader):.4f}")

print("✅ 학습 완료!")

# 모델 저장
torch.save(model.state_dict(), "spleen_unet.pth")
print("💾 모델 저장: spleen_unet.pth")
```

### 학습 결과 예시

```
🚀 학습 시작...
Epoch 1/10, Loss: 0.6744
Epoch 2/10, Loss: 0.6719
Epoch 3/10, Loss: 0.6669
Epoch 4/10, Loss: 0.6646
Epoch 5/10, Loss: 0.6625
Epoch 6/10, Loss: 0.6598
Epoch 7/10, Loss: 0.6564
Epoch 8/10, Loss: 0.6551
Epoch 9/10, Loss: 0.6517
Epoch 10/10, Loss: 0.6507
✅ 학습 완료!
```

---

## 예측 및 시각화

```python
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensityRange, Resize
import os
import glob

# 모델 로드
device = torch.device("cuda")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
).to(device)
model.load_state_dict(torch.load("spleen_unet.pth"))
model.eval()

# 테스트 이미지 로드
data_dir = os.path.expanduser("~/projects/monai/data/Task09_Spleen")
test_image = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))[0]
test_label = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))[0]

# 전처리
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensityRange(a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    Resize(spatial_size=(96, 96, 48)),
])

image = transforms(test_image).unsqueeze(0).to(device)
label = Compose([LoadImage(image_only=True), EnsureChannelFirst(), Resize(spatial_size=(96, 96, 48))])(test_label)

# 예측
with torch.no_grad():
    output = model(image)
    pred = torch.argmax(output, dim=1).cpu().numpy()[0]

# 시각화
slice_idx = 24

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(image.cpu().numpy()[0, 0, :, :, slice_idx].T, cmap="gray", origin="lower")
axes[0].set_title("CT Image")
axes[0].axis("off")

axes[1].imshow(label.numpy()[0, :, :, slice_idx].T, cmap="jet", origin="lower")
axes[1].set_title("Ground Truth")
axes[1].axis("off")

axes[2].imshow(pred[:, :, slice_idx].T, cmap="jet", origin="lower")
axes[2].set_title("Prediction")
axes[2].axis("off")

axes[3].imshow(image.cpu().numpy()[0, 0, :, :, slice_idx].T, cmap="gray", origin="lower")
axes[3].imshow(pred[:, :, slice_idx].T, cmap="jet", alpha=0.5, origin="lower")
axes[3].set_title("Overlay")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("spleen_prediction.png", dpi=150)
plt.show()

print("✅ 예측 시각화 완료!")
```

---

## 주요 컴포넌트

### 네트워크 (Networks)

| 모델 | 용도 | 차원 |
|------|------|------|
| `UNet` | 기본 세그멘테이션 | 2D/3D |
| `AttentionUnet` | Attention 기반 | 2D/3D |
| `SwinUNETR` | Transformer 기반 | 3D |
| `ViT` | Vision Transformer | 2D/3D |
| `DenseNet` | 분류 | 2D/3D |
| `SEResNet` | 분류 | 2D/3D |

### 손실 함수 (Losses)

| 손실 함수 | 용도 |
|----------|------|
| `DiceLoss` | 세그멘테이션 (클래스 불균형에 강함) |
| `DiceCELoss` | Dice + Cross Entropy |
| `FocalLoss` | 어려운 샘플에 집중 |
| `TverskyLoss` | Dice의 일반화 |
| `DiceFocalLoss` | Dice + Focal |

### Transform (전처리)

| Transform | 설명 |
|-----------|------|
| `LoadImaged` | NIfTI/DICOM 로드 |
| `EnsureChannelFirstd` | 채널 차원 조정 |
| `ScaleIntensityRanged` | 강도 정규화 |
| `CropForegroundd` | 배경 제거 |
| `RandCropByPosNegLabeld` | 랜덤 크롭 |
| `RandAffined` | 랜덤 아핀 변환 |
| `RandFlipd` | 랜덤 뒤집기 |
| `Resized` | 크기 조정 |

---

## 사용 가능한 데이터셋

MONAI에서 바로 다운로드 가능한 데이터셋:

| 데이터셋 | 태스크 | 크기 |
|---------|--------|------|
| Task01_BrainTumour | 뇌종양 세그멘테이션 | ~7GB |
| Task02_Heart | 심장 세그멘테이션 | ~1GB |
| Task03_Liver | 간 세그멘테이션 | ~30GB |
| Task04_Hippocampus | 해마 세그멘테이션 | ~300MB |
| Task05_Prostate | 전립선 세그멘테이션 | ~1GB |
| Task06_Lung | 폐 세그멘테이션 | ~60GB |
| Task07_Pancreas | 췌장 세그멘테이션 | ~12GB |
| Task08_HepaticVessel | 간혈관 세그멘테이션 | ~15GB |
| Task09_Spleen | 비장 세그멘테이션 | ~1.5GB |
| Task10_Colon | 대장 세그멘테이션 | ~5GB |

### 다운로드 예시

```python
from monai.apps import download_and_extract

# 뇌종양 데이터
download_and_extract(
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
    output_dir="./data"
)
```

---

## 문제 해결

### RTX 5090 (sm_120) 지원 오류

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
```

**해결**: PyTorch Nightly 버전 설치

```bash
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### CUDA Out of Memory

```bash
# 배치 크기 줄이기
loader = DataLoader(dataset, batch_size=1, ...)

# 또는 이미지 크기 줄이기
Resized(keys=["image", "label"], spatial_size=(64, 64, 32))
```

### Crop 크기 오류

```
ValueError: ROI size larger than image size
```

**해결**: `RandCropByPosNegLabeld` 대신 `Resized` 사용

```python
# 잘못된 예
RandCropByPosNegLabeld(..., spatial_size=(96, 96, 96))  # 이미지보다 큼

# 올바른 예
Resized(keys=["image", "label"], spatial_size=(96, 96, 48))
```

---

## 참고 자료

- [MONAI 공식 문서](https://docs.monai.io/)
- [MONAI GitHub](https://github.com/Project-MONAI/MONAI)
- [MONAI 튜토리얼](https://github.com/Project-MONAI/tutorials)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

---

## 라이선스

Apache License 2.0

---

*작성일: 2026-02-04*
*환경: Ubuntu 22.04 LTS, RTX 5090 (24GB), CUDA 12.8, PyTorch Nightly*
