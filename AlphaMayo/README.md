# Alpamayo-R1 Inference 노트북 코드 해석

## 개요

이 문서는 **Alpamayo-R1** 자율주행 AI 모델의 추론(inference) 노트북 코드를 해석합니다.

Alpamayo-R1은 NVIDIA에서 개발한 **VLA (Vision-Language-Action) 모델**로, 카메라 이미지를 분석하여 차량의 주행 경로를 예측하고, 그 이유를 자연어로 설명합니다.

---

## 전체 흐름

```
입력 이미지 → 모델 추론 → 경로 예측 + 이유 설명 → 시각화 → 정확도 평가
```

```
┌─────────────────────────────────────────────────────────────┐
│                    Alpamayo-R1 추론 과정                     │
├─────────────────────────────────────────────────────────────┤
│  1. 입력                                                    │
│     - 카메라 이미지 (전방/측면)                              │
│     - 차량 과거 위치/방향 이력                               │
│                                                             │
│  2. 모델 추론 (VLA - Vision Language Action)                │
│     - 이미지 분석 → 상황 이해                               │
│     - 언어로 이유 생성 (Chain-of-Causation)                 │
│     - 미래 경로 예측                                        │
│                                                             │
│  3. 출력                                                    │
│     - pred_xyz: 예측 경로 좌표                              │
│     - CoC: "공사 콘 피해서 왼쪽으로 이동"                    │
│     - minADE: 0.76m 오차                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 셀별 코드 해석

### 1. 라이브러리 Import

```python
import sys
sys.path.insert(0, '/home/gotree94/projects/alpamayo/repo/src')

import copy
import numpy as np
import mediapy as mp          # 이미지/비디오 시각화
import pandas as pd
import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1  # 메인 모델
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset  # 데이터 로더
from alpamayo_r1 import helper  # 전처리 유틸리티
```

| 라이브러리 | 용도 |
|-----------|------|
| `numpy` | 수치 연산 |
| `mediapy` | 이미지/비디오 시각화 |
| `pandas` | 데이터 처리 |
| `torch` | 딥러닝 프레임워크 |
| `AlpamayoR1` | Alpamayo-R1 모델 클래스 |
| `load_physical_aiavdataset` | NVIDIA Physical AI AV 데이터셋 로더 |
| `helper` | 전처리 유틸리티 함수 |

---

### 2. 모델 로드

```python
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)
```

| 항목 | 설명 |
|------|------|
| `nvidia/Alpamayo-R1-10B` | HuggingFace에서 100억 파라미터 VLA 모델 다운로드 |
| `torch.bfloat16` | 메모리 절약을 위한 16비트 부동소수점 |
| `.to("cuda")` | GPU로 모델 이동 |
| `processor` | 이미지/텍스트 전처리기 |

---

### 3. 데이터 로드 및 전처리

```python
# 클립 ID 목록에서 특정 주행 영상 선택
clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()
clip_id = clip_ids[774]

# 데이터셋에서 해당 클립 로드
data = load_physical_aiavdataset(clip_id)

# 이미지를 모델 입력 메시지 포맷으로 변환
messages = helper.create_message(data["image_frames"].flatten(0, 1))

# 토큰화 및 전처리
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

print("seq length:", inputs.input_ids.shape)

# 모델 입력 구성
model_inputs = {
    "tokenized_data": inputs,           # 토큰화된 입력
    "ego_history_xyz": data["ego_history_xyz"],  # 과거 위치 이력
    "ego_history_rot": data["ego_history_rot"],  # 과거 회전(방향) 이력
}
model_inputs = helper.to_device(model_inputs, "cuda")
```

#### 데이터 구성

| 키 | 설명 |
|----|------|
| `image_frames` | 카메라 이미지들 (전방, 측면 등 다중 카메라) |
| `ego_history_xyz` | 차량의 과거 위치 (x, y, z 좌표) |
| `ego_history_rot` | 차량의 과거 방향 (회전 행렬) |
| `ego_future_xyz` | 실제 미래 경로 (Ground Truth, 평가용) |

---

### 4. 모델 추론 (핵심)

```python
torch.cuda.manual_seed_all(42)  # 재현성을 위한 시드 설정

with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=copy.deepcopy(model_inputs),
        top_p=0.98,              # 샘플링 다양성 (nucleus sampling)
        temperature=0.6,         # 출력 랜덤성 조절 (낮을수록 결정적)
        num_traj_samples=1,      # 생성할 경로 수
        max_generation_length=256,  # 최대 생성 토큰 수
        return_extra=True,       # 추가 정보(CoC) 반환
    )

# Chain-of-Causation 출력
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])
```

#### 추론 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `top_p` | 0.98 | Nucleus sampling - 상위 98% 확률 토큰에서 샘플링 |
| `temperature` | 0.6 | 낮을수록 결정적, 높을수록 다양한 출력 |
| `num_traj_samples` | 1 | 생성할 경로 개수 (GPU 메모리에 따라 조절) |
| `max_generation_length` | 256 | 최대 생성 토큰 수 |

#### 출력 변수

| 변수 | 형태 | 설명 |
|------|------|------|
| `pred_xyz` | `[batch, traj_sets, num_samples, time_steps, 3]` | 예측된 미래 경로 (x, y, z 좌표) |
| `pred_rot` | `[batch, traj_sets, num_samples, time_steps, 3, 3]` | 예측된 미래 방향 (회전 행렬) |
| `extra["cot"]` | `List[List[str]]` | Chain-of-Causation - 주행 이유 설명 |

#### Chain-of-Causation 예시

```
"Nudge to the left to increase clearance from the construction cones encroaching into the lane"
(공사 콘이 차선을 침범하고 있어서 왼쪽으로 살짝 이동)
```

---

### 5. 입력 이미지 시각화

```python
mp.show_images(
    data["image_frames"].flatten(0, 1).permute(0, 2, 3, 1), 
    columns=4, 
    width=200
)
```

차량에 장착된 **다중 카메라 이미지**를 그리드 형태로 표시합니다 (전방, 좌측, 우측 등).

---

### 6. 경로 시각화

```python
import matplotlib.pyplot as plt

def rotate_90cc(xy):
    """좌표를 90도 반시계 방향으로 회전 (시각화용)"""
    return np.stack([-xy[1], xy[0]], axis=0)

# 예측 경로 플롯 (파란색)
for i in range(pred_xyz.shape[2]):
    pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
    pred_xy_rot = rotate_90cc(pred_xy)
    plt.plot(*pred_xy_rot, "o-", label=f"Predicted Trajectory #{i + 1}")

# 실제 경로 플롯 (빨간색)
gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
gt_xy_rot = rotate_90cc(gt_xy)
plt.plot(*gt_xy_rot, "r-", label="Ground Truth Trajectory")

plt.ylabel("y coordinate (meters)")
plt.xlabel("x coordinate (meters)")
plt.legend(loc="best")
plt.axis("equal")
```

#### 그래프 설명

| 요소 | 색상 | 설명 |
|------|------|------|
| 예측 경로 | 🔵 파란색 점선 | 모델이 예측한 미래 주행 경로 |
| 실제 경로 | 🔴 빨간색 실선 | Ground Truth (실제 주행 경로) |

---

### 7. 정확도 평가 (minADE)

```python
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
print("minADE:", diff.min(), "meters")
```

#### minADE (minimum Average Displacement Error)

- **정의**: 예측 경로와 실제 경로 사이의 평균 거리 오차
- **계산**: 각 시간 스텝에서의 유클리드 거리의 평균
- **결과**: `0.75916 meters` → 약 **76cm 오차**

```
minADE = (1/T) × Σ ||pred_xy[t] - gt_xy[t]||₂
```

---

## 핵심 개념: VLA (Vision-Language-Action) 모델

Alpamayo-R1은 **VLA 모델**로, 세 가지 모달리티를 통합합니다:

| 모달리티 | 역할 | 예시 |
|---------|------|------|
| **Vision** | 카메라 이미지 분석 | 전방 카메라, 측면 카메라 영상 |
| **Language** | 상황 이해 및 이유 설명 | "공사 콘을 피해 왼쪽으로 이동" |
| **Action** | 주행 경로 생성 | (x, y, z) 좌표 시퀀스 |

### 기존 자율주행 vs VLA 기반 자율주행

| 구분 | 기존 방식 | VLA 방식 (Alpamayo-R1) |
|------|----------|------------------------|
| 의사결정 | Rule-based / End-to-end | 언어 기반 추론 |
| 설명 가능성 | 낮음 (블랙박스) | 높음 (CoC 제공) |
| 일반화 | 제한적 | 높은 일반화 능력 |

---

## 실행 환경

| 항목 | 요구사항 |
|------|---------|
| Python | 3.12+ |
| GPU | 24GB VRAM 권장 (RTX 4090, RTX 5090 등) |
| CUDA | 12.4+ |
| 주요 라이브러리 | torch, transformers, einops, hydra-core |

---

## 참고 자료

- [Alpamayo GitHub](https://github.com/NVlabs/alpamayo)
- [Alpamayo-R1-10B (HuggingFace)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [PhysicalAI-AV Dataset (HuggingFace)](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

---

*작성일: 2026-01-27*
