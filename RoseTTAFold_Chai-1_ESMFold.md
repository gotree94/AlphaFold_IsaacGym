---

## 기타 단백질 구조 예측 도구

AlphaFold/ColabFold 외에도 다양한 단백질 구조 예측 도구들이 있습니다.

| 도구 | 개발 | 특징 | GPU 메모리 |
|------|------|------|-----------|
| AlphaFold2 | DeepMind | 최초의 고정밀 예측, 표준 | 16GB+ |
| ColabFold | Steinegger Lab | AlphaFold2 경량화, 빠른 MSA | 8GB+ |
| RoseTTAFold | Baker Lab (UW) | 3-track 구조, 가벼움 | 8GB+ |
| Chai-1 | Chai Discovery | 복합체 예측, Diffusion 기반 | 16GB+ |
| ESMFold | Meta AI | MSA 불필요, 초고속 | 8GB+ |

---

## RoseTTAFold

David Baker 연구팀(University of Washington)이 개발한 단백질 구조 예측 도구입니다.

### 특징

- **3-track 신경망**: 1D 시퀀스, 2D 거리맵, 3D 좌표를 동시에 처리
- **경량화**: AlphaFold2 대비 낮은 GPU 메모리 요구량
- **빠른 예측**: 중간 크기 단백질 기준 수 분 내 완료
- **후속 버전**: RoseTTAFold2, RoseTTAFold-All-Atom (소분자 포함 예측)

### 설치

```bash
# 프로젝트 디렉토리 생성
mkdir -p ~/projects/rosettafold
cd ~/projects/rosettafold

# 저장소 클론
git clone https://github.com/RosettaCommons/RoseTTAFold.git
cd RoseTTAFold

# Conda 환경 생성
conda env create -f RoseTTAFold-linux.yml
conda activate RoseTTAFold

# 의존성 및 가중치 다운로드
./install_dependencies.sh
```

### 데이터베이스 다운로드

```bash
# UniRef30 (필수, ~50GB)
./scripts/download_uniref30.sh

# BFD (선택, ~270GB)
./scripts/download_bfd.sh

# Structure templates (선택, ~100GB)
./scripts/download_pdb100.sh
```

### 사용법

```bash
conda activate RoseTTAFold

# 단일 서열 예측
./run_e2e_ver.sh input.fasta output_dir

# 복합체 예측 (RoseTTAFold2)
./run_complex.sh input.fasta output_dir
```

### 입력 파일 형식

```fasta
>protein_name
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV
```

### 출력 파일

| 파일 | 설명 |
|------|------|
| `*.pdb` | 예측된 3D 구조 |
| `*.npz` | 거리/각도 예측 데이터 |
| `*_lddt.txt` | 잔기별 신뢰도 점수 |

---

## Chai-1

Chai Discovery에서 2024년 하반기에 공개한 최신 단백질 구조 예측 모델입니다.

### 특징

- **복합체 예측**: 단백질-리간드, 단백질-단백질 복합체에 강점
- **Diffusion 기반**: 최신 생성 모델 아키텍처 적용
- **AlphaFold3 수준**: 복합체 예측에서 유사한 성능
- **오픈소스**: 상업적 사용 제한 있으나 연구용 무료

### 설치

```bash
# 프로젝트 디렉토리 생성
mkdir -p ~/projects/chai
cd ~/projects/chai

# 저장소 클론
git clone https://github.com/chaidiscovery/chai-lab.git
cd chai-lab

# Conda 환경 생성
conda create -n chai python=3.11 -y
conda activate chai

# 패키지 설치
pip install -e .

# 또는 PyPI에서 직접 설치
pip install chai-lab
```

### 사용법 (Python API)

```python
import torch
from chai_lab.chai1 import run_inference

# 단백질 서열 정의
fasta_content = """
>protein
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV
""".strip()

# FASTA 파일 생성
fasta_path = "/tmp/example.fasta"
with open(fasta_path, "w") as f:
    f.write(fasta_content)

# 구조 예측 실행
output_dir = "/tmp/chai_output"
candidates = run_inference(
    fasta_file=fasta_path,
    output_dir=output_dir,
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    device=torch.device("cuda:0"),
    use_esm_embeddings=True,
)

# 결과 확인
print(f"Generated {len(candidates.cif_paths)} structures")
for i, cif_path in enumerate(candidates.cif_paths):
    print(f"  Structure {i+1}: {cif_path}")
```

### 사용법 (CLI)

```bash
conda activate chai

# 단일 서열 예측
chai fold --fasta input.fasta --output output_dir

# GPU 지정
CUDA_VISIBLE_DEVICES=0 chai fold --fasta input.fasta --output output_dir
```

### 복합체 예측 예시

```python
# 단백질-리간드 복합체
fasta_content = """
>protein|chain_A
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV
>ligand|chain_B
CC(=O)OC1=CC=CC=C1C(=O)O
""".strip()

# 단백질-단백질 복합체
fasta_content = """
>protein|chain_A
MKTAYIAKQRQISFVKSHFSRQLE
>protein|chain_B
ERLGLIEVQAPILSRVGDGTQDNL
""".strip()
```

### 출력 파일

| 파일 | 설명 |
|------|------|
| `pred.model_idx_*.cif` | 예측된 구조 (mmCIF 형식) |
| `scores.model_idx_*.npz` | 신뢰도 점수 |
| `*.pt` | PyTorch 텐서 (상세 예측 데이터) |

### 신뢰도 점수

```python
import numpy as np

# 점수 파일 로드
scores = np.load("scores.model_idx_0.npz")

# pTM: 전체 구조 신뢰도 (0-1)
print(f"pTM: {scores['ptm']:.3f}")

# pLDDT: 잔기별 신뢰도 (0-100)
print(f"Mean pLDDT: {scores['plddt'].mean():.1f}")

# iPTM: 인터페이스 신뢰도 (복합체용)
if 'iptm' in scores:
    print(f"iPTM: {scores['iptm']:.3f}")
```

---

## ESMFold (보너스)

Meta AI에서 개발한 초고속 단백질 구조 예측 도구입니다.

### 특징

- **MSA 불필요**: 단일 서열만으로 예측 (초고속)
- **ESM-2 기반**: 대규모 언어 모델 활용
- **실시간 예측**: 중간 크기 단백질 수 초 내 완료

### 설치

```bash
conda create -n esmfold python=3.9 -y
conda activate esmfold

pip install fair-esm
pip install "fair-esm[esmfold]"
pip install "dllogger @ git+https://github.com/NVIDIA/dllogger.git"
pip install "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307"
```

### 사용법

```python
import torch
import esm

# 모델 로드
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# 서열 정의
sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV"

# 구조 예측
with torch.no_grad():
    output = model.infer_pdb(sequence)

# PDB 파일 저장
with open("esmfold_output.pdb", "w") as f:
    f.write(output)
```

---

## 도구별 비교

### 정확도 비교 (CASP15 기준)

| 도구 | 단량체 GDT-TS | 복합체 DockQ | 비고 |
|------|--------------|--------------|------|
| AlphaFold2 | 92.4 | - | 단량체 최고 |
| ColabFold | 91.8 | - | AlphaFold2와 동등 |
| RoseTTAFold | 87.2 | 0.45 | 빠른 속도 |
| Chai-1 | 89.5 | 0.62 | 복합체 강점 |
| ESMFold | 84.6 | - | MSA 불필요 |

### 사용 시나리오별 추천

| 시나리오 | 추천 도구 | 이유 |
|---------|----------|------|
| 단일 단백질 고정밀 예측 | ColabFold | 정확도 + 편의성 |
| 대량 서열 스크리닝 | ESMFold | 속도 |
| 단백질-리간드 복합체 | Chai-1 | 복합체 특화 |
| 제한된 GPU 환경 | RoseTTAFold | 낮은 메모리 요구 |
| 연구/논문 | AlphaFold2 | 표준 인용 |

---

## 시각화 도구

### Mol* Viewer (온라인, 권장)

1. https://molstar.org/viewer/ 접속
2. `.pdb` 또는 `.cif` 파일 드래그 앤 드롭
3. 3D 구조 확인 및 조작

### PyMOL (로컬)

```bash
# 설치
conda install -c conda-forge pymol-open-source -y

# 실행
pymol output.pdb
```

### ChimeraX (고급)

```bash
# 설치 (Ubuntu)
sudo apt install chimerax -y

# 실행
chimerax output.cif
```

---

## 참고 자료

### RoseTTAFold
- [GitHub](https://github.com/RosettaCommons/RoseTTAFold)
- [논문 (Science, 2021)](https://www.science.org/doi/10.1126/science.abj8754)
- [RoseTTAFold2 논문](https://www.science.org/doi/10.1126/science.adl2528)

### Chai-1
- [GitHub](https://github.com/chaidiscovery/chai-lab)
- [기술 보고서](https://www.chaidiscovery.com/blog/chai-1)
- [Chai Discovery 웹사이트](https://www.chaidiscovery.com/)

### ESMFold
- [GitHub](https://github.com/facebookresearch/esm)
- [논문 (Science, 2023)](https://www.science.org/doi/10.1126/science.ade2574)
- [ESM Metagenomic Atlas](https://esmatlas.com/)

---

*작성일: 2026-02-07*
*환경: Ubuntu 22.04 LTS, RTX 5090 (24GB), CUDA 12.4*
