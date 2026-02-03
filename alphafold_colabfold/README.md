# AlphaFold/ColabFold 설치 및 사용 가이드


<img width="1442" height="1370" alt="Screenshot from 2026-02-04 03-58-39" src="https://github.com/user-attachments/assets/d91c565a-70da-42b1-b560-6b0b1c64403f" />


## 개요

**ColabFold**는 AlphaFold2를 로컬에서 실행할 수 있게 해주는 도구입니다. 전체 데이터베이스(~2.5TB) 없이도 단백질 구조 예측이 가능합니다.

| 항목 | ColabFold | AlphaFold 전체 설치 |
|------|-----------|-------------------|
| 저장 용량 | ~30GB | ~2.5TB |
| MSA 생성 | 원격 서버 | 로컬 |
| 설치 난이도 | 쉬움 | 복잡 |
| 예측 품질 | 동일 | 동일 |

---

## 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | 8GB VRAM | 24GB VRAM |
| RAM | 16GB | 32GB+ |
| CUDA | 12.1+ | 12.4 |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

---

## 설치

### 1. Pixi 패키지 관리자 설치

```bash
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc
```

### 2. LocalColabFold 설치

```bash
cd ~/projects/alphafold

# 저장소 클론
git clone https://github.com/YoshitakaMo/localcolabfold.git
cd localcolabfold

# 설치 (pixi 사용)
pixi install && pixi run setup
```

### 3. PATH 설정

```bash
# ~/.bashrc에 추가
cat >> ~/.bashrc << 'EOF'

# ColabFold 활성화 함수
activate-colabfold() {
    export PATH="$HOME/projects/alphafold/localcolabfold/.pixi/envs/default/bin:$PATH"
    cd ~/projects/alphafold
    echo "🧬 ColabFold activated"
}
EOF

source ~/.bashrc
```

### 4. 설치 확인

```bash
activate-colabfold
colabfold_batch --help
```

---

## 디렉토리 구조

```
~/projects/alphafold/
├── localcolabfold/              # ColabFold 설치 디렉토리
│   └── .pixi/envs/default/bin/  # 실행 파일 위치
├── inputs/                      # 입력 FASTA 파일
└── outputs/                     # 예측 결과
    └── insulin_result/          # 예시 결과
        ├── *.pdb                # 3D 구조 파일
        ├── *.json               # 메타데이터
        └── *.png                # 시각화 이미지
```

---

## 사용법

### 기본 사용

```bash
# ColabFold 환경 활성화
activate-colabfold

# 단백질 구조 예측
colabfold_batch <입력_FASTA> <출력_디렉토리>
```

### 예시: 인슐린 구조 예측

```bash
activate-colabfold

# 입력 파일 생성
echo ">test_insulin
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT" > ~/projects/alphafold/inputs/insulin.fasta

# 예측 실행 (5-10분 소요)
colabfold_batch ~/projects/alphafold/inputs/insulin.fasta ~/projects/alphafold/outputs/insulin_result
```

### 예측 결과 예시

```
2026-02-04 03:53:59,729 reranking models by 'plddt' metric
2026-02-04 03:53:59,729 rank_001_alphafold2_ptm_model_3_seed_000 pLDDT=67.8 pTM=0.29
2026-02-04 03:53:59,729 rank_002_alphafold2_ptm_model_1_seed_000 pLDDT=63.6 pTM=0.313
2026-02-04 03:53:59,729 rank_003_alphafold2_ptm_model_4_seed_000 pLDDT=61.8 pTM=0.288
2026-02-04 03:53:59,729 rank_004_alphafold2_ptm_model_2_seed_000 pLDDT=56.6 pTM=0.309
2026-02-04 03:53:59,729 rank_005_alphafold2_ptm_model_5_seed_000 pLDDT=53.8 pTM=0.298
```

---

## 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--num-models` | 사용할 모델 수 (1-5) | 5 |
| `--num-recycle` | 재순환 횟수 (정확도↑, 속도↓) | 3 |
| `--amber` | 구조 이완 (relaxation) 활성화 | False |
| `--use-gpu-relax` | GPU로 구조 이완 실행 | False |
| `--templates` | PDB 템플릿 사용 | False |
| `--msa-mode` | MSA 생성 모드 | mmseqs2_uniref_env |

### 고급 사용 예시

```bash
# GPU 이완 + 템플릿 사용 + 재순환 6회
colabfold_batch input.fasta output/ \
    --amber \
    --use-gpu-relax \
    --templates \
    --num-recycle 6
```

---

## 평가 지표

### pLDDT (predicted Local Distance Difference Test)

각 잔기(residue)의 예측 신뢰도를 나타냅니다.

| 점수 | 신뢰도 | 색상 (시각화) |
|------|--------|--------------|
| 90+ | 매우 높음 | 파란색 |
| 70-90 | 높음 | 하늘색 |
| 50-70 | 보통 | 노란색 |
| <50 | 낮음 | 주황/빨간색 |

### pTM (predicted Template Modeling score)

전체 구조의 정확도를 나타냅니다 (0-1, 높을수록 좋음).

- **> 0.5**: 전체 구조가 올바를 가능성 높음
- **< 0.5**: 구조 예측이 불확실

---

## 출력 파일

| 파일 | 설명 |
|------|------|
| `*_unrelaxed_rank_*.pdb` | 예측된 3D 구조 (이완 전) |
| `*_relaxed_rank_*.pdb` | 이완된 3D 구조 (`--amber` 사용 시) |
| `*_scores_rank_*.json` | 예측 점수 및 메타데이터 |
| `*_pae_rank_*.png` | PAE (Predicted Aligned Error) 플롯 |
| `*_coverage.png` | MSA 커버리지 시각화 |
| `*_plddt.png` | pLDDT 점수 플롯 |

---

## 3D 구조 시각화

### 방법 1: 온라인 뷰어 (권장)

1. https://molstar.org/viewer/ 접속
2. `.pdb` 파일을 브라우저로 드래그 앤 드롭
3. 3D 구조 확인 및 조작

### 방법 2: PyMOL 설치

```bash
# conda로 설치
conda install -c conda-forge pymol-open-source -y

# 실행
pymol ~/projects/alphafold/outputs/insulin_result/*.pdb
```

### 방법 3: ChimeraX

```bash
# 설치
sudo apt install chimerax -y

# 실행
chimerax ~/projects/alphafold/outputs/insulin_result/*.pdb
```

---

## 복합체 (Multimer) 예측

여러 단백질 체인으로 구성된 복합체를 예측할 수 있습니다.

### 입력 형식

```fasta
>protein_A
SEQUENCE_OF_PROTEIN_A
>protein_B
SEQUENCE_OF_PROTEIN_B
```

### 예시

```bash
cat > ~/projects/alphafold/inputs/complex.fasta << 'EOF'
>chain_A
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
>chain_B
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST
EOF

colabfold_batch ~/projects/alphafold/inputs/complex.fasta ~/projects/alphafold/outputs/complex_result
```

---

## 문제 해결

### CUDA/cuDNN 경고 메시지

```
E0000 00:00:... Unable to register cuDNN factory
```

**무시해도 됨** - 기능에 영향 없음.

### MSA 서버 제한

```
WARNING: You are welcome to use the default MSA server...
```

공용 서버 사용량 제한. 대량 예측 시:

```bash
# 로컬 MSA 검색 사용
colabfold_search input.fasta database/ msas/
colabfold_batch msas/ output/
```

### GPU 메모리 부족

긴 단백질 서열의 경우:

```bash
# 모델 수 줄이기
colabfold_batch input.fasta output/ --num-models 1

# 또는 max-seq 제한
colabfold_batch input.fasta output/ --max-msa 512:1024
```

### PATH 충돌

ColabFold 활성화 후 다른 도구(PyMOL 등)가 안 될 때:

```bash
# 새 터미널 열기
exec bash

# 또는 PATH 없이 전체 경로 사용
/usr/bin/pymol file.pdb
```

---

## 참고 자료

- [ColabFold GitHub](https://github.com/sokrypton/ColabFold)
- [LocalColabFold GitHub](https://github.com/YoshitakaMo/localcolabfold)
- [AlphaFold 논문](https://www.nature.com/articles/s41586-021-03819-2)
- [ColabFold 논문](https://www.nature.com/articles/s41592-022-01488-1)
- [Mol* Viewer](https://molstar.org/viewer/)

---

## 라이선스

- AlphaFold: Apache License 2.0
- ColabFold: MIT License

---

*작성일: 2026-02-04*
*환경: Ubuntu 22.04 LTS, RTX 5090 (24GB), CUDA 12.4*
