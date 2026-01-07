# AlpaFold 및 AlpaMayo 설치를 위한 Ubuntu 22.04 LTS: 4TB + 2TB 디스크 분리 사용 가이드

## 개요

| 항목 | 내용 |
|------|------|
| OS | Ubuntu 22.04 LTS |
| 디스크 구성 | 4TB (정적 데이터) + 2TB (활성 작업) |
| 방식 | 독립 마운트 + 심볼릭 링크 통합 |
| 용도 | AlphaFold, Alpamayo 개발 환경 |

---

## 왜 분리 사용인가?

| 장점 | 설명 |
|------|------|
| **데이터 안전성** | 한 디스크 고장 시 다른 디스크 데이터 보존 |
| **복구 용이성** | 단순 마운트로 데이터 접근 가능 |
| **유지보수** | 디스크 교체/업그레이드 시 독립적 작업 |
| **성능 분리** | I/O 부하 분산 가능 |

---

## Part 1: 디스크 역할 분담

```
┌─────────────────────────────────────────────────────────────┐
│ 시스템 디스크 (NVMe/SSD, 512GB~2TB)                         │
│ ├── /boot/efi   (512MB)                                    │
│ ├── /boot       (1GB)                                      │
│ └── /           (나머지) - OS, 프로그램, 홈 디렉토리        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 데이터 디스크 1: 4TB → /mnt/storage (정적/대용량 데이터)    │
│ ├── alphafold-db/      (~2.5TB) 유전자 데이터베이스         │
│ ├── alpamayo-dataset/  (~1TB) Physical AI 주행 데이터       │
│ └── archives/          백업, 아카이브                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 데이터 디스크 2: 2TB → /mnt/workspace (활성 작업 공간)      │
│ ├── projects/          현재 진행 중인 프로젝트              │
│ │   ├── alphafold/     AlphaFold 실행 환경                 │
│ │   └── alpamayo/      Alpamayo 실행 환경                  │
│ ├── outputs/           실험 결과물                          │
│ ├── models/            모델 가중치 (자주 접근)              │
│ └── cache/             pip, HuggingFace, Docker 캐시       │
└─────────────────────────────────────────────────────────────┘
```

### 용량 계획

| 디스크 | 용도 | 예상 사용량 | 여유 |
|--------|------|------------|------|
| 4TB | AlphaFold DB | ~2.5TB | |
| | Alpamayo Dataset | ~1TB | |
| | 아카이브/백업 | ~500GB | **~0TB** |
| 2TB | 모델 가중치 | ~50GB | |
| | 실험 결과물 | ~200GB | |
| | 캐시 | ~100GB | |
| | 작업 공간 | ~500GB | **~1.2TB** |

---

## Part 2: Ubuntu 설치

### 2.1 부팅 USB 생성

```bash
# 다른 Linux/Mac에서
sudo dd if=ubuntu-22.04.x-desktop-amd64.iso of=/dev/sdX bs=4M status=progress

# 또는 Rufus (Windows), balenaEtcher (크로스플랫폼) 사용
```

### 2.2 BIOS/UEFI 설정

1. **Secure Boot**: Disabled (NVIDIA 드라이버 호환)
2. **SATA Mode**: AHCI
3. **Boot Order**: USB 우선

### 2.3 설치 옵션

1. "Install Ubuntu" 선택
2. 파티션 설정에서 **"Something else"** 선택
3. **시스템 디스크만** 파티셔닝:

```
/dev/nvme0n1 (또는 시스템 SSD):
├── /dev/nvme0n1p1  →  512MB   →  EFI System Partition
├── /dev/nvme0n1p2  →  1GB     →  /boot (ext4)
├── /dev/nvme0n1p3  →  32GB    →  swap (RAM과 동일하게, 최대 32GB)
└── /dev/nvme0n1p4  →  나머지  →  / (ext4)
```

4. **4TB, 2TB 디스크는 건드리지 않음** (설치 후 구성)
5. 부트로더 위치: 시스템 디스크 선택

---

## Part 3: 설치 후 데이터 디스크 구성

### 3.1 디스크 확인

```bash
# 연결된 디스크 목록
lsblk

# 예상 출력:
# NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
# nvme0n1     259:0    0   1TB  0 disk 
# ├─nvme0n1p1 259:1    0  512M  0 part /boot/efi
# ├─nvme0n1p2 259:2    0    1G  0 part /boot
# ├─nvme0n1p3 259:3    0   32G  0 part [SWAP]
# └─nvme0n1p4 259:4    0  966G  0 part /
# sda           8:0    0  3.6T  0 disk           ← 4TB
# sdb           8:16   0  1.8T  0 disk           ← 2TB

# 디스크 상세 정보
sudo fdisk -l /dev/sda /dev/sdb
```

### 3.2 파티션 생성

```bash
# 4TB 디스크 (sda)
sudo parted /dev/sda --script mklabel gpt
sudo parted /dev/sda --script mkpart primary ext4 0% 100%

# 2TB 디스크 (sdb)  
sudo parted /dev/sdb --script mklabel gpt
sudo parted /dev/sdb --script mkpart primary ext4 0% 100%

# 결과 확인
lsblk /dev/sda /dev/sdb
# NAME   SIZE TYPE
# sda    3.6T disk
# └─sda1 3.6T part
# sdb    1.8T disk
# └─sdb1 1.8T part
```

### 3.3 파일시스템 생성

```bash
# ext4 포맷 (라벨 지정으로 식별 용이)
sudo mkfs.ext4 -L "Storage" /dev/sda1
sudo mkfs.ext4 -L "Workspace" /dev/sdb1

# 라벨 확인
sudo blkid /dev/sda1 /dev/sdb1
# /dev/sda1: LABEL="Storage" UUID="xxxx-xxxx-..." TYPE="ext4"
# /dev/sdb1: LABEL="Workspace" UUID="yyyy-yyyy-..." TYPE="ext4"
```

### 3.4 마운트 포인트 생성 및 마운트

```bash
# 마운트 포인트 생성
sudo mkdir -p /mnt/storage    # 4TB
sudo mkdir -p /mnt/workspace  # 2TB

# 마운트
sudo mount /dev/sda1 /mnt/storage
sudo mount /dev/sdb1 /mnt/workspace

# 확인
df -h /mnt/storage /mnt/workspace
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/sda1       3.6T   28K  3.4T   1% /mnt/storage
# /dev/sdb1       1.8T   28K  1.7T   1% /mnt/workspace
```

### 3.5 자동 마운트 설정 (fstab)

```bash
# UUID 확인
sudo blkid /dev/sda1 /dev/sdb1

# fstab 백업
sudo cp /etc/fstab /etc/fstab.backup

# fstab에 추가 (UUID 사용 권장)
sudo tee -a /etc/fstab << 'EOF'

# Data Disks
LABEL=Storage    /mnt/storage    ext4    defaults,noatime    0    2
LABEL=Workspace  /mnt/workspace  ext4    defaults,noatime    0    2
EOF

# 또는 UUID로 (더 안정적)
# UUID=xxxx-xxxx  /mnt/storage    ext4  defaults,noatime  0  2
# UUID=yyyy-yyyy  /mnt/workspace  ext4  defaults,noatime  0  2

# fstab 검증 (오류 시 부팅 문제 발생하므로 중요!)
sudo mount -a
echo $?  # 0이면 성공

# 재부팅 테스트
sudo reboot
```

### 3.6 소유권 설정

```bash
# 현재 사용자에게 소유권 부여
sudo chown -R $USER:$USER /mnt/storage
sudo chown -R $USER:$USER /mnt/workspace

# 권한 설정
chmod 755 /mnt/storage
chmod 755 /mnt/workspace
```

---

## Part 4: 디렉토리 구조 생성

### 4.1 Storage (4TB) - 정적 데이터

```bash
# AlphaFold 데이터베이스 디렉토리
mkdir -p /mnt/storage/alphafold-db/{bfd,mgnify,pdb70,pdb_mmcif,uniclust30,uniprot,params}

# Alpamayo 데이터셋 디렉토리
mkdir -p /mnt/storage/alpamayo-dataset/{raw,processed,nurec}

# 아카이브
mkdir -p /mnt/storage/archives/{backups,old-experiments}

# 구조 확인
tree -L 2 /mnt/storage
# /mnt/storage
# ├── alphafold-db
# │   ├── bfd
# │   ├── mgnify
# │   ├── params
# │   ├── pdb70
# │   ├── pdb_mmcif
# │   ├── uniclust30
# │   └── uniprot
# ├── alpamayo-dataset
# │   ├── nurec
# │   ├── processed
# │   └── raw
# └── archives
#     ├── backups
#     └── old-experiments
```

### 4.2 Workspace (2TB) - 활성 작업

```bash
# 프로젝트 디렉토리
mkdir -p /mnt/workspace/projects/{alphafold,alpamayo}

# AlphaFold 프로젝트 구조
mkdir -p /mnt/workspace/projects/alphafold/{inputs,outputs,scripts}

# Alpamayo 프로젝트 구조
mkdir -p /mnt/workspace/projects/alpamayo/{models,outputs,configs,scripts}

# 공용 디렉토리
mkdir -p /mnt/workspace/{models,outputs,cache,downloads,tmp}

# 캐시 세부 디렉토리
mkdir -p /mnt/workspace/cache/{pip,huggingface,torch,docker}

# 구조 확인
tree -L 3 /mnt/workspace
# /mnt/workspace
# ├── cache
# │   ├── docker
# │   ├── huggingface
# │   ├── pip
# │   └── torch
# ├── downloads
# ├── models
# ├── outputs
# ├── projects
# │   ├── alphafold
# │   │   ├── inputs
# │   │   ├── outputs
# │   │   └── scripts
# │   └── alpamayo
# │       ├── configs
# │       ├── models
# │       ├── outputs
# │       └── scripts
# └── tmp
```

### 4.3 심볼릭 링크로 통합 구조 생성

```bash
# 홈 디렉토리에 통합 진입점 생성
mkdir -p ~/data

# 주요 링크 생성
ln -s /mnt/storage ~/data/storage
ln -s /mnt/workspace ~/data/workspace
ln -s /mnt/workspace/projects ~/data/projects
ln -s /mnt/workspace/cache ~/data/cache

# AlphaFold 통합 뷰 (DB는 storage, 작업은 workspace)
mkdir -p ~/data/alphafold
ln -s /mnt/storage/alphafold-db ~/data/alphafold/databases
ln -s /mnt/workspace/projects/alphafold/inputs ~/data/alphafold/inputs
ln -s /mnt/workspace/projects/alphafold/outputs ~/data/alphafold/outputs
ln -s /mnt/workspace/projects/alphafold/scripts ~/data/alphafold/scripts

# Alpamayo 통합 뷰 (Dataset은 storage, 나머지는 workspace)
mkdir -p ~/data/alpamayo
ln -s /mnt/storage/alpamayo-dataset ~/data/alpamayo/datasets
ln -s /mnt/workspace/projects/alpamayo/models ~/data/alpamayo/models
ln -s /mnt/workspace/projects/alpamayo/outputs ~/data/alpamayo/outputs
ln -s /mnt/workspace/projects/alpamayo/configs ~/data/alpamayo/configs
ln -s /mnt/workspace/projects/alpamayo/scripts ~/data/alpamayo/scripts

# 최종 구조 확인
tree -L 2 ~/data
# ~/data
# ├── alphafold
# │   ├── databases -> /mnt/storage/alphafold-db
# │   ├── inputs -> /mnt/workspace/projects/alphafold/inputs
# │   ├── outputs -> /mnt/workspace/projects/alphafold/outputs
# │   └── scripts -> /mnt/workspace/projects/alphafold/scripts
# ├── alpamayo
# │   ├── configs -> /mnt/workspace/projects/alpamayo/configs
# │   ├── datasets -> /mnt/storage/alpamayo-dataset
# │   ├── models -> /mnt/workspace/projects/alpamayo/models
# │   ├── outputs -> /mnt/workspace/projects/alpamayo/outputs
# │   └── scripts -> /mnt/workspace/projects/alpamayo/scripts
# ├── cache -> /mnt/workspace/cache
# ├── projects -> /mnt/workspace/projects
# ├── storage -> /mnt/storage
# └── workspace -> /mnt/workspace
```

---

## Part 5: 환경 변수 설정

```bash
# ~/.bashrc에 추가
cat >> ~/.bashrc << 'EOF'

# ============================================
# Data Directory Configuration
# ============================================

# Base paths
export STORAGE_DIR="/mnt/storage"
export WORKSPACE_DIR="/mnt/workspace"

# AlphaFold paths
export ALPHAFOLD_DATA_DIR="/mnt/storage/alphafold-db"
export ALPHAFOLD_OUTPUT_DIR="/mnt/workspace/projects/alphafold/outputs"

# Alpamayo paths
export ALPAMAYO_DATASET_DIR="/mnt/storage/alpamayo-dataset"
export ALPAMAYO_MODEL_DIR="/mnt/workspace/projects/alpamayo/models"
export ALPAMAYO_OUTPUT_DIR="/mnt/workspace/projects/alpamayo/outputs"

# Cache directories (시스템 디스크 부하 감소)
export PIP_CACHE_DIR="/mnt/workspace/cache/pip"
export HF_HOME="/mnt/workspace/cache/huggingface"
export HF_DATASETS_CACHE="/mnt/workspace/cache/huggingface/datasets"
export TORCH_HOME="/mnt/workspace/cache/torch"
export TRANSFORMERS_CACHE="/mnt/workspace/cache/huggingface/transformers"

# Docker data (선택사항 - 용량이 커질 수 있음)
# export DOCKER_DATA_ROOT="/mnt/workspace/cache/docker"

# Temp directory
export TMPDIR="/mnt/workspace/tmp"

# CUDA paths (설치 후 활성화)
# export PATH="/usr/local/cuda/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# ============================================
# Aliases
# ============================================
alias cddata='cd ~/data'
alias cdstorage='cd /mnt/storage'
alias cdwork='cd /mnt/workspace'
alias cdalpha='cd ~/data/alphafold'
alias cdmayo='cd ~/data/alpamayo'

# Disk usage check
alias diskcheck='df -h /mnt/storage /mnt/workspace'
alias dusage='du -sh /mnt/storage/* /mnt/workspace/* 2>/dev/null | sort -h'

EOF

# 적용
source ~/.bashrc
```

---

## Part 6: NVIDIA 드라이버 및 CUDA 설치

### 6.1 시스템 업데이트

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential dkms -y
```

### 6.2 NVIDIA 드라이버 설치

```bash
# 권장 드라이버 확인
ubuntu-drivers devices

# 출력 예시:
# driver   : nvidia-driver-560 - third-party non-free recommended
# driver   : nvidia-driver-550 - distro non-free

# 자동 설치 (권장)
sudo ubuntu-drivers autoinstall

# 또는 수동 설치 (RTX 50 시리즈용 최신)
# sudo apt install nvidia-driver-560 -y

# 재부팅 필수
sudo reboot
```

### 6.3 드라이버 확인

```bash
# NVIDIA 드라이버 상태
nvidia-smi

# 예상 출력:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 560.xx       Driver Version: 560.xx       CUDA Version: 12.x    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
# |  0%   45C    P8    15W / 175W |    500MiB / 24576MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### 6.4 CUDA Toolkit 설치

```bash
# CUDA 저장소 추가
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# CUDA Toolkit 설치 (드라이버 제외, 이미 설치됨)
sudo apt install cuda-toolkit-12-4 -y

# 환경 변수 활성화 (~/.bashrc에서 주석 해제)
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc

# 확인
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Built on ...
# Cuda compilation tools, release 12.4, V12.4.xxx
```

### 6.5 cuDNN 설치

```bash
sudo apt install libcudnn8 libcudnn8-dev -y

# 버전 확인
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

---

## Part 7: Docker 설치 (AlphaFold 권장 실행 환경)

### 7.1 Docker Engine 설치

```bash
# 기존 버전 제거
sudo apt remove docker docker-engine docker.io containerd runc 2>/dev/null

# 의존성 설치
sudo apt install ca-certificates curl gnupg lsb-release -y

# Docker GPG 키 추가
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Docker 저장소 추가
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y

# 사용자를 docker 그룹에 추가 (sudo 없이 사용)
sudo usermod -aG docker $USER
newgrp docker

# 서비스 활성화
sudo systemctl enable docker
sudo systemctl start docker

# 확인
docker --version
docker run hello-world
```

### 7.2 NVIDIA Container Toolkit (GPU 지원)

```bash
# 저장소 추가
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 설치
sudo apt update
sudo apt install nvidia-container-toolkit -y

# Docker 데몬 설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# GPU 지원 테스트
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 7.3 Docker 데이터 위치 변경 (선택사항)

```bash
# Docker 이미지가 많아지면 시스템 디스크 부족할 수 있음
# workspace 디스크로 이동

# Docker 중지
sudo systemctl stop docker

# 데이터 이동
sudo mv /var/lib/docker /mnt/workspace/cache/docker
sudo ln -s /mnt/workspace/cache/docker /var/lib/docker

# 또는 daemon.json 설정
sudo tee /etc/docker/daemon.json << 'EOF'
{
    "data-root": "/mnt/workspace/cache/docker",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# Docker 재시작
sudo systemctl start docker
```

---

## Part 8: Python 환경 설정

### 8.1 Miniconda 설치

```bash
# Miniconda 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /mnt/workspace/downloads/miniconda.sh

# 설치 (workspace에 설치하여 시스템 디스크 절약)
bash /mnt/workspace/downloads/miniconda.sh -b -p /mnt/workspace/miniconda3

# PATH 추가
echo 'export PATH="/mnt/workspace/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# conda 초기화
conda init bash
source ~/.bashrc

# 업데이트
conda update -n base -c defaults conda -y
```

### 8.2 AlphaFold 환경 생성

```bash
# AlphaFold용 conda 환경
conda create -n alphafold python=3.10 -y
conda activate alphafold

# 주요 패키지 (Docker 사용 시 불필요)
# pip install tensorflow==2.15.0
# pip install jax jaxlib

conda deactivate
```

### 8.3 Alpamayo 환경 생성

```bash
# Alpamayo용 conda 환경
conda create -n alpamayo python=3.10 -y
conda activate alpamayo

# uv 설치 (빠른 패키지 관리자)
pip install uv

# PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# HuggingFace
pip install transformers accelerate safetensors

# 기타 필수 패키지
pip install numpy scipy matplotlib jupyter

conda deactivate
```

---

## Part 9: 프로젝트별 설정

### 9.1 AlphaFold 설정

```bash
# AlphaFold 클론
cd /mnt/workspace/projects/alphafold
git clone https://github.com/google-deepmind/alphafold.git repo

# 데이터베이스 다운로드 스크립트 (시간 오래 걸림, ~2.5TB)
# 공식 스크립트 사용
cd repo
scripts/download_all_data.sh /mnt/storage/alphafold-db

# 또는 개별 다운로드
# scripts/download_bfd.sh /mnt/storage/alphafold-db
# scripts/download_mgnify.sh /mnt/storage/alphafold-db
# scripts/download_pdb70.sh /mnt/storage/alphafold-db
# scripts/download_uniclust30.sh /mnt/storage/alphafold-db
# scripts/download_uniref90.sh /mnt/storage/alphafold-db
# scripts/download_uniprot.sh /mnt/storage/alphafold-db
# scripts/download_pdb_mmcif.sh /mnt/storage/alphafold-db
# scripts/download_alphafold_params.sh /mnt/storage/alphafold-db
```

### 9.2 Alpamayo 설정

```bash
# Alpamayo 클론
cd /mnt/workspace/projects/alpamayo
git clone https://github.com/NVlabs/alpamayo.git repo

# 환경 설정
cd repo
conda activate alpamayo

# uv로 의존성 설치
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active

# HuggingFace 인증 (gated 모델 접근)
huggingface-cli login
# 토큰 입력: https://huggingface.co/settings/tokens

# 모델 다운로드 (~22GB)
# 자동으로 $HF_HOME에 캐시됨
python src/alpamayo_r1/test_inference.py
```

---

## Part 10: 유지보수 및 모니터링

### 10.1 디스크 사용량 모니터링 스크립트

```bash
# ~/bin/disk-monitor.sh 생성
mkdir -p ~/bin
cat > ~/bin/disk-monitor.sh << 'EOF'
#!/bin/bash
echo "========================================"
echo "Disk Usage Report - $(date)"
echo "========================================"
echo ""
echo "=== Mount Points ==="
df -h /mnt/storage /mnt/workspace
echo ""
echo "=== Storage (4TB) Top Directories ==="
du -sh /mnt/storage/*/ 2>/dev/null | sort -hr | head -10
echo ""
echo "=== Workspace (2TB) Top Directories ==="
du -sh /mnt/workspace/*/ 2>/dev/null | sort -hr | head -10
echo ""
echo "=== Cache Usage ==="
du -sh /mnt/workspace/cache/*/ 2>/dev/null | sort -hr
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
EOF

chmod +x ~/bin/disk-monitor.sh

# 실행
~/bin/disk-monitor.sh
```

### 10.2 자동 정리 스크립트

```bash
# ~/bin/cleanup.sh 생성
cat > ~/bin/cleanup.sh << 'EOF'
#!/bin/bash
echo "Cleaning up temporary files..."

# pip 캐시 정리
pip cache purge 2>/dev/null

# HuggingFace 캐시에서 오래된 파일 정리 (30일 이상)
find /mnt/workspace/cache/huggingface -type f -atime +30 -delete 2>/dev/null

# tmp 디렉토리 정리
rm -rf /mnt/workspace/tmp/* 2>/dev/null

# Docker 정리 (사용하지 않는 이미지/컨테이너)
docker system prune -f 2>/dev/null

echo "Cleanup complete!"
df -h /mnt/storage /mnt/workspace
EOF

chmod +x ~/bin/cleanup.sh
```

### 10.3 백업 스크립트

```bash
# ~/bin/backup-configs.sh 생성
cat > ~/bin/backup-configs.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/mnt/storage/archives/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 중요 설정 백업
cp ~/.bashrc $BACKUP_DIR/
cp /etc/fstab $BACKUP_DIR/

# conda 환경 목록
conda env list > $BACKUP_DIR/conda-envs.txt

# pip 패키지 목록
pip freeze > $BACKUP_DIR/pip-requirements.txt

# 프로젝트 설정 파일들
tar -czf $BACKUP_DIR/project-configs.tar.gz \
    /mnt/workspace/projects/*/configs/ \
    /mnt/workspace/projects/*/*.yaml \
    /mnt/workspace/projects/*/*.json \
    2>/dev/null

echo "Backup saved to: $BACKUP_DIR"
ls -la $BACKUP_DIR
EOF

chmod +x ~/bin/backup-configs.sh
```

---

## Part 11: 검증 체크리스트

```bash
# 전체 검증 스크립트
cat > ~/bin/verify-setup.sh << 'EOF'
#!/bin/bash
echo "=========================================="
echo "System Verification - $(date)"
echo "=========================================="

# 1. 디스크 마운트
echo -e "\n[1/8] Disk Mounts"
if mountpoint -q /mnt/storage && mountpoint -q /mnt/workspace; then
    echo "✓ Both disks mounted"
    df -h /mnt/storage /mnt/workspace
else
    echo "✗ Disk mount issue!"
fi

# 2. 심볼릭 링크
echo -e "\n[2/8] Symbolic Links"
for link in ~/data/alphafold/databases ~/data/alpamayo/datasets; do
    if [ -L "$link" ] && [ -e "$link" ]; then
        echo "✓ $link -> $(readlink $link)"
    else
        echo "✗ $link broken or missing"
    fi
done

# 3. NVIDIA 드라이버
echo -e "\n[3/8] NVIDIA Driver"
if nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ NVIDIA driver working"
else
    echo "✗ NVIDIA driver not found"
fi

# 4. CUDA
echo -e "\n[4/8] CUDA Toolkit"
if nvcc --version &>/dev/null; then
    nvcc --version | grep "release"
    echo "✓ CUDA installed"
else
    echo "✗ CUDA not found"
fi

# 5. Docker + GPU
echo -e "\n[5/8] Docker GPU Support"
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "✓ Docker GPU support working"
else
    echo "✗ Docker GPU support failed"
fi

# 6. Conda 환경
echo -e "\n[6/8] Conda Environments"
conda env list | grep -E "(alphafold|alpamayo)"

# 7. 환경 변수
echo -e "\n[7/8] Environment Variables"
echo "STORAGE_DIR=$STORAGE_DIR"
echo "WORKSPACE_DIR=$WORKSPACE_DIR"
echo "HF_HOME=$HF_HOME"

# 8. 쓰기 권한
echo -e "\n[8/8] Write Permissions"
for dir in /mnt/storage /mnt/workspace; do
    if touch "$dir/.test" 2>/dev/null && rm "$dir/.test"; then
        echo "✓ $dir writable"
    else
        echo "✗ $dir not writable"
    fi
done

echo -e "\n=========================================="
echo "Verification complete!"
echo "=========================================="
EOF

chmod +x ~/bin/verify-setup.sh

# 실행
~/bin/verify-setup.sh
```

---

## 문제 해결

### 부팅 시 디스크 마운트 안 됨

```bash
# Live USB로 부팅 후
sudo mount /dev/nvme0n1p4 /mnt
sudo nano /mnt/etc/fstab
# 문제 있는 라인 수정 또는 주석 처리
```

### 권한 문제

```bash
sudo chown -R $USER:$USER /mnt/storage /mnt/workspace
```

### 심볼릭 링크 깨짐

```bash
# 링크 재생성
rm ~/data/alphafold/databases
ln -s /mnt/storage/alphafold-db ~/data/alphafold/databases
```

### NVIDIA 드라이버 문제

```bash
# 드라이버 재설치
sudo apt purge nvidia-* -y
sudo apt autoremove -y
sudo ubuntu-drivers autoinstall
sudo reboot
```

---

## 요약: 최종 디렉토리 구조

```
/mnt/storage (4TB) - 정적 데이터
├── alphafold-db/          # ~2.5TB
├── alpamayo-dataset/      # ~1TB
└── archives/

/mnt/workspace (2TB) - 활성 작업
├── projects/
│   ├── alphafold/
│   └── alpamayo/
├── models/
├── outputs/
├── cache/
│   ├── pip/
│   ├── huggingface/
│   ├── torch/
│   └── docker/
├── miniconda3/
└── tmp/

~/data (심볼릭 링크 통합 뷰)
├── alphafold/
│   ├── databases -> /mnt/storage/alphafold-db
│   ├── inputs -> /mnt/workspace/projects/alphafold/inputs
│   └── outputs -> /mnt/workspace/projects/alphafold/outputs
├── alpamayo/
│   ├── datasets -> /mnt/storage/alpamayo-dataset
│   ├── models -> /mnt/workspace/projects/alpamayo/models
│   └── outputs -> /mnt/workspace/projects/alpamayo/outputs
└── ...
```

---

*가이드 생성일: 2026-01-07*
*대상 환경: ASUS ROG Strix SCAR 16 G635LX (RTX 5090 Laptop, 24GB VRAM)*
*디스크 구성: 4TB + 2TB 분리 사용*
