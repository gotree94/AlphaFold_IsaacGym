# 완전 가이드: AlphaFold + Alpamayo + Isaac Lab 설치
## Ubuntu 22.04 LTS | ASUS ROG Strix SCAR 16 G635LX | 4TB + 2TB

---

# 목차

1. [개요](#part-0-개요)
2. [Ubuntu 설치](#part-1-ubuntu-2204-lts-설치)
3. [디스크 구성](#part-2-디스크-구성-4tb--2tb-분리)
4. [기반 시스템](#part-3-기반-시스템-설정)
5. [Isaac Lab 설치](#part-4-isaac-lab--isaac-sim-설치)
6. [AlphaFold 설치](#part-5-alphafoldcolabfold-설치)
7. [Alpamayo 설치](#part-6-alpamayo-r1-설치)
8. [환경 관리](#part-7-환경-관리-및-전환)
9. [검증](#part-8-전체-검증)
10. [문제 해결](#part-9-문제-해결)

---

# Part 0: 개요

## 시스템 사양

| 항목 | 사양 |
|------|------|
| 노트북 | ASUS ROG Strix SCAR 16 G635LX-RW047W |
| GPU | NVIDIA RTX 5090 Laptop (24GB GDDR7) |
| CPU | Intel Core Ultra 9 275HX |
| RAM | 32GB DDR5 (64GB 권장, 업그레이드 고려) |
| Storage | 시스템 SSD + 4TB + 2TB |
| OS | Ubuntu 22.04 LTS |

## 설치할 프로젝트

| 프로젝트 | 용도 | Python | 환경 이름 |
|---------|------|--------|----------|
| **Isaac Lab** | 로봇 시뮬레이션 + RL | 3.11 | `isaaclab` |
| **Isaac Sim** | 시뮬레이터 엔진 | 3.11 | (Isaac Lab에 포함) |
| **AlphaFold/ColabFold** | 단백질 구조 예측 | 3.10 | `alphafold` |
| **Alpamayo** | 자율주행 VLA 모델 | 3.10 | `alpamayo` |

## 디스크 구성 전략

```
┌─────────────────────────────────────────────────────────────────┐
│ 시스템 SSD (내장)                                               │
│ └── Ubuntu 22.04 LTS (/, /boot, swap)                          │
├─────────────────────────────────────────────────────────────────┤
│ 4TB 디스크 → /mnt/storage (정적/대용량 데이터)                  │
│ ├── alphafold-db/        유전자 DB (~2.5TB, 선택)              │
│ ├── alpamayo-dataset/    Physical AI Dataset (~1TB)            │
│ └── archives/            백업, 아카이브                         │
├─────────────────────────────────────────────────────────────────┤
│ 2TB 디스크 → /mnt/workspace (활성 작업 공간)                    │
│ ├── miniconda3/          Conda 설치                            │
│ ├── projects/            모든 프로젝트                          │
│ ├── models/              모델 가중치                            │
│ └── cache/               pip, HuggingFace 캐시                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Part 1: Ubuntu 22.04 LTS 설치

## 1.1 부팅 USB 생성

다른 PC에서:
```bash
# Linux/Mac
sudo dd if=ubuntu-22.04.4-desktop-amd64.iso of=/dev/sdX bs=4M status=progress sync

# Windows: Rufus 또는 balenaEtcher 사용
```

## 1.2 BIOS 설정 (F2 또는 DEL)

| 설정 | 값 | 이유 |
|------|-----|------|
| Secure Boot | **Disabled** | NVIDIA 드라이버 호환 |
| SATA Mode | AHCI | 표준 모드 |
| Boot Priority | USB First | USB 부팅 |

## 1.3 Ubuntu 설치

1. USB 부팅 → "Install Ubuntu" 선택
2. 언어: 한국어 또는 English
3. 키보드: Korean (101/104)
4. 설치 유형: **"Something else"** (수동 파티션)

## 1.4 파티션 설정 (시스템 SSD만)

시스템 SSD(예: `/dev/nvme0n1`)만 파티셔닝:

| 파티션 | 크기 | 타입 | 마운트 |
|--------|------|------|--------|
| nvme0n1p1 | 512MB | EFI System Partition | /boot/efi |
| nvme0n1p2 | 1GB | ext4 | /boot |
| nvme0n1p3 | 32GB | swap | swap |
| nvme0n1p4 | 나머지 | ext4 | / |

> ⚠️ **4TB, 2TB 디스크는 건드리지 않음** (설치 후 구성)

5. 부트로더 위치: `/dev/nvme0n1` 선택
6. 설치 진행 → 재부팅

---

# Part 2: 디스크 구성 (4TB + 2TB 분리)

## 2.1 디스크 확인

```bash
# 연결된 디스크 확인
lsblk

# 예상 출력:
# NAME        SIZE  TYPE MOUNTPOINT
# nvme0n1      1T   disk 
# ├─nvme0n1p1 512M  part /boot/efi
# ├─nvme0n1p2   1G  part /boot
# ├─nvme0n1p3  32G  part [SWAP]
# └─nvme0n1p4 966G  part /
# sda         3.6T  disk            ← 4TB
# sdb         1.8T  disk            ← 2TB
```

## 2.2 파티션 생성

```bash
# 2TB 디스크 (nvme1n1)
sudo parted /dev/nvme1n1 --script mklabel gpt
sudo parted /dev/nvme1n1 --script mkpart primary ext4 0% 100%

# 확인
lsblk /dev/nvme1n1
```

## 2.3 파일시스템 생성

```bash
# ext4 포맷
sudo mkfs.ext4 -L "Workspace" /dev/nvme1n1p1

# 마운트 포인트 생성
sudo mkdir -p /mnt/workspace

# 마운트
sudo mount /dev/nvme1n1p1 /mnt/workspace

# 소유권 설정
sudo chown -R $USER:$USER /mnt/workspace

# 확인
df -h /mnt/workspace
```

## 2.4 마운트 포인트 생성 및 마운트

```bash
# 마운트 포인트 생성
sudo mkdir -p /mnt/workspace

# 마운트
sudo mount /dev/nvme1n1p1 /mnt/workspace

# 확인
df -h /mnt/workspace
```

## 2.5 자동 마운트 설정 (fstab)

```bash
# fstab 백업
sudo cp /etc/fstab /etc/fstab.backup

# fstab에 추가
echo 'LABEL=Workspace  /mnt/workspace  ext4  defaults,noatime  0  2' | sudo tee -a /etc/fstab

# 검증 (오류 없어야 함)
sudo mount -a
echo $?  # 0이면 성공

# 재부팅 테스트
sudo reboot
```

## 2.6 소유권 설정

```bash
# 현재 사용자에게 소유권 부여
sudo chown -R $USER:$USER /mnt/workspace
```

## 2.7 디렉토리 구조 생성

```bash
# Workspace (2TB) - 데이터/캐시 저장소
mkdir -p /mnt/workspace/{datasets,models,cache,outputs,tmp}
mkdir -p /mnt/workspace/datasets/{alphafold-db,alpamayo}
mkdir -p /mnt/workspace/cache/{pip,huggingface,torch}

# 홈 디렉토리 (4TB) - 프로젝트 코드
mkdir -p ~/projects/{alphafold,alpamayo,isaaclab}
mkdir -p ~/projects/alphafold/{inputs,outputs}

# 심볼릭 링크 (편의용)
ln -s /mnt/workspace ~/workspace
```

---

# Part 3: 기반 시스템 설정

## 3.1 시스템 업데이트

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential git curl wget unzip \
    software-properties-common apt-transport-https \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    vulkan-tools mesa-vulkan-drivers
```

## 3.2 NVIDIA 드라이버 설치

```bash
# 기존 드라이버 제거 (혹시 있다면)
sudo apt purge nvidia-* -y 2>/dev/null
sudo apt autoremove -y

# 권장 드라이버 확인
ubuntu-drivers devices

# 자동 설치 (560 이상)
sudo ubuntu-drivers autoinstall

# 재부팅
sudo reboot
```

### 드라이버 확인

```bash
nvidia-smi

# 예상 출력:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 560.xx       Driver Version: 560.xx       CUDA Version: 12.x    |
# |-------------------------------+----------------------+----------------------+
# |   0  NVIDIA GeForce RTX 5090  |   24576MiB          |                      |
# +-------------------------------+----------------------+----------------------+
```

## 3.3 CUDA Toolkit 12.4 설치

```bash
# CUDA 저장소 추가
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# CUDA Toolkit 설치
sudo apt install cuda-toolkit-12-4 -y

# 환경 변수 설정
cat >> ~/.bashrc << 'EOF'

# CUDA
export PATH="/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
EOF

source ~/.bashrc

# 확인
nvcc --version
```

## 3.4 cuDNN 설치

```bash
sudo apt install libcudnn9-cuda-12 libcudnn9-dev-cuda-12 -y
```

## 3.5 Docker + NVIDIA Container Toolkit

```bash
# Docker 설치
sudo apt install docker.io -y
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install nvidia-container-toolkit -y
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 로그아웃 후 다시 로그인, 또는:
newgrp docker

# 테스트
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 3.6 Miniconda 설치

```bash
# 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /mnt/workspace/downloads/miniconda.sh

# 설치 (workspace에 설치)
bash /mnt/workspace/downloads/miniconda.sh -b -p /mnt/workspace/miniconda3

# PATH 설정
cat >> ~/.bashrc << 'EOF'

# Miniconda
export PATH="/mnt/workspace/miniconda3/bin:$PATH"
EOF

source ~/.bashrc
conda init bash
source ~/.bashrc

# 업데이트
conda update -n base -c defaults conda -y
```

## 3.7 환경 변수 설정

```bash
cat >> ~/.bashrc << 'EOF'

# ============================================
# Project Directories
# ============================================
export STORAGE_DIR="/mnt/storage"
export WORKSPACE_DIR="/mnt/workspace"

# Cache (시스템 디스크 부하 감소)
export PIP_CACHE_DIR="/mnt/workspace/cache/pip"
export HF_HOME="/mnt/workspace/cache/huggingface"
export TORCH_HOME="/mnt/workspace/cache/torch"
export TMPDIR="/mnt/workspace/tmp"

# Project-specific
export ALPHAFOLD_DATA_DIR="/mnt/storage/alphafold-db"
export ALPAMAYO_DIR="/mnt/workspace/projects/alpamayo"
export ISAACLAB_PATH="/mnt/workspace/projects/isaaclab"
EOF

source ~/.bashrc
```

---

# Part 4: Isaac Lab + Isaac Sim 설치

> Isaac Lab을 설치하면 Isaac Sim이 함께 설치됨.
> Isaac Gym은 deprecated되어 Isaac Lab에 통합됨.

## 4.1 Conda 환경 생성

```bash
# Python 3.11 필수 (Isaac Sim 5.x 요구사항)
conda create -n isaaclab python=3.11 -y
conda activate isaaclab
```

## 4.2 Isaac Sim 설치 (pip 방식)

```bash
# pip 업그레이드
pip install --upgrade pip

# Isaac Sim 설치 (~15GB 다운로드, 시간 소요)
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

# 설치 확인
python -c "import isaacsim; print('Isaac Sim OK')"
```

## 4.3 Isaac Lab 설치

```bash
cd /mnt/workspace/projects/isaaclab

# Isaac Lab 클론
git clone https://github.com/isaac-sim/IsaacLab.git repo
cd repo

# 설치
./isaaclab.sh --install

# 또는 수동 설치
pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
```

## 4.4 RL 라이브러리 설치

```bash
# RSL-RL (ETH Zurich - 가장 많이 사용)
pip install rsl-rl

# rl_games
pip install rl-games

# Stable Baselines 3
pip install stable-baselines3

# SKRL (선택)
pip install skrl
```

## 4.5 설치 확인

```bash
conda activate isaaclab
cd /mnt/workspace/projects/isaaclab/repo

# 기본 테스트 (headless 모드)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

# 강화학습 테스트 (Cartpole)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Cartpole-v0 --headless --num_envs=64

# 더 복잡한 환경 (Ant)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-v0 --headless --num_envs=256

conda deactivate
```

---

# Part 5: AlphaFold/ColabFold 설치

> LocalColabFold 사용 (전체 DB 2.5TB 불필요, ~30GB로 실행 가능)

## 5.1 Conda 환경 생성

```bash
conda create -n alphafold python=3.10 -y
conda activate alphafold
```

## 5.2 LocalColabFold 설치

```bash
cd /mnt/workspace/projects/alphafold

# 설치 스크립트 다운로드
wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh

# 설치 실행 (~10-15분)
bash install_colabbatch_linux.sh

# PATH 추가
cat >> ~/.bashrc << 'EOF'

# LocalColabFold
export PATH="/mnt/workspace/projects/alphafold/localcolabfold/colabfold-conda/bin:$PATH"
EOF

source ~/.bashrc
```

## 5.3 모델 파라미터 다운로드

```bash
conda activate alphafold

# 모델 가중치 다운로드 (~15GB)
python -m colabfold.download

# 캐시를 workspace로 이동 (선택)
mkdir -p /mnt/workspace/models/colabfold
mv ~/.cache/colabfold/* /mnt/workspace/models/colabfold/ 2>/dev/null
rm -rf ~/.cache/colabfold
ln -s /mnt/workspace/models/colabfold ~/.cache/colabfold
```

## 5.4 설치 확인

```bash
conda activate alphafold

# 버전 확인
colabfold_batch --help

# 테스트 예측
mkdir -p /mnt/workspace/projects/alphafold/{inputs,outputs}

echo ">test_protein
MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTG" > /mnt/workspace/projects/alphafold/inputs/test.fasta

colabfold_batch \
    /mnt/workspace/projects/alphafold/inputs/test.fasta \
    /mnt/workspace/projects/alphafold/outputs/test_result

# 결과 확인
ls /mnt/workspace/projects/alphafold/outputs/test_result/

conda deactivate
```

## 5.5 (선택) 전체 데이터베이스 설치

오프라인 사용 또는 대량 배치 처리 시에만 필요 (~2.5TB)

```bash
# Docker 방식으로 AlphaFold 전체 설치
cd /mnt/workspace/projects/alphafold
git clone https://github.com/google-deepmind/alphafold.git alphafold-full
cd alphafold-full

# 데이터베이스 다운로드 (매우 오래 걸림 - 수일 소요 가능)
# scripts/download_all_data.sh /mnt/storage/alphafold-db
```

---

# Part 6: Alpamayo-R1 설치

## 6.1 Conda 환경 생성

```bash
conda create -n alpamayo python=3.10 -y
conda activate alpamayo
```

## 6.2 PyTorch 설치

```bash
# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 확인
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 6.3 Alpamayo 클론 및 설치

```bash
cd /mnt/workspace/projects/alpamayo

# 레포지토리 클론
git clone https://github.com/NVlabs/alpamayo.git repo
cd repo

# uv 설치 (빠른 패키지 관리자)
pip install uv

# 가상환경 생성 및 의존성 설치
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
```

## 6.4 HuggingFace 인증

```bash
# HuggingFace CLI
pip install huggingface_hub

# 로그인
huggingface-cli login
# 토큰 입력 (https://huggingface.co/settings/tokens)
```

### Gated 리소스 접근 요청 (웹 브라우저에서)

아래 링크에서 각각 "Request Access" 클릭:
- https://huggingface.co/nvidia/Alpamayo-R1-10B
- https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

## 6.5 테스트 추론

```bash
conda activate alpamayo
cd /mnt/workspace/projects/alpamayo/repo
source ar1_venv/bin/activate

# 테스트 (모델 자동 다운로드 ~22GB)
python src/alpamayo_r1/test_inference.py

# 또는 Jupyter 노트북
pip install jupyter
jupyter notebook notebook/inference.ipynb

deactivate
conda deactivate
```

---

# Part 7: 환경 관리 및 전환

## 7.1 활성화 스크립트 생성

```bash
mkdir -p ~/bin

# Isaac Lab 환경
cat > ~/bin/activate-isaaclab << 'EOF'
#!/bin/bash
echo "🤖 Activating Isaac Lab environment..."
source /mnt/workspace/miniconda3/bin/activate isaaclab
cd /mnt/workspace/projects/isaaclab/repo
echo ""
echo "Environment: isaaclab (Python 3.11)"
echo "Isaac Sim + Isaac Lab ready"
echo ""
echo "Examples:"
echo "  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless"
echo "  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless"
EOF

# AlphaFold 환경
cat > ~/bin/activate-alphafold << 'EOF'
#!/bin/bash
echo "🧬 Activating AlphaFold/ColabFold environment..."
source /mnt/workspace/miniconda3/bin/activate alphafold
cd /mnt/workspace/projects/alphafold
echo ""
echo "Environment: alphafold (Python 3.10)"
echo "LocalColabFold ready"
echo ""
echo "Usage:"
echo "  colabfold_batch input.fasta output_dir/"
echo "  colabfold_batch --help"
EOF

# Alpamayo 환경
cat > ~/bin/activate-alpamayo << 'EOF'
#!/bin/bash
echo "🚗 Activating Alpamayo environment..."
source /mnt/workspace/miniconda3/bin/activate alpamayo
cd /mnt/workspace/projects/alpamayo/repo
source ar1_venv/bin/activate
echo ""
echo "Environment: alpamayo (Python 3.10 + uv venv)"
echo "Alpamayo-R1 ready"
echo ""
echo "Usage:"
echo "  python src/alpamayo_r1/test_inference.py"
echo "  jupyter notebook notebook/inference.ipynb"
EOF

# 실행 권한
chmod +x ~/bin/activate-*

# PATH 추가
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 7.2 사용법

```bash
# Isaac Lab 작업
source activate-isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless

# AlphaFold 작업
source activate-alphafold
colabfold_batch my_protein.fasta results/

# Alpamayo 작업
source activate-alpamayo
python src/alpamayo_r1/test_inference.py

# 환경 비활성화
conda deactivate
```

## 7.3 디스크 모니터링 스크립트

```bash
cat > ~/bin/disk-status << 'EOF'
#!/bin/bash
echo "========================================"
echo "Disk Status - $(date)"
echo "========================================"
echo ""
echo "=== Mount Points ==="
df -h /mnt/storage /mnt/workspace /
echo ""
echo "=== Storage (4TB) Usage ==="
du -sh /mnt/storage/*/ 2>/dev/null | sort -hr
echo ""
echo "=== Workspace (2TB) Usage ==="
du -sh /mnt/workspace/*/ 2>/dev/null | sort -hr
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv
EOF

chmod +x ~/bin/disk-status
```

---

# Part 8: 전체 검증

## 8.1 검증 스크립트 생성

```bash
cat > ~/bin/verify-all << 'EOF'
#!/bin/bash
echo "=========================================="
echo "Complete System Verification"
echo "$(date)"
echo "=========================================="

echo -e "\n[1/6] System Info"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"

echo -e "\n[2/6] Disk Mounts"
if mountpoint -q /mnt/storage && mountpoint -q /mnt/workspace; then
    echo "✓ Storage (4TB): $(df -h /mnt/storage | tail -1 | awk '{print $4}') available"
    echo "✓ Workspace (2TB): $(df -h /mnt/workspace | tail -1 | awk '{print $4}') available"
else
    echo "✗ Disk mount issue!"
fi

echo -e "\n[3/6] NVIDIA & CUDA"
if nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | cut -d',' -f1)"
    echo "✓ NVIDIA stack OK"
else
    echo "✗ NVIDIA driver issue"
fi

echo -e "\n[4/6] Isaac Lab"
source /mnt/workspace/miniconda3/bin/activate isaaclab 2>/dev/null
if python -c "import isaacsim" 2>/dev/null; then
    echo "✓ Isaac Sim OK"
    echo "✓ Isaac Lab OK"
else
    echo "✗ Isaac Lab issue"
fi
conda deactivate 2>/dev/null

echo -e "\n[5/6] AlphaFold/ColabFold"
source /mnt/workspace/miniconda3/bin/activate alphafold 2>/dev/null
if command -v colabfold_batch &>/dev/null; then
    echo "✓ ColabFold OK"
else
    echo "✗ ColabFold issue"
fi
conda deactivate 2>/dev/null

echo -e "\n[6/6] Alpamayo"
source /mnt/workspace/miniconda3/bin/activate alpamayo 2>/dev/null
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "✓ PyTorch + CUDA OK"
    echo "✓ Alpamayo environment OK"
else
    echo "✗ Alpamayo issue"
fi
conda deactivate 2>/dev/null

echo -e "\n=========================================="
echo "Verification Complete"
echo "=========================================="
EOF

chmod +x ~/bin/verify-all
```

## 8.2 검증 실행

```bash
verify-all
```

예상 출력:
```
==========================================
Complete System Verification
==========================================

[1/6] System Info
OS: Ubuntu 22.04.4 LTS
Kernel: 6.x.x-generic

[2/6] Disk Mounts
✓ Storage (4TB): 3.4T available
✓ Workspace (2TB): 1.7T available

[3/6] NVIDIA & CUDA
NVIDIA GeForce RTX 5090 Laptop GPU, 24576 MiB, 560.xx
CUDA: 12.4
✓ NVIDIA stack OK

[4/6] Isaac Lab
✓ Isaac Sim OK
✓ Isaac Lab OK

[5/6] AlphaFold/ColabFold
✓ ColabFold OK

[6/6] Alpamayo
✓ PyTorch + CUDA OK
✓ Alpamayo environment OK

==========================================
Verification Complete
==========================================
```

---

# Part 9: 문제 해결

## 디스크 마운트 안 됨

```bash
# fstab 확인
cat /etc/fstab

# 수동 마운트 시도
sudo mount -a

# 오류 시 Live USB로 부팅 후 fstab 수정
```

## NVIDIA 드라이버 문제

```bash
# Secure Boot 확인 (비활성화 필요)
mokutil --sb-state

# 드라이버 재설치
sudo apt purge nvidia-* -y
sudo apt autoremove -y
sudo ubuntu-drivers autoinstall
sudo reboot
```

## Isaac Sim 실행 안 됨

```bash
# Vulkan 확인
vulkaninfo | head -20

# 헤드리스 모드로 테스트
conda activate isaaclab
python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('OK'); app.close()"
```

## ColabFold MSA 서버 연결 실패

```bash
# 인터넷 연결 확인
curl -I https://api.colabfold.com

# 로컬 MSA 사용 (선택)
colabfold_batch input.fasta output/ --local-search
```

## Alpamayo 모델 다운로드 실패

```bash
# HuggingFace 인증 확인
huggingface-cli whoami

# 캐시 정리 후 재시도
rm -rf /mnt/workspace/cache/huggingface/hub/models--nvidia--Alpamayo*
python src/alpamayo_r1/test_inference.py
```

## 메모리 부족 (OOM)

```bash
# Isaac Lab: 환경 수 줄이기
./isaaclab.sh -p train.py --task=Isaac-Ant-v0 --num_envs=64  # 256 대신 64

# Alpamayo: 배치 크기 줄이기
# test_inference.py에서 num_traj_samples=1로 설정

# AlphaFold: 작은 단백질부터 테스트
```

---

# 최종 디렉토리 구조

```
/mnt/storage (4TB) - 정적 데이터
├── alphafold-db/           # (선택) 전체 DB ~2.5TB
├── alpamayo-dataset/       # Physical AI Dataset
└── archives/

/mnt/workspace (2TB) - 활성 작업
├── miniconda3/             # Conda
├── projects/
│   ├── alphafold/
│   │   ├── localcolabfold/ # ColabFold 설치
│   │   ├── inputs/
│   │   └── outputs/
│   ├── alpamayo/
│   │   └── repo/           # Alpamayo-R1
│   └── isaaclab/
│       └── repo/           # Isaac Lab + Isaac Sim
├── models/
│   └── colabfold/          # AlphaFold 모델 파라미터
├── cache/
│   ├── pip/
│   ├── huggingface/        # Alpamayo 모델 캐시
│   └── torch/
├── downloads/
└── tmp/

~/bin/
├── activate-isaaclab       # 🤖 로봇 시뮬레이션
├── activate-alphafold      # 🧬 단백질 예측
├── activate-alpamayo       # 🚗 자율주행
├── disk-status             # 디스크 상태
└── verify-all              # 전체 검증
```

---

# 용량 예상

| 항목 | 용량 | 디스크 |
|------|------|--------|
| Isaac Sim + Lab | ~20GB | Workspace |
| ColabFold + 모델 | ~30GB | Workspace |
| Alpamayo 모델 | ~25GB | Workspace (cache) |
| 기타 (conda, cache) | ~20GB | Workspace |
| **Workspace 사용** | **~100GB** | 2TB 중 |
| AlphaFold 전체 DB (선택) | ~2.5TB | Storage |
| Physical AI Dataset (선택) | ~1TB | Storage |
| **Storage 사용** | **~3.5TB** | 4TB 중 |

---

*가이드 작성일: 2026-01-17*
*대상: ASUS ROG Strix SCAR 16 G635LX-RW047W*
*RTX 5090 Laptop (24GB) | 4TB + 2TB | Ubuntu 22.04 LTS*
