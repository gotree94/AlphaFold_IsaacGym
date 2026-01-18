# 완전 가이드: AlphaFold + Alpamayo + Isaac Lab 설치
## Ubuntu 22.04 LTS | ASUS ROG Strix SCAR 16 G635LX-RW047W

---

# 목차

1. [개요](#part-0-개요)
2. [Ubuntu 설치](#part-1-ubuntu-2204-lts-설치) ✅ 완료
3. [NVIDIA 드라이버](#part-2-nvidia-드라이버-설치) ✅ 완료
4. [디스크 구성](#part-3-디스크-구성-2tb-데이터-디스크)
5. [기반 시스템](#part-4-기반-시스템-설정)
6. [Isaac Lab 설치](#part-5-isaac-lab--isaac-sim-설치)
7. [AlphaFold 설치](#part-6-alphafoldcolabfold-설치)
8. [Alpamayo 설치](#part-7-alpamayo-r1-설치)
9. [환경 관리](#part-8-환경-관리-및-전환)
10. [검증](#part-9-전체-검증)
11. [문제 해결](#part-10-문제-해결)

---

# Part 0: 개요

## 시스템 사양

| 항목 | 사양 |
|------|------|
| 노트북 | ASUS ROG Strix SCAR 16 G635LX-RW047W |
| GPU | NVIDIA RTX 5090 Laptop (24GB GDDR7) |
| CPU | Intel Core Ultra 9 275HX |
| RAM | 32GB DDR5 |
| Storage | 4TB NVMe (시스템) + 2TB NVMe (데이터) |
| OS | Ubuntu 22.04 LTS |

## 설치할 프로젝트

| 프로젝트 | 용도 | Python | 환경 이름 |
|---------|------|--------|----------|
| **Isaac Lab** | 로봇 시뮬레이션 + RL | 3.11 | `isaaclab` |
| **Isaac Sim** | 시뮬레이터 엔진 | 3.11 | (Isaac Lab에 포함) |
| **AlphaFold/ColabFold** | 단백질 구조 예측 | 3.10 | `alphafold` |
| **Alpamayo** | 자율주행 VLA 모델 | 3.10 | `alpamayo` |

## 디스크 구성

```
nvme0n1 (4TB) - Ubuntu 시스템 디스크
├── /                        # 시스템
├── /boot                    # 부트
├── /boot/efi                # EFI
└── /home/$USER              # 홈 디렉토리
    ├── miniconda3/          # Conda 설치
    └── projects/            # 프로젝트 코드
        ├── alphafold/
        ├── alpamayo/
        └── isaaclab/

nvme1n1 (2TB) - 데이터 디스크
└── /mnt/workspace
    ├── datasets/            # 대용량 데이터셋
    │   ├── alphafold-db/    # AlphaFold DB (선택, ~2.5TB)
    │   └── alpamayo/        # Physical AI Dataset
    ├── models/              # 모델 가중치
    ├── cache/               # pip, HuggingFace 캐시
    │   ├── pip/
    │   ├── huggingface/
    │   └── torch/
    └── outputs/             # 실험 결과물
```

---

# Part 1: Ubuntu 22.04 LTS 설치 ✅ 완료

## 1.1 BIOS 설정 (F2 또는 DEL)

| 설정 | 값 | 이유 |
|------|-----|------|
| Secure Boot | **Disabled** | NVIDIA 드라이버 호환 |
| SATA Mode | AHCI | 표준 모드 |

## 1.2 설치 시 파티션 설정

Installation type 화면에서 **"Something else"** 선택 후 수동 파티션:

| 파티션 | 크기 | Type | Location | Use as | Mount |
|--------|------|------|----------|--------|-------|
| p1 | 512MB | Primary | Beginning | EFI System Partition | - |
| p2 | 1GB | Primary | Beginning | ext4 journaling | /boot |
| p3 | 32GB | Primary | Beginning | swap area | - |
| p4 | 나머지 | Primary | Beginning | ext4 journaling | / |

> ⚠️ 2TB 디스크(nvme1n1)는 설치 중 건드리지 않음

---

# Part 2: NVIDIA 드라이버 설치 ✅ 완료

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 권장 드라이버 확인
ubuntu-drivers devices

# 자동 설치
sudo ubuntu-drivers autoinstall

# 재부팅
sudo reboot

# 확인
nvidia-smi
```

### HDMI 외부 모니터 연결 시

드라이버 설치 후 외부 모니터 사용 가능. 색상 문제 발생 시:

```bash
# NVIDIA 설정 열기
nvidia-settings
```

- X Server Display Configuration → 모니터 선택 → Advanced
- Color Range를 **Full** 또는 **Limited** 전환해보기

---

# Part 3: 디스크 구성 (2TB 데이터 디스크)

## 3.1 현재 디스크 확인

```bash
lsblk
```

예상 출력:
```
NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
nvme0n1     259:0    0   3.6T  0 disk 
├─nvme0n1p1 259:1    0   487M  0 part /boot/efi
├─nvme0n1p2 259:2    0   977M  0 part /boot
├─nvme0n1p3 259:3    0  29.8G  0 part [SWAP]
└─nvme0n1p4 259:4    0   3.6T  0 part /
nvme1n1     259:5    0   1.9T  0 disk            ← 2TB 데이터 디스크
```

## 3.2 2TB 디스크 파티션 생성

```bash
# 기존 파티션 삭제 및 GPT 생성
sudo parted /dev/nvme1n1 --script mklabel gpt
sudo parted /dev/nvme1n1 --script mkpart primary ext4 0% 100%

# 확인
lsblk /dev/nvme1n1
```

## 3.3 파일시스템 생성

```bash
# ext4 포맷 (라벨: Workspace)
sudo mkfs.ext4 -L "Workspace" /dev/nvme1n1p1
```

## 3.4 마운트

```bash
# 마운트 포인트 생성
sudo mkdir -p /mnt/workspace

# 마운트
sudo mount /dev/nvme1n1p1 /mnt/workspace

# 소유권 설정
sudo chown -R $USER:$USER /mnt/workspace

# 확인
df -h /mnt/workspace
```

## 3.5 자동 마운트 설정 (fstab)

```bash
# fstab 백업
sudo cp /etc/fstab /etc/fstab.backup

# fstab에 추가
echo 'LABEL=Workspace  /mnt/workspace  ext4  defaults,noatime  0  2' | sudo tee -a /etc/fstab

# 검증 (오류 없어야 함)
sudo mount -a
echo $?  # 0이면 성공
```

## 3.6 디렉토리 구조 생성

```bash
# 데이터 디스크 (2TB)
mkdir -p /mnt/workspace/{datasets,models,cache,outputs}
mkdir -p /mnt/workspace/datasets/{alphafold-db,alpamayo}
mkdir -p /mnt/workspace/cache/{pip,huggingface,torch}

# 홈 디렉토리 (4TB 시스템 디스크)
mkdir -p ~/projects/{alphafold,alpamayo,isaaclab}
mkdir -p ~/projects/alphafold/{inputs,outputs}
mkdir -p ~/projects/alpamayo/{configs,outputs}
mkdir -p ~/projects/isaaclab

# 심볼릭 링크 (편의용)
ln -s /mnt/workspace ~/workspace
```

## 3.7 확인

```bash
# 디스크 상태
df -h / /mnt/workspace

# 구조 확인
ls -la ~/projects/
ls -la /mnt/workspace/
```

---

# Part 4: 기반 시스템 설정

## 4.1 필수 패키지 설치

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential git curl wget unzip \
    software-properties-common apt-transport-https \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    vulkan-tools mesa-vulkan-drivers
```

## 4.2 CUDA Toolkit 12.4 설치

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

## 4.3 cuDNN 설치

```bash
sudo apt install libcudnn9-cuda-12 libcudnn9-dev-cuda-12 -y
```

## 4.4 Docker + NVIDIA Container Toolkit

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

## 4.5 Miniconda 설치

```bash
# 다운로드
mkdir -p ~/downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ~/downloads/miniconda.sh

# 설치 (홈 디렉토리에 설치)
bash ~/downloads/miniconda.sh -b -p ~/miniconda3

# PATH 설정
cat >> ~/.bashrc << 'EOF'

# Miniconda
export PATH="$HOME/miniconda3/bin:$PATH"
EOF

source ~/.bashrc
conda init bash
source ~/.bashrc

# 업데이트
conda update -n base -c defaults conda -y
```

## 4.6 환경 변수 설정

```bash
cat >> ~/.bashrc << 'EOF'

# ============================================
# Project Directories
# ============================================
export WORKSPACE_DIR="/mnt/workspace"
export PROJECTS_DIR="$HOME/projects"

# Cache (2TB 디스크 사용)
export PIP_CACHE_DIR="/mnt/workspace/cache/pip"
export HF_HOME="/mnt/workspace/cache/huggingface"
export TORCH_HOME="/mnt/workspace/cache/torch"
export TMPDIR="/mnt/workspace/tmp"

# Project-specific
export ALPHAFOLD_DATA_DIR="/mnt/workspace/datasets/alphafold-db"
export ALPAMAYO_DATASET_DIR="/mnt/workspace/datasets/alpamayo"
EOF

source ~/.bashrc

# tmp 디렉토리 생성
mkdir -p /mnt/workspace/tmp
```

---

# Part 5: Isaac Lab + Isaac Sim 설치

> Isaac Lab을 설치하면 Isaac Sim이 함께 설치됨.
> Isaac Gym은 deprecated → Isaac Lab에 통합됨.

## 5.1 Conda 환경 생성

```bash
# Python 3.11 필수 (Isaac Sim 5.x 요구사항)
conda create -n isaaclab python=3.11 -y
conda activate isaaclab
```

## 5.2 Isaac Sim 설치 (pip 방식)

```bash
# pip 업그레이드
pip install --upgrade pip

# Isaac Sim 설치 (~15GB 다운로드)
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

# 설치 확인
python -c "import isaacsim; print('Isaac Sim OK')"
```

## 5.3 Isaac Lab 설치

```bash
cd ~/projects/isaaclab

# Isaac Lab 클론
git clone https://github.com/isaac-sim/IsaacLab.git repo
cd repo

# 설치
./isaaclab.sh --install

# 또는 수동 설치
pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
```

## 5.4 RL 라이브러리 설치

```bash
# RSL-RL (가장 많이 사용)
pip install rsl-rl

# rl_games
pip install rl-games

# Stable Baselines 3
pip install stable-baselines3
```

## 5.5 설치 확인

```bash
conda activate isaaclab
cd ~/projects/isaaclab/repo

# 기본 테스트 (headless 모드)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

# 강화학습 테스트 (Cartpole)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Cartpole-v0 --headless --num_envs=64

conda deactivate
```

---

# Part 6: AlphaFold/ColabFold 설치

> LocalColabFold 사용 (전체 DB 2.5TB 불필요, ~30GB로 실행 가능)

## 6.1 Conda 환경 생성

```bash
conda create -n alphafold python=3.10 -y
conda activate alphafold
```

## 6.2 LocalColabFold 설치

```bash
cd ~/projects/alphafold

# 설치 스크립트 다운로드
wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh

# 설치 실행 (~10-15분)
bash install_colabbatch_linux.sh

# PATH 추가
cat >> ~/.bashrc << 'EOF'

# LocalColabFold
export PATH="$HOME/projects/alphafold/localcolabfold/colabfold-conda/bin:$PATH"
EOF

source ~/.bashrc
```

## 6.3 모델 파라미터 다운로드

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

## 6.4 설치 확인

```bash
conda activate alphafold

# 버전 확인
colabfold_batch --help

# 테스트 예측
echo ">test_protein
MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTG" > ~/projects/alphafold/inputs/test.fasta

colabfold_batch \
    ~/projects/alphafold/inputs/test.fasta \
    ~/projects/alphafold/outputs/test_result

# 결과 확인
ls ~/projects/alphafold/outputs/test_result/

conda deactivate
```

---

# Part 7: Alpamayo-R1 설치

## 7.1 Conda 환경 생성

```bash
conda create -n alpamayo python=3.10 -y
conda activate alpamayo
```

## 7.2 PyTorch 설치

```bash
# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 확인
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 7.3 Alpamayo 클론 및 설치

```bash
cd ~/projects/alpamayo

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

## 7.4 HuggingFace 인증

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

## 7.5 테스트 추론

```bash
conda activate alpamayo
cd ~/projects/alpamayo/repo
source ar1_venv/bin/activate

# 테스트 (모델 자동 다운로드 ~22GB)
python src/alpamayo_r1/test_inference.py

deactivate
conda deactivate
```

---

# Part 8: 환경 관리 및 전환

## 8.1 활성화 스크립트 생성

```bash
mkdir -p ~/bin

# Isaac Lab 환경
cat > ~/bin/activate-isaaclab << 'EOF'
#!/bin/bash
echo "🤖 Activating Isaac Lab environment..."
source ~/miniconda3/bin/activate isaaclab
cd ~/projects/isaaclab/repo
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
source ~/miniconda3/bin/activate alphafold
cd ~/projects/alphafold
echo ""
echo "Environment: alphafold (Python 3.10)"
echo "LocalColabFold ready"
echo ""
echo "Usage:"
echo "  colabfold_batch input.fasta output_dir/"
EOF

# Alpamayo 환경
cat > ~/bin/activate-alpamayo << 'EOF'
#!/bin/bash
echo "🚗 Activating Alpamayo environment..."
source ~/miniconda3/bin/activate alpamayo
cd ~/projects/alpamayo/repo
source ar1_venv/bin/activate
echo ""
echo "Environment: alpamayo (Python 3.10 + uv venv)"
echo "Alpamayo-R1 ready"
echo ""
echo "Usage:"
echo "  python src/alpamayo_r1/test_inference.py"
EOF

# 실행 권한
chmod +x ~/bin/activate-*

# PATH 추가
cat >> ~/.bashrc << 'EOF'

# Custom scripts
export PATH="$HOME/bin:$PATH"
EOF

source ~/.bashrc
```

## 8.2 사용법

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

## 8.3 유틸리티 스크립트

### 디스크 상태 확인

```bash
cat > ~/bin/disk-status << 'EOF'
#!/bin/bash
echo "========================================"
echo "Disk Status - $(date)"
echo "========================================"
echo ""
echo "=== System (4TB) ==="
df -h /
echo ""
echo "=== Workspace (2TB) ==="
df -h /mnt/workspace
echo ""
echo "=== Workspace Usage ==="
du -sh /mnt/workspace/*/ 2>/dev/null | sort -hr
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv
EOF

chmod +x ~/bin/disk-status
```

---

# Part 9: 전체 검증

## 9.1 검증 스크립트 생성

```bash
cat > ~/bin/verify-all << 'EOF'
#!/bin/bash
echo "=========================================="
echo "Complete System Verification"
echo "$(date)"
echo "=========================================="

echo -e "\n[1/7] System Info"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"

echo -e "\n[2/7] Disk Mounts"
echo "System (4TB): $(df -h / | tail -1 | awk '{print $4}') available"
if mountpoint -q /mnt/workspace; then
    echo "Workspace (2TB): $(df -h /mnt/workspace | tail -1 | awk '{print $4}') available"
    echo "✓ Disks OK"
else
    echo "✗ Workspace not mounted!"
fi

echo -e "\n[3/7] NVIDIA & CUDA"
if nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | cut -d',' -f1)"
    echo "✓ NVIDIA stack OK"
else
    echo "✗ NVIDIA driver issue"
fi

echo -e "\n[4/7] Docker GPU"
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "✓ Docker GPU OK"
else
    echo "✗ Docker GPU issue"
fi

echo -e "\n[5/7] Isaac Lab"
source ~/miniconda3/bin/activate isaaclab 2>/dev/null
if python -c "import isaacsim" 2>/dev/null; then
    echo "✓ Isaac Sim OK"
    echo "✓ Isaac Lab OK"
else
    echo "✗ Isaac Lab not installed or issue"
fi
conda deactivate 2>/dev/null

echo -e "\n[6/7] AlphaFold/ColabFold"
source ~/miniconda3/bin/activate alphafold 2>/dev/null
if command -v colabfold_batch &>/dev/null; then
    echo "✓ ColabFold OK"
else
    echo "✗ ColabFold not installed"
fi
conda deactivate 2>/dev/null

echo -e "\n[7/7] Alpamayo"
source ~/miniconda3/bin/activate alpamayo 2>/dev/null
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

## 9.2 검증 실행

```bash
verify-all
```

예상 출력:
```
==========================================
Complete System Verification
==========================================

[1/7] System Info
OS: Ubuntu 22.04.x LTS
Kernel: 6.x.x-generic

[2/7] Disk Mounts
System (4TB): 3.xT available
Workspace (2TB): 1.xT available
✓ Disks OK

[3/7] NVIDIA & CUDA
NVIDIA GeForce RTX 5090 Laptop GPU, 24576 MiB, 560.xx
CUDA: 12.4
✓ NVIDIA stack OK

[4/7] Docker GPU
✓ Docker GPU OK

[5/7] Isaac Lab
✓ Isaac Sim OK
✓ Isaac Lab OK

[6/7] AlphaFold/ColabFold
✓ ColabFold OK

[7/7] Alpamayo
✓ PyTorch + CUDA OK
✓ Alpamayo environment OK

==========================================
Verification Complete
==========================================
```

---

# Part 10: 문제 해결

## 디스크 마운트 안 됨

```bash
# fstab 확인
cat /etc/fstab

# 수동 마운트
sudo mount -a

# 라벨 확인
sudo blkid | grep Workspace
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

## HDMI 외부 모니터 색상 문제

```bash
# NVIDIA 설정
nvidia-settings
# Display Configuration → Advanced → Color Range 조정

# 또는 xrandr로 (HDMI-1-0을 실제 이름으로 변경)
xrandr --output HDMI-1-0 --set "Broadcast RGB" "Full"
```

## Isaac Sim 실행 안 됨

```bash
# Vulkan 확인
vulkaninfo | head -20

# 헤드리스 모드로 테스트
conda activate isaaclab
python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('OK'); app.close()"
```

## 메모리 부족 (OOM)

```bash
# Isaac Lab: 환경 수 줄이기
./isaaclab.sh -p train.py --task=Isaac-Ant-v0 --num_envs=64

# Alpamayo: 배치 크기 줄이기
# test_inference.py에서 num_traj_samples=1로 설정
```

---

# 최종 디렉토리 구조

```
/home/$USER (4TB 시스템 디스크)
├── miniconda3/              # Conda
├── projects/
│   ├── alphafold/
│   │   ├── localcolabfold/  # ColabFold 설치
│   │   ├── inputs/
│   │   └── outputs/
│   ├── alpamayo/
│   │   └── repo/            # Alpamayo-R1
│   └── isaaclab/
│       └── repo/            # Isaac Lab + Isaac Sim
├── bin/
│   ├── activate-isaaclab
│   ├── activate-alphafold
│   ├── activate-alpamayo
│   ├── disk-status
│   └── verify-all
└── workspace -> /mnt/workspace (심볼릭 링크)

/mnt/workspace (2TB 데이터 디스크)
├── datasets/
│   ├── alphafold-db/        # AlphaFold DB (선택)
│   └── alpamayo/            # Physical AI Dataset
├── models/
│   └── colabfold/           # AlphaFold 모델 파라미터
├── cache/
│   ├── pip/
│   ├── huggingface/         # Alpamayo 모델 캐시
│   └── torch/
├── outputs/
└── tmp/
```

---

# 설치 진행 체크리스트

- [x] Ubuntu 22.04 설치 (4TB 디스크)
- [x] NVIDIA 드라이버 설치
- [ ] 2TB 디스크 구성 (Part 3)
- [ ] CUDA, cuDNN, Docker 설치 (Part 4)
- [ ] Miniconda 설치 (Part 4.5)
- [ ] 환경 변수 설정 (Part 4.6)
- [ ] Isaac Lab 설치 (Part 5)
- [ ] AlphaFold 설치 (Part 6)
- [ ] Alpamayo 설치 (Part 7)
- [ ] 활성화 스크립트 생성 (Part 8)
- [ ] 전체 검증 (Part 9)

---

# 용량 예상

| 항목 | 용량 | 위치 |
|------|------|------|
| Ubuntu + 프로그램 | ~50GB | 4TB (시스템) |
| Miniconda + 환경들 | ~30GB | 4TB (홈) |
| 프로젝트 코드 | ~5GB | 4TB (홈) |
| Isaac Sim | ~15GB | 4TB (conda env) |
| **4TB 사용량** | **~100GB** | 여유: ~3.5TB |
| ColabFold 모델 | ~15GB | 2TB |
| Alpamayo 모델 | ~25GB | 2TB (cache) |
| 캐시 (pip, HF) | ~20GB | 2TB |
| 실험 결과물 | 가변 | 2TB |
| **2TB 사용량** | **~60GB+** | 여유: ~1.8TB |

---

*가이드 작성일: 2026-01-17*
*대상: ASUS ROG Strix SCAR 16 G635LX-RW047W*
*RTX 5090 Laptop (24GB) | 4TB (시스템) + 2TB (데이터) | Ubuntu 22.04 LTS*
