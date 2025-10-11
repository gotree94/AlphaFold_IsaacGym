# AlphaFold_IsaacGym

## Open Kernel Module 드라이버 설치
**1. 현재 드라이버 제거**
```
bashsudo apt remove --purge nvidia-driver-580 -y
sudo apt autoremove -y
```

**2. Open Kernel Module 드라이버 설치**
```
bash# Open 버전 드라이버 설치
sudo apt update
sudo apt install nvidia-driver-580-open nvidia-utils-580 -y
```

**3. 재부팅**
```
bashsudo reboot
```

**4. 재접속 후 확인**
```
bashnvidia-smi
```

```
gotree94@G635LX:~$ nvidia-smi
Sat Oct 11 14:26:31 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090 ...    Off |   00000000:02:00.0  On |                  N/A |
| N/A   45C    P8              8W /   95W |     114MiB /  24463MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1951      G   /usr/lib/xorg/Xorg                       64MiB |
|    0   N/A  N/A            2111      G   /usr/bin/gnome-shell                     13MiB |
+-----------------------------------------------------------------------------------------+
```

## 드라이버 모듈 확인

```
lsmod | grep nvidia
```

```
nvidia_uvm           2170880  0
nvidia_drm            139264  4
nvidia_modeset       1744896  6 nvidia_drm
nvidia              14360576  73 nvidia_uvm,nvidia_modeset
nvidia_wmi_ec_backlight    12288  0
ecc                    45056  2 ecdh_generic,nvidia
video                  77824  6 nvidia_wmi_ec_backlight,asus_wmi,asus_nb_wmi,xe,i915,nvidia_modeset
wmi                    28672  5 video,nvidia_wmi_ec_backlight,asus_wmi,wmi_bmof,mfd_aaeon

```
# Miniconda 설치
```bash
# Miniconda 다운로드 및 설치
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 설치 중 yes, 기본 경로 사용, yes (conda init)
# 설치 완료 후
source ~/.bashrc

# conda 확인
conda --version
```
## CUDA 환경 변수 설정
```bash
# .bashrc에 CUDA 경로 추가
echo 'export PATH=/usr/lib/nvidia-cuda-toolkit/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

# AlphaFold

## 1. AlphaFold 설치
```bash
# AlphaFold 환경 생성
conda create -n alphafold python=3.10 -y
conda activate alphafold

# 필수 패키지 설치
conda install -c conda-forge openmm==8.0.0 pdbfixer -y
pip install --upgrade pip

# AlphaFold (ColabFold 버전 - 더 쉬움)
pip install colabfold[alphafold]

# JAX GPU 버전 설치 (CUDA 12 지원)
pip install --upgrade "jax[cuda12]"
```

## 2. Isaac Gym 설치
** Isaac Gym은 NVIDIA에서 직접 다운로드해야 합니다.**
** Isaac Gym 다운로드 방법:**

** 1.NVIDIA 계정으로 다운로드:**
   * https://developer.nvidia.com/isaac-gym
   * "Join now" 클릭하여 NVIDIA Developer 계정 생성/로그인
   * "Download" 버튼 클릭
   * IsaacGym_Preview_4_Package.tar.gz 다운로드
     
** 2.MobaXterm으로 파일 전송:**
   * MobaXterm 왼쪽 사이드바에서 파일 업로드
   * 또는 SFTP로 전송

** 3.서버에서 설치:**

```bash
# Isaac Gym 환경 생성
conda create -n isaacgym python=3.8 -y
conda activate isaacgym

# 업로드한 파일 압축 해제
cd ~
tar -xzf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python

# Isaac Gym 설치
pip install -e .

# PyTorch 설치 (CUDA 12 지원)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 테스트
cd examples
python joint_monkey.py
```


