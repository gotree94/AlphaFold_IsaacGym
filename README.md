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
# Anaconda 설치
## Anaconda Full Version install

```bash
# 1. Anaconda 다운로드
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

# 2. 설치
bash Anaconda3-2024.10-1-Linux-x86_64.sh

# 설치 중:
# - Enter 눌러서 라이선스 읽기
# - 'yes' 입력하여 라이선스 동의
# - Enter로 기본 설치 경로 확인 (/home/gotree94/anaconda3)
# - 'yes' 입력하여 conda init 활성화

# 3. 설치 완료 후
source ~/.bashrc

# 4. 확인
conda --version
python --version
```
