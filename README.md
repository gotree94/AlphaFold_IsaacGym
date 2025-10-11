# AlphaFold_IsaacGym

✅ Isaac Gym      : 완벽 (최고 성능)
✅ AlphaFold      : 완벽 (대형 단백질 처리)
✅ MONAI          : 완벽 (3D 의료 영상)
✅ Clara          : 완벽 (Enterprise 워크로드)
⚠️ BioNeMo        : 양호 (중형 모델까지, 대형은 양자화 필요)


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
## Anaconda 설치 후 AlphaFold 설정

```bash
conda deactivate

# AlphaFold 환경 생성
conda create -n alphafold python=3.10 -y
conda activate alphafold

# 필수 패키지 (Anaconda에 이미 많이 포함됨)
conda install numpy scipy matplotlib -y
pip install --upgrade pip

# JAX GPU 버전 설치
pip install --upgrade "jax[cuda12]"

# ColabFold 설치
pip install colabfold[alphafold]
```

```
mkdir ~/alphafold_test
cd ~/alphafold_test
```


```
cat > test.fasta
```

   * (‘>Test’ 는 단백질 이름 헤더,
   * 그 아래는 아미노산 서열입니다.)
   * 입력이 끝나면 Ctrl + D 를 눌러 저장하고 종료합니다.
     
```
>Test
ACDEFGHIKLMNPQRSTVWY
```

```
echo -e ">Test\nACDEFGHIKLMNPQRSTVWY" > test.fasta
```

```
ls
cat test.fasta
```

   * **test_run.py**
```
from colabfold import run
import os

# 입력 파일 (FASTA)
fasta_path = "test.fasta"

# 출력 폴더
out_dir = "test_output"
os.makedirs(out_dir, exist_ok=True)

# 예측 옵션 — 기본 설정
job = run(
    input_fasta=fasta_path,
    output_dir=out_dir,
    model_type="auto",       # 또는 "monomer", "multimer"
    num_models=1,
    use_templates=False,
    is_multimer=False,
)

print("Job result:", job)

```

```
python test_run.py
```








## Isaac Gym 설정
```bash
conda deactivate

# Isaac Gym 환경
conda create -n isaacgym python=3.8 -y
conda activate isaacgym

# PyTorch 설치
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

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


