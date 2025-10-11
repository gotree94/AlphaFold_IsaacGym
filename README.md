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

```

CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels. Please accept or remove them before proceeding:
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r

To accept these channels' Terms of Service, run the following commands:
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

For information on safely removing channels from your conda configuration,
please see the official documentation:

    https://www.anaconda.com/docs/tools/working-with-conda/channels


CondaError: Run 'conda init' before 'conda activate'


CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels. Please accept or remove them before proceeding:
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r

To accept these channels' Terms of Service, run the following commands:
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

For information on safely removing channels from your conda configuration,
please see the official documentation:

    https://www.anaconda.com/docs/tools/working-with-conda/channels

Requirement already satisfied: pip in ./miniconda3/lib/python3.13/site-packages (25.1)
Collecting pip
  Downloading pip-25.2-py3-none-any.whl.metadata (4.7 kB)
Downloading pip-25.2-py3-none-any.whl (1.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 9.5 MB/s eta 0:00:00
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 25.1
    Uninstalling pip-25.1:
      Successfully uninstalled pip-25.1
Successfully installed pip-25.2
Collecting colabfold[alphafold]
  Downloading colabfold-1.5.4-py3-none-any.whl.metadata (19 kB)
Collecting absl-py<2.0.0,>=1.0.0 (from colabfold[alphafold])
  Downloading absl_py-1.4.0-py3-none-any.whl.metadata (2.3 kB)
Collecting alphafold-colabfold==v2.3.6 (from colabfold[alphafold])
  Downloading alphafold_colabfold-2.3.6-py3-none-any.whl.metadata (1.3 kB)
Collecting appdirs<2.0.0,>=1.4.4 (from colabfold[alphafold])
  Downloading appdirs-1.4.4-py2.py3-none-any.whl.metadata (9.0 kB)
Collecting biopython<=1.82 (from colabfold[alphafold])
  Downloading biopython-1.82.tar.gz (19.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 19.4/19.4 MB 10.0 MB/s  0:00:01
  Preparing metadata (setup.py) ... done
Collecting dm-haiku<0.0.12,>=0.0.11 (from colabfold[alphafold])
  Downloading dm_haiku-0.0.11-py3-none-any.whl.metadata (19 kB)
Collecting importlib-metadata<5.0.0,>=4.8.2 (from colabfold[alphafold])
  Downloading importlib_metadata-4.13.0-py3-none-any.whl.metadata (4.9 kB)
Collecting jax<0.5.0,>=0.4.20 (from colabfold[alphafold])
  Downloading jax-0.4.38-py3-none-any.whl.metadata (22 kB)
Collecting matplotlib<4.0.0,>=3.2.2 (from colabfold[alphafold])
  Downloading matplotlib-3.10.7-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
Collecting numpy<2.0.0,>=1.21.6 (from colabfold[alphafold])
  Downloading numpy-1.26.4.tar.gz (15.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 15.8/15.8 MB 11.4 MB/s  0:00:01
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
Collecting pandas<2.0.0,>=1.3.4 (from colabfold[alphafold])
  Downloading pandas-1.5.3.tar.gz (5.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.2/5.2 MB 10.7 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting py3Dmol<3.0.0,>=2.0.1 (from colabfold[alphafold])
  Downloading py3dmol-2.5.3-py2.py3-none-any.whl.metadata (2.1 kB)
Requirement already satisfied: requests<3.0.0,>=2.26.0 in ./miniconda3/lib/python3.13/site-packages (from colabfold[alphafold]) (2.32.4)
Collecting tensorflow-cpu<3.0.0,>=2.12.1 (from colabfold[alphafold])
  Downloading tensorflow_cpu-2.20.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.5 kB)
Requirement already satisfied: tqdm<5.0.0,>=4.62.2 in ./miniconda3/lib/python3.13/site-packages (from colabfold[alphafold]) (4.67.1)
Collecting chex (from alphafold-colabfold==v2.3.6->colabfold[alphafold])
  Downloading chex-0.1.91-py3-none-any.whl.metadata (18 kB)
Collecting dm-tree (from alphafold-colabfold==v2.3.6->colabfold[alphafold])
  Downloading dm_tree-0.1.9-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.4 kB)
Collecting immutabledict (from alphafold-colabfold==v2.3.6->colabfold[alphafold])
  Downloading immutabledict-4.2.1-py3-none-any.whl.metadata (3.5 kB)
Collecting ml-collections (from alphafold-colabfold==v2.3.6->colabfold[alphafold])
  Downloading ml_collections-1.1.0-py3-none-any.whl.metadata (22 kB)
Collecting scipy (from alphafold-colabfold==v2.3.6->colabfold[alphafold])
  Downloading scipy-1.16.2-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (62 kB)
Collecting jmp>=0.0.2 (from dm-haiku<0.0.12,>=0.0.11->colabfold[alphafold])
  Downloading jmp-0.0.4-py3-none-any.whl.metadata (8.9 kB)
Collecting tabulate>=0.8.9 (from dm-haiku<0.0.12,>=0.0.11->colabfold[alphafold])
  Downloading tabulate-0.9.0-py3-none-any.whl.metadata (34 kB)
Collecting flax>=0.7.1 (from dm-haiku<0.0.12,>=0.0.11->colabfold[alphafold])
  Downloading flax-0.12.0-py3-none-any.whl.metadata (11 kB)
Collecting zipp>=0.5 (from importlib-metadata<5.0.0,>=4.8.2->colabfold[alphafold])
  Downloading zipp-3.23.0-py3-none-any.whl.metadata (3.6 kB)
Collecting jaxlib<=0.4.38,>=0.4.38 (from jax<0.5.0,>=0.4.20->colabfold[alphafold])
  Downloading jaxlib-0.4.38-cp313-cp313-manylinux2014_x86_64.whl.metadata (1.0 kB)
Collecting ml_dtypes>=0.4.0 (from jax<0.5.0,>=0.4.20->colabfold[alphafold])
  Downloading ml_dtypes-0.5.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.9 kB)
Collecting opt_einsum (from jax<0.5.0,>=0.4.20->colabfold[alphafold])
  Downloading opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)
Collecting contourpy>=1.0.1 (from matplotlib<4.0.0,>=3.2.2->colabfold[alphafold])
  Downloading contourpy-1.3.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib<4.0.0,>=3.2.2->colabfold[alphafold])
  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib<4.0.0,>=3.2.2->colabfold[alphafold])
  Downloading fonttools-4.60.1-cp313-cp313-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (112 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib<4.0.0,>=3.2.2->colabfold[alphafold])
  Downloading kiwisolver-1.4.9-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.3 kB)
Requirement already satisfied: packaging>=20.0 in ./miniconda3/lib/python3.13/site-packages (from matplotlib<4.0.0,>=3.2.2->colabfold[alphafold]) (25.0)
Collecting pillow>=8 (from matplotlib<4.0.0,>=3.2.2->colabfold[alphafold])
  Downloading pillow-11.3.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
Collecting pyparsing>=3 (from matplotlib<4.0.0,>=3.2.2->colabfold[alphafold])
  Downloading pyparsing-3.2.5-py3-none-any.whl.metadata (5.0 kB)
Collecting python-dateutil>=2.7 (from matplotlib<4.0.0,>=3.2.2->colabfold[alphafold])
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting pytz>=2020.1 (from pandas<2.0.0,>=1.3.4->colabfold[alphafold])
  Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Requirement already satisfied: charset_normalizer<4,>=2 in ./miniconda3/lib/python3.13/site-packages (from requests<3.0.0,>=2.26.0->colabfold[alphafold]) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in ./miniconda3/lib/python3.13/site-packages (from requests<3.0.0,>=2.26.0->colabfold[alphafold]) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in ./miniconda3/lib/python3.13/site-packages (from requests<3.0.0,>=2.26.0->colabfold[alphafold]) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in ./miniconda3/lib/python3.13/site-packages (from requests<3.0.0,>=2.26.0->colabfold[alphafold]) (2025.8.3)
Collecting astunparse>=1.6.0 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting flatbuffers>=24.3.25 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading flatbuffers-25.9.23-py2.py3-none-any.whl.metadata (875 bytes)
Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading gast-0.6.0-py3-none-any.whl.metadata (1.3 kB)
Collecting google_pasta>=0.1.1 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting libclang>=13.0.0 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl.metadata (5.2 kB)
Collecting protobuf>=5.28.0 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading protobuf-6.32.1-cp39-abi3-manylinux2014_x86_64.whl.metadata (593 bytes)
Requirement already satisfied: setuptools in ./miniconda3/lib/python3.13/site-packages (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold]) (78.1.1)
Collecting six>=1.12.0 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting termcolor>=1.1.0 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading termcolor-3.1.0-py3-none-any.whl.metadata (6.4 kB)
Requirement already satisfied: typing_extensions>=3.6.6 in ./miniconda3/lib/python3.13/site-packages (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold]) (4.12.2)
Collecting wrapt>=1.11.0 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading wrapt-1.17.3-cp313-cp313-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (6.4 kB)
Collecting grpcio<2.0,>=1.24.3 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading grpcio-1.75.1-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (3.7 kB)
Collecting tensorboard~=2.20.0 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading tensorboard-2.20.0-py3-none-any.whl.metadata (1.8 kB)
Collecting keras>=3.10.0 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading keras-3.11.3-py3-none-any.whl.metadata (5.9 kB)
Collecting h5py>=3.11.0 (from tensorflow-cpu<3.0.0,>=2.12.1->colabfold[alphafold])
  Downloading h5py-3.14.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
INFO: pip is looking at multiple versions of ml-dtypes to determine which version is compatible with other requirements. This could take a while.
Collecting ml_dtypes>=0.4.0 (from jax<0.5.0,>=0.4.20->colabfold[alphafold])
  Downloading ml_dtypes-0.5.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (21 kB)
Collecting pandas<2.0.0,>=1.3.4 (from colabfold[alphafold])
  Downloading pandas-1.5.2.tar.gz (5.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.2/5.2 MB 8.5 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading pandas-1.5.1.tar.gz (5.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.2/5.2 MB 8.4 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading pandas-1.5.0.tar.gz (5.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.2/5.2 MB 11.1 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
INFO: pip is still looking at multiple versions of ml-dtypes to determine which version is compatible with other requirements. This could take a while.
  Downloading pandas-1.4.4.tar.gz (4.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 11.1 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading pandas-1.4.3.tar.gz (4.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 11.1 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading pandas-1.4.2.tar.gz (4.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 11.0 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
  Downloading pandas-1.4.1.tar.gz (4.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 9.6 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading pandas-1.4.0.tar.gz (4.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 11.3 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading pandas-1.3.5.tar.gz (4.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.7/4.7 MB 10.5 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading pandas-1.3.4.tar.gz (4.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.7/4.7 MB 4.7 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting matplotlib<4.0.0,>=3.2.2 (from colabfold[alphafold])
  Downloading matplotlib-3.10.6-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
  Downloading matplotlib-3.10.5-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
  Downloading matplotlib-3.10.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
  Downloading matplotlib-3.10.1-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
  Downloading matplotlib-3.10.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
  Downloading matplotlib-3.9.4-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
  Downloading matplotlib-3.9.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
  Downloading matplotlib-3.9.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
  Downloading matplotlib-3.9.1.post1.tar.gz (36.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 36.1/36.1 MB 11.2 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.9.0.tar.gz (36.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 36.1/36.1 MB 7.4 MB/s  0:00:04
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.8.4.tar.gz (35.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.9/35.9 MB 11.5 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.8.3.tar.gz (35.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.9/35.9 MB 9.1 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.8.2.tar.gz (35.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.9/35.9 MB 11.4 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.8.1.tar.gz (35.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.9/35.9 MB 9.3 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.8.0.tar.gz (35.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.9/35.9 MB 11.3 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.7.5.tar.gz (38.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.1/38.1 MB 11.3 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.7.4.tar.gz (38.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.1/38.1 MB 9.8 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.7.3.tar.gz (38.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.1/38.1 MB 10.5 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.7.2.tar.gz (38.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.1/38.1 MB 11.1 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting pyparsing<3.1,>=2.3.1 (from matplotlib<4.0.0,>=3.2.2->colabfold[alphafold])
  Downloading pyparsing-3.0.9-py3-none-any.whl.metadata (4.2 kB)
Collecting matplotlib<4.0.0,>=3.2.2 (from colabfold[alphafold])
  Downloading matplotlib-3.7.1.tar.gz (38.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.0/38.0 MB 10.4 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.7.0.tar.gz (36.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 36.3/36.3 MB 11.4 MB/s  0:00:03
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Downloading matplotlib-3.6.3.tar.gz (35.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.9/35.9 MB 11.3 MB/s  0:00:03
  Preparing metadata (setup.py) ... done
  Downloading matplotlib-3.6.2.tar.gz (35.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.8/35.8 MB 10.6 MB/s  0:00:03
  Preparing metadata (setup.py) ... done
  Downloading matplotlib-3.6.1.tar.gz (35.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.8/35.8 MB 11.1 MB/s  0:00:03
  Preparing metadata (setup.py) ... done
  Downloading matplotlib-3.6.0.tar.gz (35.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.7/35.7 MB 11.2 MB/s  0:00:03
  Preparing metadata (setup.py) ... done
  Downloading matplotlib-3.5.3.tar.gz (35.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.2/35.2 MB 11.3 MB/s  0:00:03
  Preparing metadata (setup.py) ... done
  Downloading matplotlib-3.5.2.tar.gz (35.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.2/35.2 MB 11.2 MB/s  0:00:03
  Preparing metadata (setup.py) ... done
  Downloading matplotlib-3.5.1.tar.gz (35.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.3/35.3 MB 11.5 MB/s  0:00:03
  Preparing metadata (setup.py) ... done
  Downloading matplotlib-3.5.0.tar.gz (35.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.0/35.0 MB 11.4 MB/s  0:00:03
  Preparing metadata (setup.py) ... done
  Downloading matplotlib-3.4.3.tar.gz (37.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.9/37.9 MB 11.4 MB/s  0:00:03
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [35 lines of output]
      /tmp/pip-install-nl0zm80o/matplotlib_e7dc6ffad0d54fa3ba16dbf73ad505e3/setup.py:34: SetuptoolsDeprecationWarning: The test command is disabled and references to it are deprecated.
      !!

              ********************************************************************************
              Please remove any references to `setuptools.command.test` in all supported versions of the affected package.

              This deprecation is overdue, please update your project and remove deprecated
              calls to avoid build errors in the future.
              ********************************************************************************

      !!
        from setuptools.command.test import test as TestCommand
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
          exec(compile('''
          ~~~~^^^^^^^^^^^^
          # This is <pip-setuptools-caller> -- a caller that pip uses to run setup.py
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          ...<32 lines>...
          exec(compile(setup_py_code, filename, "exec"))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          ''' % ('/tmp/pip-install-nl0zm80o/matplotlib_e7dc6ffad0d54fa3ba16dbf73ad505e3/setup.py',), "<pip-setuptools-caller>", "exec"))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<pip-setuptools-caller>", line 35, in <module>
        File "/tmp/pip-install-nl0zm80o/matplotlib_e7dc6ffad0d54fa3ba16dbf73ad505e3/setup.py", line 54, in <module>
          __version__ = versioneer.get_version()
        File "/tmp/pip-install-nl0zm80o/matplotlib_e7dc6ffad0d54fa3ba16dbf73ad505e3/versioneer.py", line 1410, in get_version
          return get_versions()["version"]
                 ~~~~~~~~~~~~^^
        File "/tmp/pip-install-nl0zm80o/matplotlib_e7dc6ffad0d54fa3ba16dbf73ad505e3/versioneer.py", line 1344, in get_versions
          cfg = get_config_from_root(root)
        File "/tmp/pip-install-nl0zm80o/matplotlib_e7dc6ffad0d54fa3ba16dbf73ad505e3/versioneer.py", line 401, in get_config_from_root
          parser = configparser.SafeConfigParser()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      AttributeError: module 'configparser' has no attribute 'SafeConfigParser'. Did you mean: 'RawConfigParser'?
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
Collecting jax[cuda12]
  Downloading jax-0.7.2-py3-none-any.whl.metadata (13 kB)
Collecting jaxlib<=0.7.2,>=0.7.2 (from jax[cuda12])
  Downloading jaxlib-0.7.2-cp313-cp313-manylinux_2_27_x86_64.whl.metadata (1.3 kB)
Collecting ml_dtypes>=0.5.0 (from jax[cuda12])
  Using cached ml_dtypes-0.5.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.9 kB)
Collecting numpy>=2.0 (from jax[cuda12])
  Using cached numpy-2.3.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting opt_einsum (from jax[cuda12])
  Using cached opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)
Collecting scipy>=1.13 (from jax[cuda12])
  Using cached scipy-1.16.2-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (62 kB)
Collecting jax-cuda12-plugin<=0.7.2,>=0.7.2 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading jax_cuda12_plugin-0.7.2-cp313-cp313-manylinux_2_27_x86_64.whl.metadata (2.0 kB)
Collecting jax-cuda12-pjrt==0.7.2 (from jax-cuda12-plugin<=0.7.2,>=0.7.2->jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading jax_cuda12_pjrt-0.7.2-py3-none-manylinux_2_27_x86_64.whl.metadata (579 bytes)
Collecting nvidia-cublas-cu12>=12.1.3.1 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_cublas_cu12-12.9.1.4-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-cupti-cu12>=12.1.105 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_cuda_cupti_cu12-12.9.79-py3-none-manylinux_2_25_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cuda-nvcc-cu12>=12.6.85 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_cuda_nvcc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-runtime-cu12>=12.1.105 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cudnn-cu12<10.0,>=9.8 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_cudnn_cu12-9.14.0.64-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cufft-cu12>=11.0.2.54 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_cufft_cu12-11.4.1.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusolver-cu12>=11.4.5.107 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_cusolver_cu12-11.7.5.82-py3-none-manylinux_2_27_x86_64.whl.metadata (1.9 kB)
Collecting nvidia-cusparse-cu12>=12.1.0.106 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_cusparse_cu12-12.5.10.65-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nccl-cu12>=2.18.1 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_nccl_cu12-2.28.3-py3-none-manylinux_2_18_x86_64.whl.metadata (2.0 kB)
Collecting nvidia-nvjitlink-cu12>=12.1.105 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_nvjitlink_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-nvrtc-cu12>=12.1.55 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-nvshmem-cu12>=3.2.5 (from jax-cuda12-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda12"->jax[cuda12])
  Downloading nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.1 kB)
Downloading jax-0.7.2-py3-none-any.whl (2.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.8/2.8 MB 9.8 MB/s  0:00:00
Downloading jax_cuda12_plugin-0.7.2-cp313-cp313-manylinux_2_27_x86_64.whl (5.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 10.9 MB/s  0:00:00
Downloading jax_cuda12_pjrt-0.7.2-py3-none-manylinux_2_27_x86_64.whl (132.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 132.7/132.7 MB 11.5 MB/s  0:00:11
Downloading jaxlib-0.7.2-cp313-cp313-manylinux_2_27_x86_64.whl (78.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.2/78.2 MB 11.5 MB/s  0:00:06
Downloading nvidia_cudnn_cu12-9.14.0.64-py3-none-manylinux_2_27_x86_64.whl (647.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 647.1/647.1 MB 11.3 MB/s  0:00:56
Downloading ml_dtypes-0.5.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (4.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 11.0 MB/s  0:00:00
Using cached numpy-2.3.3-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.6 MB)
Downloading nvidia_cublas_cu12-12.9.1.4-py3-none-manylinux_2_27_x86_64.whl (581.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 581.2/581.2 MB 11.1 MB/s  0:00:51
Downloading nvidia_cuda_cupti_cu12-12.9.79-py3-none-manylinux_2_25_x86_64.whl (10.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.8/10.8 MB 10.7 MB/s  0:00:01
Downloading nvidia_cuda_nvcc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (40.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.5/40.5 MB 11.0 MB/s  0:00:03
Downloading nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (89.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 89.6/89.6 MB 11.3 MB/s  0:00:07
Downloading nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.5/3.5 MB 10.8 MB/s  0:00:00
Downloading nvidia_cufft_cu12-11.4.1.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (200.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200.9/200.9 MB 11.4 MB/s  0:00:17
Downloading nvidia_cusolver_cu12-11.7.5.82-py3-none-manylinux_2_27_x86_64.whl (338.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 338.1/338.1 MB 10.0 MB/s  0:00:32
Downloading nvidia_cusparse_cu12-12.5.10.65-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (366.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 366.5/366.5 MB 11.3 MB/s  0:00:32
Downloading nvidia_nccl_cu12-2.28.3-py3-none-manylinux_2_18_x86_64.whl (295.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 295.9/295.9 MB 11.1 MB/s  0:00:26
Downloading nvidia_nvjitlink_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (39.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 39.7/39.7 MB 10.1 MB/s  0:00:03
Downloading nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (139.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 139.1/139.1 MB 10.8 MB/s  0:00:12
Downloading scipy-1.16.2-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.7/35.7 MB 11.0 MB/s  0:00:03
Downloading opt_einsum-3.4.0-py3-none-any.whl (71 kB)
Installing collected packages: jax-cuda12-pjrt, opt_einsum, nvidia-nvshmem-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-nvcc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, jax-cuda12-plugin, scipy, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, ml_dtypes, nvidia-cusolver-cu12, jaxlib, jax
Successfully installed jax-0.7.2 jax-cuda12-pjrt-0.7.2 jax-cuda12-plugin-0.7.2 jaxlib-0.7.2 ml_dtypes-0.5.3 numpy-2.3.3 nvidia-cublas-cu12-12.9.1.4 nvidia-cuda-cupti-cu12-12.9.79 nvidia-cuda-nvcc-cu12-12.9.86 nvidia-cuda-nvrtc-cu12-12.9.86 nvidia-cuda-runtime-cu12-12.9.79 nvidia-cudnn-cu12-9.14.0.64 nvidia-cufft-cu12-11.4.1.4 nvidia-cusolver-cu12-11.7.5.82 nvidia-cusparse-cu12-12.5.10.65 nvidia-nccl-cu12-2.28.3 nvidia-nvjitlink-cu12-12.9.86 nvidia-nvshmem-cu12-3.4.5 opt_einsum-3.4.0 scipy-1.16.2
(base) gotree94@G635LX:~$

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


