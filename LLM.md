로컬 LLM 실행 방법

1. 주요 LLM 프레임워크
A. Ollama (가장 쉬움, 추천!)
장점: 설치 간단, 모델 관리 쉬움, 최적화 우수
bash# 설치
curl -fsSL https://ollama.com/install.sh | sh

# 서비스 시작
sudo systemctl start ollama

# 모델 실행 (예: Llama 3.1 70B)
ollama run llama3.1:70b

# 다른 추천 모델들
ollama run llama3.1:8b          # 빠른 응답
ollama run codellama:34b        # 코딩
ollama run mistral:7b           # 효율적
ollama run deepseek-coder:33b   # 코딩 특화
ollama run qwen2.5:72b          # 다국어, 추론

# API 서버로 사용
# http://localhost:11434 에서 OpenAI 호환 API 제공
B. llama.cpp (최적화 최고)
장점: C++ 기반, 빠름, 양자화 우수
bash# 설치
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUDA=1

# 모델 실행 (GGUF 포맷)
./main -m models/llama-3.1-70b-Q4_K_M.gguf -n 512 -p "Your prompt"
C. vLLM (서버용, 고성능)
장점: 배치 처리, PagedAttention, 추론 속도 최고
bash# 설치
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm

# 모델 실행
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 1

# API: http://localhost:8000/v1
D. Text Generation WebUI (GUI)
장점: 웹 인터페이스, ChatGPT 스타일
bash# 설치
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
./start_linux.sh

# 웹 브라우저에서 http://localhost:7860 접속
# 모델 다운로드 및 채팅 가능
E. LM Studio (가장 사용자 친화적)
장점: GUI 기반, 드래그 앤 드롭, Windows/Mac/Linux
bash# 다운로드
wget https://releases.lmstudio.ai/linux/0.2.29/LM-Studio-0.2.29-x86_64.AppImage
chmod +x LM-Studio-0.2.29-x86_64.AppImage
./LM-Studio-0.2.29-x86_64.AppImage

2. RTX 5090 24GB에서 실행 가능한 모델
모델 크기별 VRAM 사용량
모델 크기양자화VRAM 사용속도품질7BFP1614GB매우 빠름좋음7BQ4_K_M4GB초고속양호13BFP1626GB❌ 불가-13BQ4_K_M8GB빠름좋음34BQ4_K_M20GB보통매우 좋음70BQ4_K_M40GB❌ 불가-70BQ3_K_M27GB❌ 불가-70BQ2_K20GB보통양호
실제 추천 모델 (24GB 기준)
최고 성능 (품질 우선)
bash# Qwen 2.5 32B (Q4) - 다국어, 수학, 코딩 우수
ollama run qwen2.5:32b

# DeepSeek Coder V2 16B - 코딩 최고
ollama run deepseek-coder-v2:16b

# Llama 3.1 8B (FP16) - 빠르고 정확
ollama run llama3.1:8b
균형 (속도 + 품질)
bash# Mistral 7B Instruct - 범용
ollama run mistral:7b-instruct

# CodeLlama 34B (Q4) - 코딩
ollama run codellama:34b

# Phi-3 Medium 14B - 효율적
ollama run phi3:14b
속도 우선
bash# Llama 3.2 3B - 초고속
ollama run llama3.2:3b

# Gemma 2 9B - 빠른 추론
ollama run gemma2:9b

# Mistral 7B (Q4) - 2GB VRAM
ollama run mistral:7b-instruct-q4_0

3. RTX 5090 24GB 최적 구성
추천 #1: Ollama (초보자/일반 사용)
bash# 설치
curl -fsSL https://ollama.com/install.sh | sh

# 다목적 모델 설치
ollama pull qwen2.5:32b          # 추론, 다국어, 수학
ollama pull deepseek-coder-v2:16b # 코딩
ollama pull llama3.1:8b          # 빠른 채팅

# 사용
ollama run qwen2.5:32b
>>> 한국어로 답변해줘. 양자역학을 설명해봐.
추천 #2: vLLM (서버/API 사용)
bash# 환경 생성
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm

# Qwen 2.5 32B 서버 실행
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192

# OpenAI API 스타일로 사용
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
추천 #3: Text Generation WebUI (GUI 선호)
bash# 설치
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
./start_linux.sh --api --listen

# 브라우저: http://localhost:7860
# Model 탭에서 다운로드:
# - Qwen/Qwen2.5-32B-Instruct-GGUF
# - TheBloke/deepseek-coder-33B-instruct-GGUF

4. 성능 비교 (RTX 5090 24GB)
추론 속도 (tokens/sec)
모델양자화VRAM속도용도Llama 3.2 3BQ42GB~150 tok/s채팅, 빠른 응답Llama 3.1 8BFP1614GB~80 tok/s범용Mistral 7BQ44GB~100 tok/s범용Qwen 2.5 14BQ48GB~60 tok/s추론, 다국어DeepSeek Coder 16BFP1618GB~50 tok/s코딩Qwen 2.5 32BQ420GB~30 tok/s고품질 추론CodeLlama 34BQ420GB~25 tok/s코딩

5. 특수 용도별 추천
생명과학/의료 AI 통합
bashconda create -n biomedical-llm python=3.10 -y
conda activate biomedical-llm

# vLLM 설치
pip install vllm

# BioGPT 또는 Medical LLM
# Med-PaLM 2는 비공개이지만 대안:
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/BioGPT-Large

# 또는 Meditron (의료 특화)
ollama pull meditron:70b
코딩 전용
bash# DeepSeek Coder V2 (최고 성능)
ollama run deepseek-coder-v2:16b

# 또는 CodeLlama
ollama run codellama:34b

# 사용 예
>>> Python으로 AlphaFold 단백질 구조 예측 코드 작성해줘
다국어 + 추론
bash# Qwen 2.5 (중국어, 한국어, 영어 우수)
ollama run qwen2.5:32b

# 또는 Command R+ (RAG 최적화)
ollama run command-r-plus:104b  # Q2 양자화 필요

6. 설치 용량
소프트웨어
Ollama                  : 500MB
llama.cpp               : 200MB
vLLM                    : 2GB
Text Generation WebUI   : 5GB
모델 용량 (GGUF 기준)
7B Q4                   : 4GB
13B Q4                  : 8GB
32B Q4                  : 20GB
70B Q2                  : 26GB
권장 디스크

소프트웨어: 10GB
모델 3-5개: 50-100GB
총: 100GB 여유 권장


7. 최종 추천 구성 (RTX 5090 24GB)
옵션 A: 올인원 (Ollama)
bash# 설치
curl -fsSL https://ollama.com/install.sh | sh

# 3개 모델 설치 (다용도)
ollama pull qwen2.5:32b           # 메인 (추론, 다국어)
ollama pull deepseek-coder-v2:16b # 코딩
ollama pull llama3.2:3b           # 빠른 작업

# 총 VRAM: 20GB (qwen), 18GB (deepseek), 2GB (llama3.2)
# 동시 실행 불가, 전환 사용
옵션 B: 서버 + API (vLLM)
bashconda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm

# 단일 모델 서버 (최고 성능)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --gpu-memory-utilization 0.9
옵션 C: GUI (Text Generation WebUI)
bashgit clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
./start_linux.sh

# 브라우저에서 http://localhost:7860
# 모델 다운로드 및 채팅

8. 통합 환경 구성
생명과학 + LLM 통합
bash# 환경 생성
conda create -n bio-ai-llm python=3.10 -y
conda activate bio-ai-llm

# PyTorch + CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 생명과학 라이브러리
pip install monai[all]
pip install bionemo-framework
pip install colabfold[alphafold]

# LLM 프레임워크
pip install vllm
pip install transformers accelerate bitsandbytes

# Ollama는 시스템 레벨로 별도 설치
curl -fsSL https://ollama.com/install.sh | sh

9. RAM + VRAM 통합 사용 (Offloading)
24GB VRAM으로 70B 모델 실행하기:
bash# llama.cpp with GPU offloading
./main \
    -m models/llama-3.1-70b-Q4_K_M.gguf \
    -ngl 35 \  # 35개 레이어만 GPU (나머지 CPU/RAM)
    -n 512

# 또는 Ollama
OLLAMA_NUM_GPU_LAYERS=35 ollama run llama3.1:70b
VRAM + RAM 조합:

VRAM 24GB: 35개 레이어
RAM 32GB: 나머지 레이어
속도: ~15 tok/s (느리지만 가능)


10. 비용 및 라이선스
무료 & 오픈소스

Llama 3.1 (Meta)
Mistral (Mistral AI)
Qwen 2.5 (Alibaba)
DeepSeek Coder (DeepSeek)
Phi-3 (Microsoft)

상업적 사용

대부분 Apache 2.0 또는 MIT
Llama: 월 사용자 7억 미만 무료
모두 로컬 실행 시 API 비용 없음


최종 추천
🥇 1순위: Ollama + Qwen 2.5 32B
bashcurl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:32b
ollama run qwen2.5:32b
이유: 쉬운 설치, 뛰어난 성능, 한국어 지원, 추론 능력 우수
🥈 2순위: vLLM + DeepSeek Coder V2 16B
bashpip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-Coder-V2-Instruct-0724
이유: 코딩 작업, API 서버, 빠른 추론
🥉 3순위: Text Generation WebUI
bashgit clone https://github.com/oobabooga/text-generation-webui
./start_linux.sh