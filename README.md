# DotsOCR – Vision-Language Model Training & OmniDocBench Inference

본 레포지토리는 **문서 이해(Document Understanding)**를 위한 Vision-Language Model(VLM)을 학습하고, **OmniDocBench**와 같은 문서 벤치마크 데이터셋에서 추론을 수행하기 위한 코드들을 포함합니다.
본 프로젝트는 IITP의 '클라우드 장애극복을 위한 AI 어시스턴트 기반 운영·관리 자동화 기술개발' 연구과제를 위한 Document Parser 기능을 목표로 합니다.

## 1. Repository Overview

이 레포지토리는 다음과 같은 목적을 가집니다.

- PDF / 이미지 기반 문서 입력 처리
- Qwen-VL 계열 모델 기반 Vision-Language Model 학습
- vLLM 기반 고속 추론 파이프라인 구성
- OmniDocBench 평가 포맷에 맞춘 결과 생성

연구 및 실험 목적의 코드로, 문서 OCR 및 멀티모달 문서 이해 성능 검증에 활용됩니다.

## 2. Repository Structure
```text
.
├── train.py
├── inference_omnidocbench.py
└── requirements.txt
```

## 3. File Description
### 3.1 train.py

Vision-Language Model 학습 스크립트

문서 이미지와 텍스트를 함께 입력으로 사용하는 Vision-Language Model(VLM)을
학습하기 위한 메인 학습 코드입니다.
Qwen-VL 계열 모델을 기반으로 하며, LoRA 기반 파인튜닝을 지원합니다.

주요 기능
	•	Hugging Face transformers 기반 학습 파이프라인 구성
	•	Vision + Text 입력을 처리하는 멀티모달 모델 학습
	•	LoRA(PEFT) 적용을 통한 효율적 파인튜닝
	•	문서 이미지 전처리 및 vision token 정규화
	•	멀티 GPU / 분산 학습 환경 지원

핵심 구성
	•	AutoModelForCausalLM, AutoProcessor 기반 모델 로딩
	•	Trainer, TrainingArguments를 이용한 학습 관리
	•	커스텀 Dataset 구현을 통한 PDF / 이미지 로딩
	•	qwen_vl_utils.process_vision_info를 활용한 vision 입력 처리
	•	DotsOCR 내부 유틸 모듈 연동

사용 목적
	•	문서 OCR 및 구조 이해를 위한 VLM 사전학습 / 파인튜닝
	•	OmniDocBench, PubLayNet, 내부 문서 데이터셋 학습 실험
	•	문서 질의응답 및 문서 파싱 성능 향상 연구

### 3.2 inference_omnidocbench.py

OmniDocBench 추론 전용 스크립트

학습된 Vision-Language Model을 이용해
OmniDocBench 포맷의 문서 데이터셋에 대해 자동 추론을 수행하는 스크립트입니다.

주요 기능
	•	vLLM 기반 고속 추론 파이프라인
	•	PDF 문서 페이지 단위 이미지 변환 및 처리
	•	Prompt mode 기반 문서 질의 생성
	•	멀티프로세싱 / 병렬 추론 지원
	•	OmniDocBench 제출 형식(JSON) 결과 저장

처리 흐름
	1.	PDF 또는 이미지 문서 로딩
	2.	페이지 단위 이미지 변환
	3.	해상도 및 픽셀 수 기준 리사이즈
	4.	Prompt mode에 따른 질의 생성
	5.	vLLM 엔진을 통한 병렬 추론 수행
	6.	결과를 OmniDocBench 형식 JSON으로 저장

사용 목적
	•	OmniDocBench 벤치마크 자동 평가
	•	문서 이해 모델 추론 성능 검증
	•	학습된 VLM의 실제 문서 처리 능력 평가

### 3.3 requirements.txt
본 레포지토리를 실행하기 위해 필요한 Python 패키지 목록을 정의합니다.
