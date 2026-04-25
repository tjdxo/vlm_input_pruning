# VLM 입력 전처리 프로세스

이 문서는 현재 프로젝트에서 이미지 입력이 들어왔을 때 어떤 순서로 처리되고,
최종적으로 Qwen3-VL에 어떤 입력이 전달되는지 설명합니다.

## 목표

비교 대상은 두 가지입니다.

- 기본 방식: 원본 이미지 + 사용자 질문 -> Qwen3-VL
- 제안 방식: 원본 이미지 -> 전처리 -> compact crop canvas + context prompt -> Qwen3-VL

목표는 단순 resize로 세부 정보를 잃지 않으면서, 최종 VLM에 들어가는 시각
토큰 수를 줄이는 것입니다.

## 현재 권장 모델

- main VLM: `Qwen/Qwen3-VL-4B-Instruct`
- front-stage sVLM: `HuggingFaceTB/SmolVLM-256M-Instruct`
- detector: YOLO, 기본 모델 `yolov8n.pt`

앞단 sVLM은 `SmolVLM-256M-Instruct`를 기본 추천합니다. 500M 모델이 더
좋은 장면 요약을 만들 수 있지만, Colab GPU에서 Qwen3-VL과 YOLO를 함께
돌리는 상황에서는 256M 쪽이 GPU 메모리 부담이 작습니다.

## 처리 순서

1. **이미지 로드**

   입력 이미지를 RGB 이미지로 읽습니다.

2. **앞단 작업 병렬 실행**

   `frontend.parallel: true`이면 이미지 로드 후 두 작업을 동시에 실행합니다.

   - sVLM scene context: SmolVLM이 전체 장면과 중요한 시각 영역을 요약합니다.
   - detector: YOLO가 객체 후보 bbox를 찾습니다.

   두 작업은 서로 결과에 의존하지 않기 때문에 병렬로 실행할 수 있습니다.

3. **객체 후보 점수화**

   detector가 만든 각 bbox에 importance score를 붙입니다. 현재 기준은 다음과
   같습니다.

   - detector confidence
   - bbox 면적
   - 이미지 중앙에 가까운 정도
   - 사용자 질문과 detector label의 단어 겹침
   - SmolVLM scene context와 detector label의 단어 겹침

4. **crop 제외 규칙 적용**

   최종 crop canvas가 원본보다 커지는 상황을 막기 위해 다음 규칙을 적용합니다.

   - bbox 좌표가 완전히 동일한 detector 결과는 첫 번째 결과만 남깁니다.
   - 원본 이미지의 75% 이상을 차지하는 bbox는 제거합니다.
   - 다른 남은 bbox 안에 완전히 포함되는 bbox는 제거합니다.
   - 선택된 crop들의 총 픽셀 수가 원본 이미지 픽셀 수를 넘으면 작은 crop부터 제거합니다.
   - 최종 composed canvas의 추정 이미지 토큰 수가 원본보다 크면 작은 crop부터 제거합니다.

   큰 bbox 제거를 포함 bbox 제거보다 먼저 수행합니다. 그래야 거대한 scene-level
   bbox가 작은 객체들을 먼저 제거하고, 그 뒤 자신도 제거되어 정보가 모두 사라지는
   상황을 피할 수 있습니다.

5. **compact crop canvas 생성**

   선택된 bbox를 원본 이미지에서 margin과 함께 crop합니다. 그 crop들을 하나의
   이미지 canvas에 타일처럼 배치합니다.

   기존처럼 항상 1600x1600 canvas를 쓰지 않고, 실제 crop들이 차지하는 크기만큼
   canvas를 줄입니다. 작은 벤치마크 이미지가 전처리 후 오히려 커지는 문제를
   줄이기 위한 규칙입니다.

6. **최종 prompt 생성**

   최종 prompt에는 다음 정보가 들어갑니다.

   - SmolVLM이 만든 전체 장면 context
   - 선택된 crop metadata
   - 원래 사용자 질문
   - 세부 시각 근거는 crop canvas를 우선 사용하라는 지시

7. **Qwen3-VL 호출**

   Qwen3-VL에는 다음 두 입력이 들어갑니다.

   - compact crop canvas 이미지
   - 최종 prompt 텍스트

## 생성 출력

각 실행은 다음 파일을 만듭니다.

- `*_composed.jpg`: Qwen3-VL에 전달되는 compact crop canvas
- `*_detections.jpg`: detector bbox 시각화 이미지
- `*_final_prompt.txt`: crop canvas와 함께 전달되는 최종 prompt
- `*_metadata.json`: detection, crop, latency, token estimate 등 메타데이터

MMBench Colab 비교 노트북은 추가로 다음 파일을 만듭니다.

- `mmbench_20_qwen_results.csv`: 샘플별 예측, 정답, 시간, GPU 메모리
- `mmbench_20_qwen_summary.csv`: 정확도, 평균 시간, GPU 메모리, 토큰 감소 요약

## 평가 지표

MMBench 공식 점수는 기본적으로 정답 정확도입니다. 이 프로젝트는 실험용으로
다음 엔지니어링 지표를 추가 기록합니다.

- 평균 소요 시간
- `nvidia-smi` 기준 peak GPU memory
- PyTorch peak memory delta
- approximate image-token reduction ratio
- detection 수와 최종 selected crop 수

현재 토큰 추정치는 Qwen 계열 patch 기준의 근사치입니다. 실제 모델 tokenizer의
정확한 내부 토큰 수와 완전히 같지는 않습니다.
