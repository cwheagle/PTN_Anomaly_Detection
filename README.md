# PTN EMS 이상탐지 솔루션 (PTN Anomaly Detection)

> **PTN(Packet Transport Network) EMS 장비의 시계열 성능 데이터를 기반으로, LSTM-Autoencoder 딥러닝 모델을 활용한 실시간 이상 탐지 및 근본 원인 분석(RCA) 솔루션**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1?style=flat-square&logo=mysql&logoColor=white)](https://www.mysql.com)
[![Vue](https://img.shields.io/badge/Vue_3-Vite-42B883?style=flat-square&logo=vuedotjs&logoColor=white)](https://vitejs.dev)

---

## 📌 프로젝트 개요

통신 장비 운용 환경에서 **15분 단위**로 수집되는 성능 지표(트래픽, 광 모듈 상태)를 분산된 MySQL DB로부터 수집·분석하여 장애를 **사전에 탐지**하고 **예지 정비(Predictive Maintenance)**를 가능하게 하는 AI 엔진 및 대시보드입니다.

단순한 임계치 경보 수준을 넘어, **어떤 장비의 어떤 포트에서 어떤 장애가 의심되는지**를 진단하고 구체적인 조치 방법을 운용자에게 제시합니다.

---

## ✨ 핵심 기능

| 기능 | 설명 |
|------|------|
| 🔍 **실시간 이상 탐지** | LSTM-Autoencoder 기반 Reconstruction Error로 15분마다 자동 추론 |
| 📊 **심각도 스코어링** | MSE 기반 0~100점 상대 심각도 + 3단계 경보(Minor/Major/Critical) |
| 🔗 **RCA 진단 엔진** | Feature Contribution 분석 + 도메인 룰 기반 장애명 및 조치 방법 출력 |
| ⏱️ **RUL 예측** | 현재 추세 기반 장애까지 남은 예상 시간(Remaining Useful Life) 계산 |
| 🌡️ **동적 임계치** | 3-Sigma 기반 시간대/요일별 가변 임계치로 오탐(False Positive) 최소화 |
| 🔄 **자동 재학습** | Data Drift 감지 시 백그라운드 재학습 → Hot-Reload 무중단 배포 |
| 🛠️ **Rule Management UI** | 네트워크 엔지니어가 직접 RCA 룰을 등록·수정하는 관리 인터페이스 |
| 📡 **실시간 대시보드** | FastAPI SSE 기반 실시간 이벤트 스트리밍 + Vue 3/Vite 프론트엔드 |

---

## 🏗️ 시스템 아키텍처

```
MySQL DB (분산)          Kafka Message Bus           FastAPI & Vue
┌─────────────┐         ┌─────────────────┐         ┌───────────────────┐
│ Traffic DB  │──수집──▶│ ptn_metrics     │──구독──▶│ Consumer Workers  │
│ Optical DB  │         │ (Kafka Topic)   │         │ (추론 & RCA 진단) │
└─────────────┘         └─────────────────┘         └─────────┬─────────┘
                                                              │ 웹훅 알람
                                                    ┌─────────▼─────────┐
                                                    │ API Server (SSE)  │──▶ 대시보드
                                                    └───────────────────┘
```

### 분석 트랙

| 트랙 | 모니터링 피처 (3종 / 2종) |
|------|---------------------------|
| **Traffic** | In Packet, Out Packet, Error Packet (3종) |
| **Optical** | Tx 평균 광 파워, Rx 평균 광 파워 (2종) |

---

## 📁 프로젝트 구조

```
PTN_Anomaly_Detection/
├── src/
│   ├── api/                # FastAPI 라우터 및 SSE 엔드포인트
│   ├── data/               # DataFetcher, Preprocessor, DB Connector
│   ├── models/             # LSTM-Autoencoder (model.py, trainer.py)
│   ├── pipeline/           # InferenceEngine, Scheduler, Dynamic Threshold
│   ├── rca/                # RCA 엔진, Feature Contribution, Rule Engine
│   │   └── rules/          # 도메인 룰셋 JSON (16종 PTN 룰)
│   └── config.py           # 중앙화된 설정 관리
├── ui/                     # Vue + Vite 프론트엔드
│   └── src/views/
│       ├── DashboardView.vue       # 실시간 이상탐지 대시보드
│       ├── ModelManagementView.vue # 모델 파라미터 설정 및 훈련 관리
│       └── RuleManagementView.vue  # RCA 도메인 룰 등록·수정
├── tools/
│   └── simulator/          # 테스트용 데이터 시뮬레이터 (이상 주입, 이력 생성)
├── tests/
│   ├── module/                 # 모듈별 단위 동작 검증 (pytest)
│   │   ├── test_data.py        # 데이터 수집·전처리 모듈
│   │   ├── test_model.py       # LSTM-AE 모델 아키텍처
│   │   ├── test_trainer.py     # 학습 루프
│   │   └── test_scheduler.py  # 스케줄러
│   ├── test_train.py           # 모델 훈련 실행 (Traffic/Optical 전체)
│   ├── test_inference.py       # 추론 엔진 실행 테스트
│   ├── test_rca.py             # RCA 엔진 진단 테스트
│   └── test_run.py             # 전체 파이프라인 통합 실행
├── scripts/
│   ├── collect_data.py         # 데이터 수집 단독 실행
│   ├── main_scheduler.py       # 스케줄러 테스트 가동
│   └── evaluate_model.py       # 오프라인 모델 검증 및 F1-Score 평가
├── docs/                   # 설계 문서 및 참고 자료
├── models/                 # ⚡ 동적 생성 — 학습된 모델 가중치 저장소 (git 제외)
├── data/                   # ⚡ 동적 생성 — 추론 결과 및 평가용 CSV (git 제외)
├── plan.md                 # 단계별 구현 계획 (Single Source of Truth)
├── lessons.md              # 시행착오 및 교훈 기록
└── requirements.txt        # Python 의존성
```

---

## 🚀 빠른 시작

### 사전 요구사항

- Python 3.10+
- Node.js 18+ (프론트엔드)
- MySQL 8.0+
- PyTorch 2.x (CUDA 권장)

### 1. 백엔드 설치 및 설정

```bash
# 의존성 설치
pip install -r requirements.txt
pip install torch  # PyTorch는 공식 사이트에서 CUDA 버전에 맞게 설치 권장

# 설정 파일 복사 후 DB 접속 정보 입력
cp src/config.py.example src/config.py
# config.py 편집: DB 호스트, 포트, 계정, 대상 장비 정보 입력
```

### 2. 인프라 및 서버 실행 (Docker Compose)

모든 시스템 컴포넌트(UI, API, Kafka, Redis, Consumer, Producer)는 Docker Compose로 통합되어 한 번에 실행됩니다. (Phase 12 적용)

```bash
# 전체 시스템 빌드 및 백그라운드 가동
docker-compose up -d --build

# 실행 중인 컨테이너 상태 확인
docker-compose ps
```

서버 기동 후 **Model Management UI** 또는 API로 초기 모델 학습을 시작합니다.

```bash
# Traffic 모델 학습
curl -X POST "http://localhost:8000/api/model/train?feature_type=traffic"

# Optical 모델 학습
curl -X POST "http://localhost:8000/api/model/train?feature_type=optical"
```

> 학습 진행 상태 및 파라미터 설정은 **Model Management** 뷰에서 실시간으로 확인할 수 있습니다.

### 3. 프론트엔드 직접 실행 (로컬 개발 시)

Docker를 사용하지 않고 로컬에서 UI를 띄울 때는 아래를 참고하세요.
```bash
cd ui
npm install
npm run dev
# http://localhost:5173 접속
```

---

## 🛠️ 빌드, 실행 및 디버깅 방법

### Docker Compose 환경 (운영/통합 테스트 권장)
- **전체 종료 및 초기화:** `docker-compose down -v` (볼륨까지 완전히 삭제)
- **특정 서비스 실시간 로그 확인 (디버깅 핵심):**
  ```bash
  docker-compose logs -f api       # API 서버 로그 (SSE 웹훅 브로드캐스트 등)
  docker-compose logs -f consumer  # AI 추론 엔진 로그 (이상 탐지 및 알람 판정)
  docker-compose logs -f producer  # 데이터 수집기 폴링 로그
  docker-compose logs -f ui        # 프론트엔드 Nginx 접속 로그
  ```
- **Consumer 버퍼(Redis 윈도우) 초기화:** 
  기존 테스트 데이터가 꼬였을 때 상태를 비웁니다.
  ```bash
  docker exec redis redis-cli flushall
  ```

### 로컬 개발 환경 (개별 컴포넌트 디버깅 시)
코드를 수정하면서 즉시 반영(Reload)하거나 브레이크포인트를 걸기 위해서는 인프라만 컨테이너로 띄우고, 앱은 터미널에서 직접 실행하는 것이 편리합니다.
1. `docker-compose up -d kafka redis zookeeper` (인프라만 실행)
2. `.env` 파일을 만들거나 환경 변수를 세팅하여 `DB_HOST`, `REDIS_HOST`, `KAFKA_BROKERS` 등을 `localhost`로 맞춥니다.
3. 터미널을 여러 개 열어 개별 실행:
   - API 서버: `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
   - Consumer: `python src/pipeline/kafka_consumer.py`
   - Producer (과거 데이터 시뮬레이션 테스트): 
     ```bash
     python src/pipeline/kafka_producer.py --mode sim --start "2026-07-27 06:00:00" --end "2026-07-27 11:00:00"
     ```

---

## 📦 배포 가이드

운영 서버에 소스 코드를 유출하지 않고 안전하게 배포(납품)하는 오프라인 배포 프로세스

### 1. 사내 빌드 서버에서 도커 이미지 패키징
먼저 인터넷이 연결된 환경에서 외부 인프라 이미지를 다운로드(Pull)하고, 자체 소스 코드를 도커 이미지로 빌드합니다.
빌드 전 `docker-compose.prod.yml` 의 `platform` 속성을 배포 서버의 운영체제에 맞게 설정합니다.
```bash
docker-compose pull
docker-compose build
```

### 2. 도커 이미지를 단일 `.tar` 압축 파일로 추출
자체 개발한 4개의 애플리케이션 이미지뿐만 아니라, **인프라 이미지(Kafka, Zookeeper, Redis)도 반드시 함께 추출**해야 인터넷이 없는 환경에서 구동됩니다.
```bash
docker save -o ptn_anomaly_detection_v1.tar ptn_anomaly_detection-api:latest ptn_anomaly_detection-ui:latest ptn_anomaly_detection-consumer:latest ptn_anomaly_detection-producer:latest confluentinc/cp-kafka:7.5.0 confluentinc/cp-zookeeper:7.5.0 redis:7.2-alpine
```

### 3. 배포 서버에 이미지 이식 (Load)
배포 서버로 `ptn_anomaly_detection_v1.tar`, `docker-compose.yml` 파일을 복사한 뒤, `docker-compose.yml`에서 `build`속성을 삭제하고 `image: 'container_name':latest`으로 변경한 후 이미지를 로드합니다.
```bash
docker load -i ptn_anomaly_detection_v1.tar
```

### 4. 배포 서버 환경 설정 및 최종 구동
`docker-compose.yml` 내의 환경 변수(`DB_HOST` 등)를 배포 서버 환경에 맞게 수정한 후 실행합니다.
```bash
docker-compose up -d
```
이후 `http://배포_서버_IP` 로 접속하여 대시보드가 정상적으로 뜨는지 확인합니다.

---

## 🤖 RCA 엔진 — 도메인 룰 구조

이상 탐지 후 각 Feature의 MSE 기여도를 분석하여 장애 원인을 진단합니다.

```json
{
  "id": "TR-001",
  "track": "traffic",
  "priority": 100,
  "contributions": {
    "rx_packet": 35,
    "tx_packet": 35
  },
  "raw_conditions": {
    "min_rx_packet_ratio": 1.5,
    "min_tx_packet_ratio": 1.5
  },
  "diagnosis": "양방향 트래픽 동시 급증 (L2 브로드캐스트 스톰 또는 루핑 의심)",
  "action": "하위 스위치 루프(Loop) 구성 및 MAC Address 플래핑 이력 확인"
}
```

- **`track`**: 적용 대상 트랙 (`traffic` / `optical`)
- **`priority`**: 룰 매칭 우선순위 (높을수록 먼저 적용)
- **`contributions`**: 이상 판정에 기여해야 할 Feature와 기여도 기준(%)
- **`raw_conditions`**: 원시 값 기반 추가 조건 (비율, 절댓값 등)
- **`diagnosis`**: 출력할 장애 진단명
- **`action`**: 운용자에게 제시할 조치 방법

현재 **16종의 PTN 도메인 룰셋**이 주입되어 있으며, Rule Management UI를 통해 추가·수정이 가능합니다.


---

## 📈 개발 단계 현황

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 1 | 기반 설정 및 MySQL 데이터 연동 | ✅ 완료 |
| Phase 2 | LSTM-Autoencoder 모델 개발 및 학습 | ✅ 완료 |
| Phase 3 | 추론 엔진 및 15분 주기 스케줄러 | ✅ 완료 |
| Phase 4 | Traffic/Optical 트랙 분리 아키텍처 | ✅ 완료 |
| Phase 5 | 데이터 전처리 고도화 및 통합 검증 | ✅ 완료 |
| Phase 6 | 심각도 스코어링, RUL 예측, 추세 분석 | ✅ 완료 |
| Phase 7 | FastAPI 서버, SSE, Hot-Reload MLOps | ✅ 완료 |
| Phase 8 | RCA 엔진 코어 및 Rule Management UI | ✅ 완료 |
| Phase 9 | 3-Sigma 동적 임계치 + 자동 재학습 파이프라인 | ✅ 완료 |
| Phase 10 | 오프라인 모델 검증 및 TTF-Aware F1-Score 평가 | ✅ 완료 |
| Phase 11 | MLOps 고도화 (Alert Dampening, Lightweight Registry) | ✅ 완료 |
| Phase 12 | 대용량 분산 처리 (Kafka, Redis, Dockerization) | ✅ 완료 |
| **Phase 13** | **XAI (SHAP/LIME) 및 능동 학습 (Active Learning)** | 🔜 진행 예정 |

---

## 🧪 테스트 실행

```bash
# 모듈별 단위 동작 검증
pytest tests/module/ -v

# 모델 훈련
python tests/test_train.py

# 추론 엔진 테스트
python tests/test_inference.py

# RCA 엔진 테스트
python tests/test_rca.py

# 전체 파이프라인 통합 실행
python tests/test_run.py

# 데이터 수집 단독 실행
python scripts/collect_data.py

# 모델 오프라인 평가 및 F1-Score 산출
python scripts/evaluate_model.py
```

---

## 🛠️ 기술 스택

**백엔드 & 인프라**
- Python, PyTorch (LSTM-Autoencoder)
- FastAPI, Uvicorn (ASGI 서버)
- Apache Kafka (분산 메시징 버스), Redis (분산 상태 관리)
- MySQL, mysql-connector-python
- scikit-learn (RobustScaler), Pandas, joblib

**프론트엔드**
- Vue 3 + Vite + TypeScript
- TailwindCSS
- Server-Sent Events (SSE) 실시간 스트리밍

**개발 도구**
- `tools/simulator` — 이상 데이터 주입, 과거 이력 생성, 실시간 주입기 (테스트 환경 구축용)

---

## 📄 참고 문서

- [`plan.md`](./plan.md) — 단계별 구현 계획 (프로젝트 유일한 진실 공급원)
- [`lessons.md`](./lessons.md) — 개발 중 겪은 시행착오 및 교훈 기록
- [`GEMINI.md`](./GEMINI.md) — AI 에이전트 기반 개발(Harness Coding) 지침

---

## 🔒 보안 참고

`src/config.py`에는 DB 접속 정보 등 민감한 값이 포함되어 있습니다.  
이 파일은 `.gitignore`에 의해 저장소에서 제외됩니다.  
`src/config.py.example`을 참고하여 로컬 환경에서 직접 생성하세요.
