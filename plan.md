# PTN EMS 딥러닝 기반 이상탐지 솔루션 구현 계획서 (plan.md)

## 1. Background & Motivation (배경 및 목적)
- **배경:** PTN(Packet Transport Network) EMS(Element Management System) 장비에서 15분 단위로 성능 데이터(누적 Packet, Error, BPS, PPS)와 광모듈 파워 데이터(Tx/Rx)가 수집되어 MySQL DB에 저장되고 있음.
- **목표:** 수집된 시계열 데이터를 기반으로 시스템의 이상 징후를 선제적으로 파악하기 위해 딥러닝 기반(LSTM-Autoencoder)의 이상탐지 솔루션을 개발 및 온프레미스 환경에 구축.
- **하네스 도입:** AI 에이전트 기반 자율 개발(Harness Coding) 파이프라인을 적용하여, 설계-구현-테스트-학습의 제어 루프를 통해 코드의 안정성과 품질을 확보.

## 2. Scope & Architecture (범위 및 아키텍처)
- **개발 언어 및 프레임워크:** Python, PyTorch (Deep Learning), Pandas/NumPy (Data Processing), SQLAlchemy (DB Connection), APScheduler (Task Scheduling).
- **데이터베이스:** MySQL (입력 데이터 소스 및 탐지 결과 저장소).
- **핵심 컴포넌트:**
  1. `DataFetcher`: MySQL DB에서 주기적(15분)으로 최신 데이터를 조회.
  2. `Preprocessor`: 결측치 처리, 파생 변수 생성(예: 누적 Packet의 증감량 변환), 스케일링(MinMaxScaler).
  3. `AnomalyDetector (Model)`: LSTM-Autoencoder 모델 아키텍처 정의, 학습(Train) 및 추론(Inference) 기능.
  4. `InferenceEngine`: 스케줄러를 통해 주기적으로 전체 파이프라인(Fetch -> Preprocess -> Detect -> Save) 실행.

## 3. Implementation Steps (하네스 기반 파이프라인 구현 단계)

### Phase 1: 기반 설정 및 데이터 파이프라인 구축 (Go - Data Layer)
- **목표:** MySQL 연결 및 데이터 추출, 전처리 로직 구현.
- **작업 내용:**
  - `config.py` 작성 (DB 접속 정보, 모델 하이퍼파라미터 등 설정 관리).
  - `db_connector.py` 작성 (데이터베이스 연결 및 쿼리 실행).
  - `data_processor.py` 작성 (데이터 정제, 스케일링, 시계열 시퀀스(Window) 생성).
- **검증 (QA):** Mock 데이터를 활용하여 전처리 결과물 형태 및 DB 연결 정상 여부 단위 테스트 (`test_data.py`).

### Phase 2: 모델 개발 및 학습 로직 (Go - Model Layer)
- **목표:** LSTM-AE 모델 구조 정의 및 학습 코드 작성.
- **작업 내용:**
  - `model.py` 작성 (PyTorch `nn.Module`을 상속받은 LSTM-Autoencoder 클래스 구현).
  - `trainer.py` 작성 (Loss function(MSE), Optimizer 설정, 모델 학습 루프 및 모델 가중치(pth) 저장).
- **검증 (QA):** 더미 텐서를 입력으로 받아 모델의 forward/backward 패스가 에러 없이 동작하는지 테스트 (`test_model.py`).

### Phase 3: 추론 엔진 및 스케줄러 통합 (Go - Application Layer)
- **목표:** 15분 주기로 동작하는 자동화된 추론 및 결과 저장 파이프라인 완성.
- **작업 내용:**
  - `inference.py` 작성 (저장된 모델 로드, 최신 데이터 예측, Reconstruction Error 기반 이상 점수 산출 및 임계치(Threshold) 비교).
  - `scheduler.py` 작성 (APScheduler 적용, 15분마다 `inference.py`의 핵심 함수 호출).
  - 추론 결과를 DB 테이블에 `INSERT` 하는 로직 추가.
- **검증 (QA):** 전체 파이프라인(End-to-End) 모의 실행 테스트.

## 4. Harness Checklists & Rules (하네스 체크리스트 및 원칙)
- **(방향 제시) Target-Plan 매핑:** 작성된 모든 모듈은 본 `plan.md`의 목표와 연결되어야 함. 목적 없는 코드 작성 금지.
- **(가이드 제공) 레이어 분리:** 데이터 접근(DB), 비즈니스 로직(전처리), 모델 로직(PyTorch)을 철저히 분리하여 의존성을 최소화할 것.
- **(결과 분류 및 제어 루프) 단위 테스트 필수:** 각 모듈 작성 후 반드시 테스트 코드를 작성 및 실행(PASS/FAIL 확인). 
- **(자가 학습 루프) 3-Strike 룰:** 동일한 오류(예: PyTorch Tensor 차원 불일치)가 3회 이상 발생 시, 해당 모듈 수정을 멈추고 `_lessons.md` 파일에 오류 원인과 해결 원칙을 누적 기록한 뒤 재설계 진행.

## 5. Verification & Testing (검증 계획)
- **Unit Test:** `pytest` 프레임워크 사용. 각 모듈의 기능(DB 연결, 차원 변환, 모델 연산 등)을 독립적으로 검증.
- **Integration Test:** 전체 데이터 흐름(조회 -> 전처리 -> 모델 예측 -> DB 저장)이 끊김 없이 동작하는지 통합 테스트 진행.
- **에러 핸들링:** DB 접속 실패, 데이터 결측 등 예외 상황 발생 시 프로세스가 죽지 않고 로그를 남기며 복구되거나 다음 스케줄을 대기하도록 방어적 코드(Try-Except) 작성.

## 6. Migration & Rollback
- 기존 운영 중인 EMS 시스템에 영향을 주지 않는 독립된 환경(컨테이너 또는 별도 프로세스)에서 동작하도록 설계.
- 이상탐지 결과 테이블은 기존 테이블과 분리하여 생성. 오류 발생 시 추론 프로세스(Scheduler)만 종료하면 기존 시스템으로의 원복(Rollback) 완료.
