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

### Phase 1: 기반 설정 및 데이터 파이프라인 구축 (Complete - Data Layer)
- **목표:** MySQL 연결 및 데이터 추출, 전처리 로직 구현.
- **작업 내용:**
  - `config.py` 작성 (DB 접속 정보, 모델 하이퍼파라미터 등 설정 관리).
  - `db_connector.py` 작성 (데이터베이스 연결 및 쿼리 실행).
  - `data_processor.py` 작성 (데이터 정제, 스케일링, 시계열 시퀀스(Window) 생성).
- **검증 (QA):** Mock 데이터를 활용하여 전처리 결과물 형태 및 DB 연결 정상 여부 단위 테스트 (`test_data.py`).

### Phase 2: 모델 개발 및 학습 로직 (Complete - Model Layer)
- **목표:** LSTM-AE 모델 구조 정의 및 학습 코드 작성.
- **작업 내용:**
  - `model.py` 작성 (PyTorch `nn.Module`을 상속받은 LSTM-Autoencoder 클래스 구현).
  - `trainer.py` 작성 (Loss function(MSE), Optimizer 설정, 모델 학습 루프 및 모델 가중치(pth) 저장).
- **검증 (QA):** 더미 텐서를 입력으로 받아 모델의 forward/backward 패스가 에러 없이 동작하는지 테스트 (`test_model.py`).

### Phase 3: 추론 엔진 및 스케줄러 통합 (Complete - Application Layer)
- **목표:** 15분 주기로 동작하는 자동화된 추론 및 결과 저장 파이프라인 완성.
- **작업 내용:**
  - `inference.py` 작성 (저장된 모델 로드, 최신 데이터 예측, Reconstruction Error 기반 이상 점수 산출 및 임계치(Threshold) 비교).
  - `scheduler.py` 작성 (APScheduler 적용, 15분마다 `inference.py`의 핵심 함수 호출).
  - 추론 결과를 DB 테이블에 `INSERT` 하는 로직 추가.
- **검증 (QA):** 전체 파이프라인(End-to-End) 모의 실행 테스트.

### Phase 4: 앙상블 아키텍처 및 데이터 표준화 (Complete - Advanced Layer)
- **목표:** Traffic과 Optical 트랙을 분리하여 정확도를 높이고 실데이터 대응력 강화.
- **작업 내용:**
  - 데이터 스키마 표준화 (`ip_addr`, `cid`, `lid` 기반 식별 체계 확립).
  - Traffic/Optical 독립 트랙 구축 및 앙상블 탐지 로직 구현.
  - 실제 장애 패턴 분석을 위한 진단 메시지(`anomaly_reason`) 생성 로직 추가.
- **검증 (QA):** `export_anomalies.py`를 통한 과거 장애 이력 재현 및 탐지율 검증.

### Phase 5: 모델 최적화 및 통합 검증 파이프라인 (Complete - Operations Layer)
- **목표:** 실제 운영 환경 데이터를 수용하는 고정밀 모델 학습 및 자동화된 검증 체계 구축.
- **작업 내용:**
  - **비선형 전처리**: Traffic 데이터 `log1p` 변환 및 `RobustScaler` 적용을 통한 이상치 내성 확보.
  - **모델 고도화**: LSTM-Autoencoder 100 에포크 학습, Loss 0.003 수준의 안정적 수렴 달성.
  - **통합 검증 엔진**: `test_inference.py`를 개편하여 [추론 + CSV 저장 + 상세 진단 리포트] 기능 통합.
  - **시각화 자동화**: 탐지된 이상 사례의 트래픽 패턴과 점수를 매칭한 그래프 생성 도구(`visualize_results.py`) 개발.
- **검증 (QA):**
    - 과거 장애 이력 데이터를 활용한 실제 탐지 성능(Hit Rate) 확인 및 시각적 검증 완료.
    - **장기 가동 테스트**: 15분 주기 스케줄러의 안정적 반복 동작 확인 및 DB/CSV 저장 무결성 검증 완료. (2026-05-11)

### Phase 6: 예지 정비 및 지능형 분석 엔진 (Complete - Intelligence Layer)
- **목표:** 탐지된 이상을 수치화하고, 추세 분석을 통해 미래 장애를 예측하는 지능형 엔진 구현.
- **작업 내용:**
  - **심각도 정규화 (Severity Scoring)**: 원본 MSE 점수를 0~100 사이의 직관적 점수로 변환하는 비선형 매핑 함수 구현 완료.
  - **다단계 경보 체계 (Alerting)**: 점수 구간별 주의(Minor), 경고(Major), 심각(Critical) 등급 부여 로직 수립 완료.
  - **추세 분석 (Trend Analysis)**: 이상 점수의 상승 기울기(Slope) 분석을 통한 초기 단계 장애 징후 포착 및 중복 저장 방지 필터링 구현 완료.
  - **잔여 수명 예측 (RUL Prediction)**: 현재 추세 지속 시 장애 발생 예상 시점(Time to Failure)을 산출하고, 15분 수집 주기에 맞춰 정규화(Ceil) 처리 로직 구현 완료. (2026-05-11)
- **검증 (QA):** 실제 장애 발생 시나리오 기반의 경보 리드타임 측정 및 RUL 정확도 검증 완료.

### Phase 7: 통합 서비스 아키텍처 및 Web UI 구현 (Complete - Service Layer)
- **목표:** 엔진을 TSDN 컨트롤러와 연동 가능한 서비스 형태로 승격시키고, 실시간성을 보장하는 현대적인 웹 대시보드 구축.
- **작업 내용:**
  - **Backend API 기반 통합 (FastAPI)**: 기존 스케줄러를 내재화한 통합 API 서버 구축 및 REST 엔드포인트 제공 완료.
  - **API 고도화 및 통합**: `/api/trend`를 `/api/anomalies`로 통합하고, 유연한 필터링(Min/Max Severity, Rising Trend) 및 가장 최신 배치 데이터 자동 조회 기능 구현 완료.
  - **실시간 알림 엔진 (EventStream/SSE)**: Critical 경보 발생 시 웹/컨트롤러로 즉시 데이터를 푸시하는 SSE 엔드포인트 구현 완료.
  - **통합 모니터링 Web GUI (Vue 3)**: Vue 3 기반 실시간 대시보드 구현. Watchlist(상승 추세 감시), 클라이언트 측 페이징(Pagination), 직관적인 심각도 시각화 포함.
  - **시계열 분석 그래프 통합 (Chart.js)**: Chart.js를 이용한 포트별 24시간 시계열 추이(이상 점수, 트래픽/광파워) 시각화 완료.
  - **수동 새로고침 및 스케줄러 제어 로직 보강**: UI 편의성 개선 및 스케줄러 정지 시 데이터 초기화 로직 구현 완료.
  - **[Advanced] 모델 관리 시스템**: 훈련/추론 설정 분리, 날짜 기반 맞춤형 학습 파이프라인, **실시간 학습 상태(에포크, 손실률) 추적 및 학습 강제 중지 API** 구현 완료. (2026-05-12)
  - **신뢰성 강화**: **Early Stopping** 도입으로 최적 모델 자동 선발, MySQL 데이터 정제(NaN/inf 처리) 로직 통합 완료. (2026-05-12)
- **검증 (QA):** API-UI 연동 무결성 확인 및 실시간 이벤트 스트리밍을 통한 대시보드 자동 갱신 검증 완료. (2026-05-12)

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
