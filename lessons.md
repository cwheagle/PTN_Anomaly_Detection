# PTN 이상탐지 프로젝트 학습 레슨 (lessons.md)

이 파일은 개발 과정에서 발생한 시행착오와 해결책을 기록하여, 향후 유사 프로젝트나 운영 단계에서 동일한 실수를 방지하기 위한 지식 베이스입니다.

## 1. 데이터 스키마 및 일관성 (Data Consistency)
- **교훈:** DB 컬럼명(`ip_addr`, `cid`, `lid`)과 코드 내 변수명(`equipment_id`, `port_id`)이 혼용될 경우 추론 및 결과 매핑 단계에서 반드시 오류가 발생함.
- **원칙:** 전 계층(DB -> Preprocess -> Model -> Inference -> Save)에서 원본 DB의 식별자 명칭을 최우선으로 유지하며, 변환이 필요한 경우 별도의 매핑 레이어를 명시적으로 둠.

## 2. 시계열 데이터 정합성 (Time-Series Integrity)
- **교훈:** 15분 단위 데이터에서 초 단위 오차나 누락이 발생하면 LSTM 시퀀스(Window)가 밀리거나 잘못된 시점의 이상을 탐지하게 됨.
- **해결:** `pd.to_datetime().dt.round('1min')`을 통해 시간 정규화(Rounding)를 필수적으로 수행.

## 3. 모델 차원 관리 (Dimensionality Management)
- **교훈:** `config.py`의 `input_dim`과 `DataProcessor`의 `feature_cols` 개수가 일치하지 않으면 런타임에 텐서 차원 오류가 발생함.
- **원칙:** 피처 추가/삭제 시 `config.py`를 중앙 제어판으로 사용함.

## 4. 메모리 및 성능 최적화 (Resource Management)
- **교훈:** 대량의 데이터를 한꺼번에 로드하면 메모리 부족(OOM)으로 프로세스가 종료될 수 있음.
- **해결:** 배치 단위 처리(`DataLoader`) 및 장비별 그룹화 처리를 통해 메모리 점유율을 분산시킴.

## 5. MySQL 데이터 정제 및 무결성 (MySQL Data Cleaning)
- **교훈:** 딥러닝 연산 중 발생하는 `NaN`이나 `inf` 수치는 MySQL 저장 시 에러를 유발함.
- **해결:** `DBConnector`에 `_clean_value`를 도입하여 저장 전 모든 수치를 안전한 값으로 치환함.

## 6. 조기 종료 및 최적 가중치 보존 (Early Stopping & Best Weights)
- **교훈:** 정해진 에포크를 모두 수행하면 과적합으로 인해 최종 모델의 성능이 저하될 수 있음.
- **해결:** Early Stopping을 구현하고, 검증 손실이 가장 낮았던 시점의 가중치를 로드하여 저장함.

## 7. 중단 가능한 장기 실행 작업 (Interruptible Tasks)
- **교훈:** 학습 등 장시간 소요되는 작업을 강제 종료하면 시스템 불안정성을 초래함.
- **해결:** `stop_checker` 패턴을 도입하여 루프 곳곳에서 중단 요청을 확인하도록 설계함.

## 8. API 버전 관리 및 의미론적 버전 (Semantic Versioning)
- **교훈:** 시스템 규모가 커짐에 따라 단순 문자열 버전 관리는 호환성 추적을 어렵게 함.
- **해결:** `API_VERSION`을 Major, Minor, Patch 단위로 분리 관리하고 응답 헤더에 강제 포함함.

## 9. 비동기 스케줄러와 이벤트 루프 (Async Scheduler)
- **교훈:** `AsyncIOScheduler` 실행 함수가 일반 `def`이면 이벤트 루프를 차단할 수 있음.
- **해결:** 스케줄러 핵심 작업을 `async def`로 전환하여 비차단(Non-blocking) 특성을 강화함.

## 10. 다중 모델의 통합 판단 로직 (Dominant Metric Selection)
- **교훈:** 여러 트랙의 점수를 합산/평균내면 심각한 이상이 희석될 수 있음.
- **해결:** 가장 심각한(Dominant) 점수를 가진 트랙을 기준으로 전체 심각도와 RUL을 결정함.

## 11. torch.compile과 멀티스레딩 환경의 CUDA Graphs 충돌
- **교훈:** `torch.compile`의 `reduce-overhead`나 `max-autotune` 모드에서 사용하는 CUDA Graphs는 FastAPI의 `BackgroundTasks` 같은 멀티스레딩 환경에서 `AssertionError`를 유발할 수 있음.
- **해결:** 멀티스레딩 환경에서는 `torch.compile(model)` (Default 모드)을 사용하거나, 명시적으로 `options={"triton.cudagraphs": False}`를 설정하여 스레드 안전성을 확보해야 함.

## 12. 고성능 GPU(Blackwell)에서의 배치 사이즈 최적화
- **교훈:** 대용량 시계열 데이터(23만건) 학습 시, 무조건 큰 배치(256)보다 적절한 배치(128)가 검증 손실(Validation Loss) 측면에서 약 28% 이상의 정확도 향상을 보임.
- **원칙:** 학습 속도와 정확도의 균형을 고려하여, 성능 개선폭이 유의미한 수준(10% 이상)일 때까지 하향 조정을 검토함.

## 13. 인메모리 모델 실시간 갱신 (In-memory Model Hot-Reloading)
- **교훈:** 새로 학습된 모델을 적용하기 위해 서비스를 재시작하면 실시간 추론의 연속성이 끊기고 다운타임이 발생함.
- **해결:** `AnomalyDetector` 클래스 내부에 `reload_model` 메서드를 구현하여, 실행 중인 인스턴스의 모델 가중치와 스케일러를 즉각 교체할 수 있도록 설계함. 이를 통해 무중단(Zero-downtime) 모델 배포가 가능해짐.

---
*최종 업데이트: 2026-05-20*

## 14. 분산 환경에서의 DB 접속 조용히 실패(Silent Failure) 방지
- **교훈:** GPU 서버와 로컬 PC처럼 분리된 환경에서 DB 접속을 시도할 때, db_connector 등에서 예외를 None 반환으로 무시하면, 상위 로직(DataCollector 등)이 실패를 인지하지 못하고 과거에 남아있던 캐시 데이터(csv)로 잘못된 학습을 진행하게 됨. 이는 디버깅을 극도로 어렵게 만듦.
- **해결:** DB 접속 실패 등 치명적인 초기화 오류는 try-except로 조용히 넘기지 말고, 명시적인 예외(Exception)를 발생시켜 파이프라인 전체를 중단시키도록 설계해야 함.

## 15. 시계열 예지(Predictive) 모델 평가 시의 '사후 정답(Post-Failure)' 처리
- **교훈:** 예지 정비(Predictive Maintenance) 관점에서 평가 스크립트를 작성할 때, `start_time` ~ `failure_time` 구간에 울린 알람만 정답(True Positive)으로 채점하면, 고장 상태가 방치된 '사후(Post-Failure)' 기간 동안 모델이 훌륭하게 울린 정당한 알람이 모조리 오탐(False Positive)으로 처리되어 F1-Score가 심각하게 훼손됨. (이로 인해 11만 개의 억울한 오탐이 발생했었음)
- **해결:** 장애 유지 기간을 고려하여 평가 윈도우를 여유롭게 확장하고, 피로도 억제(Dampening) 로직에 의해 살짝 지연된 알람도 정상 감지로 인정할 수 있도록 `evaluate_model.py`의 채점 로직을 현실적으로 보완함.

## 16. 전역 설정 dict를 참조로 받아 덮어쓰는 문제 (Shared Mutable Config)
- **교훈:** `self.paths = PATHS[ft]`는 복사가 아닌 참조라서, 학습 시 버전 경로로 덮어쓰면 전역 `PATHS`가 오염되어 다음 학습 파일명이 `traffic_ae_v1_v2.pth`처럼 누적되고 다른 모듈의 경로 조회도 어긋남.
- **해결:** 전역 설정은 `dict(...)`로 복사해서 쓰고, 인스턴스별로 바뀌는 값은 인스턴스에만 둠. 경로는 하드코딩 대신 레지스트리의 Active 버전(`get_active_model_info`)으로 조회.

## 17. 컨테이너 분리 시 공유 상태와 접속 주소 (Container Boundaries)
- **교훈:** 서비스를 컨테이너로 나누면 (1) `localhost`는 컨테이너 자신을 가리키고, (2) 이미지에 복사한 `models/`·룰 파일은 서비스마다 별개의 복사본이 됨. API에서 재학습/룰 편집을 해도 Consumer에는 반영되지 않음.
- **해결:** DB는 `host.docker.internal` 등 실제 주소를 환경변수로 주입, 모델·룰은 named volume으로 공유하고 변경은 Redis Pub/Sub(`reload`, `reload_rules`)으로 전파.

## 18. 결측을 0으로 채우면 안 되는 센서 데이터 (Missing != Zero)
- **교훈:** 광파워 0 dBm, 트래픽 0은 유효한 측정값임. Producer에서 `fillna(0)`하면 '장비에 해당 데이터가 없는 포트'가 이상으로 오인됨.
- **해결:** 결측은 null로 전달하고 전처리(보간 1개 한정, NaN 윈도우 제거)에서 처리. 전부 null인 컬럼은 object 타입이 되므로 수신 측에서 `pd.to_numeric(errors='coerce')` 필수.

## 19. 모니터링 지표와 기준값은 같은 지표여야 함 (Drift Baseline)
- **교훈:** 드리프트 기준을 `val_loss`(전체 타임스텝 평균)로 두고 실시간 `score`(마지막 시점 MSE)와 비교하면 단위가 달라 감지가 사실상 무력화됨.
- **해결:** 학습 시 추론과 동일한 지표의 평균을 `baseline_mse`로 저장.

## 20. 설정에서 동적 계산으로 바뀐 값은 테스트도 함께 갱신 (Stale Tests)
- **교훈:** `input_dim`이 config 상수에서 파생 변수 기반 동적 계산으로, `diagnose()`가 문자열에서 튜플 반환으로 바뀌었는데 테스트가 갱신되지 않아 10건이 실패한 채 방치됨. 회귀 방어망이 사실상 꺼져 있었음.
- **원칙:** 인터페이스/상수 변경 시 해당 테스트를 같은 작업에서 갱신하고, 큰 작업 종료 시 전체 pytest를 반드시 실행. 룰 문구처럼 사람이 수정하는 값은 문구가 아닌 구조로 검증.

## 21. 스트림(건별) 처리에서는 '목록에 없음 = 복구'로 판단하지 않는다 (Per-Entity State)
- **교훈:** 배치 시절의 "이번 결과에 없는 알람은 복구"라는 로직을 Kafka 건별 웹훅에 그대로 두면, 다른 포트의 메시지 한 건이 무관한 포트의 알람을 해제함. 구조(배치→스트림)가 바뀌면 상태 전이 규칙도 다시 검증해야 함.
- **해결:** 알람 상태 전이는 payload에 포함된 개체(포트)만으로 판단(`alarm_tracker.plan_alarm_events`), 복구를 알리려면 송신 측(Consumer)이 직전 상태를 기억해 복구 시에도 전달.

## 22. 평가용 정답지를 학습 파이프라인에서 쓰지 않는다 (No Label Leakage)
- **교훈:** 학습 데이터 정제에 평가용 정답지(`eval_dataset.csv`)를 암묵적으로 읽으면 검증 수치가 부풀고, 운영(정답지 없음)과 동작이 달라짐. 운영 코드가 도구/시뮬레이터 산출물 경로에 의존하는 것도 결합도 문제.
- **원칙:** 제외 구간 등은 호출자가 명시적으로 주입(`exclude_path`)하고 기본은 '없음'. 평가 지표는 관대한 정답 창 + 자명한 베이스라인(항상 알람/고정 임계)과 함께 보고.
