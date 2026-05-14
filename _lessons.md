# PTN 이상탐지 프로젝트 학습 레슨 (_lessons.md)

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

---
*최종 업데이트: 2026-05-14*
