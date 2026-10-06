# PTN Anomaly Detection API Specification (v1.0.0)

본 문서는 PTN EMS 이상탐지 엔진 연동을 위한 REST API 및 SSE 규격을 정의합니다.

## 1. Global Settings
- **Base URL**: `http://localhost:8000`
- **Versioning**: HTTP Response Header `X-API-Version: 1.0.0` 포함

## 2. REST API 엔드포인트

### 2.1. 이상 탐지 및 추세 데이터 조회
가장 최신 분석 주기(Batch)의 이상 탐지 결과 및 추세 데이터를 필터 조건에 따라 조회합니다.

- **URL**: `/api/anomalies`
- **Method**: `GET`
- **Query Parameters**:
    - `severity_min` (Optional): 최소 경보 등급 필터 (0: NORMAL, 1: MINOR, 2: MAJOR, 3: CRITICAL)
    - `severity_max` (Optional): 최대 경보 등급 필터 (0: NORMAL, 1: MINOR, 2: MAJOR, 3: CRITICAL)
    - `rising_only` (Optional): 점수 상승 추세(`RISING`)인 데이터만 필터링 (Default: `false`)
- **Response**:
```json
[
  {
    "occur_date": "2026-05-14 10:15:00",
    "ip_addr": "192.168.99.226",
    "slot_id": 1,
    "port_id": 5,
    "severity": 95.5,
    "alarm_level": 3,
    "alarm_label": "CRITICAL",
    "slope": 1.5,
    "slope_label": "RISING",
    "ttf_minutes": 30.0,
    "expected_fatal_time": "2026-05-14 10:45:00",
    "anomaly_reason": "Traffic (TX:1200, RX:0)",
    "is_traffic_anomaly": 1,
    "is_optical_anomaly": 0,
    "rca_diagnosis": "물리 계층 심각한 장애",
    "rca_action": "즉각적인 광 선로 단선 및 커넥터 파손 여부 점검",
    "feature_contribution": "{\"rx_avg_power\": 60.0, \"error_packet\": 40.0}"
  }
]
```

### 2.2. 특정 포트 과거 이력 조회
특정 포트의 시계열 성능 데이터와 이상 점수 이력을 조회합니다 (그래프용).

- **URL**: `/api/anomalies/history`
- **Method**: `GET`
- **Query Parameters**:
    - `ip_addr` (Required): 대상 장비 IP
    - `slot_id` (Required): 대상 슬롯 ID (CID)
    - `port_id` (Required): 대상 포트 ID (LID)
    - `days` (Optional): 조회 기간 (Default: `1`)
- **Response**:
```json
[
  {
    "occur_date": "2026-05-14 10:00:00",
    "tx_packet": 1200,
    "rx_packet": 1150,
    "error_packet": 0,
    "tx_avg_power": -15.5,
    "rx_avg_power": -18.2,
    "anomaly_score": 0.02,
    "threshold": 0.15,
    "severity": 15.2,
    "alarm_level": 3,
    "alarm_label": "CRITICAL",
    "anomaly_reason": "Optical (RX:-14.31, TX:-6.30)",
    "traffic_score": 0.0238122,
    "traffic_threshold": 0.73775,
    "traffic_severity": 15.2,
    "optical_score": 0.519406,
    "optical_threshold": 0.0123261,
    "optical_severity": 5.0,
    "is_traffic_anomaly": 0,
    "is_optical_anomaly": 1
  }
]
```


### 2.4. 모델 관리 및 훈련

#### 2.4.1 모델 상태 조회
- **Endpoint**: `GET /api/model/status`
- **Description**: 현재 학습된 모델의 유무, 마지막 학습 시간, 그리고 **현재 진행 중인 학습 상태(에포크, 손실률 등)** 및 설정을 반환합니다.
- **Note**: `inference_config` 및 `training_config`는 수정 가능한 항목들만 노출합니다.
- **Response**:
```json
{
  "traffic": {
    "exists": true,
    "last_trained": "2026-05-14 14:30:00",
    "samples_used": 15200,
    "training": {
      "is_training": true,
      "current_epoch": 45,
      "total_epochs": 100,
      "loss": 0.0024,
      "val_loss": 0.0031,
      "last_error": null,
      "success_msg": null
    },
    "inference_config": {
      "threshold": 0.1245,
      "slope_threshold": 1.5
    },
    "training_config": {
      "epochs": 100,
      "learning_rate": 0.001,
      "batch_size": 32,
      "threshold_percentile": 99.9,
      "patience": 10
    }
  }
}
```

#### 2.4.2 모델 학습 실행
- **Endpoint**: `POST /api/model/train`
- **Description**: 사용자가 지정한 훈련 파라미터와 데이터 기간을 사용하여 모델을 재학습합니다. (백그라운드 실행)
- **Query Params**: 
  - `ft`: 모델 타입 (traffic/optical)
  - `train_start` (Optional): 훈련 데이터 시작일 (YYYY-MM-DD)
  - `train_end` (Optional): 훈련 데이터 종료일 (YYYY-MM-DD)
  - `test_start` (Optional): 검증 데이터 시작일 (YYYY-MM-DD)
  - `test_end` (Optional): 검증 데이터 종료일 (YYYY-MM-DD)
- **Body (JSON)**:
```json
{
  "epochs": 150,
  "learning_rate": 0.0005,
  "batch_size": 64,
  "threshold_percentile": 99.5,
  "patience": 15
}
```

#### 2.4.3 모델 학습 중지
- **Endpoint**: `POST /api/model/train/stop`
- **Description**: 현재 실행 중인 모델 학습 작업을 즉시 중단합니다.
- **Query Params**: `ft=traffic|optical`
- **Response**:
```json
{
  "status": "success",
  "message": "Stop request sent to traffic trainer."
}
```

#### 2.4.4 모델 버전 조회
- **Endpoint**: `GET /api/model/versions`
- **Description**: 트랙별 모델 버전 목록과 상태를 반환합니다. 재학습 결과는 **후보(candidate)** 로 저장되며, 승격해야 활성(active)이 됩니다. 활성이었다가 교체된 버전은 `retired` 입니다.
- **Query Params**: `ft=traffic|optical` (생략 시 두 트랙 모두)
- **Response**:
```json
{
  "optical": {
    "active_version": "v1",
    "versions": [
      {"version": "v1", "status": "active", "trained_at": "2026-09-18 10:08:41", "threshold": 0.1704, "final_val_loss": 0.0175, "baseline_mse": null, "samples_used": null},
      {"version": "v2", "status": "candidate", "trained_at": "2026-10-01 15:09:47", "threshold": 8.6756, "final_val_loss": 0.6886, "baseline_mse": 0.61, "samples_used": 899}
    ]
  }
}
```

#### 2.4.5 모델 승격 (후보 → 활성)
- **Endpoint**: `POST /api/model/promote`
- **Description**: 지정한 버전을 활성화하고 Consumer 에 Redis 로 리로드를 알립니다. 후보가 현재 활성 모델과 크게 다르면(임계치 또는 검증 손실이 3배 초과/미만) **409 와 경고 목록**을 반환하며, 확인 후 `force=true` 로 다시 요청해야 합니다.
- **Query Params**: `ft=traffic|optical`, `version=v2`, `force=false`, `reason=manual`
    - `reason` (string, 선택, 기본 `manual`, 2026-10-07 추가): 승격 사유. 레지스트리(`<model_dir>/<ft>_registry.json`)의 활성화 이력 `history[].reason` 에 기록됩니다. 앞뒤 공백을 제거하며, 제거 후 빈 값이면 `manual` 로 기록합니다. 최대 100자(공백 제거 **전** 원본 기준), 초과 시 `422`. `force=true` 승격에도 같은 방식으로 기록됩니다. 응답 형식은 변경 없음(사유는 응답에 포함되지 않음).
    - 예: `POST /api/model/promote?ft=traffic&version=v2&force=true&reason=v1%20비정상%20기준(G2),%20P1-3%20V1~V4%20합격` (P1-3 설계서 `docs/design/p1_3_model_promotion.md` 7장 4단계의 "사유 기록"이 이 인자로 남음)
    - 이력에 남는 다른 사유 값: `rollback`(롤백, 2.4.6), `auto-gate`(드리프트 auto 모드 자동 승격), `trained`/`initial`(최초 학습), `legacy`(구 레지스트리 이관).
    - **한계**: 사용자가 `reason=rollback`(또는 `auto-gate` 등)을 직접 넣어도 막지 않으므로, 이력만으로는 실제 롤백·자동 승격과 구분할 수 없습니다.
- **Response (200)**: `{"status": "success", "track": "optical", "previous": "v1", "active": "v2", "warnings": []}`
- **Response (409, 경고)**:
```json
{"detail": {"message": "후보가 현재 활성 모델과 크게 다릅니다. 확인 후 force=true 로 다시 요청하세요.",
            "warnings": ["임계치가 현재 모델 대비 50.9배 (0.1704 -> 8.676)", "검증 손실이 현재 모델 대비 39.3배 (0.01753 -> 0.6886)"]}}
```
- **오류**: `404` 버전 없음 / `409` 이미 활성이거나 모델 파일 누락 / `422` 잘못된 트랙 또는 `reason` 100자 초과

#### 2.4.6 모델 롤백
- **Endpoint**: `POST /api/model/rollback`
- **Description**: 직전에 활성이었던 버전으로 되돌립니다(활성화 이력 기준, 파일이 남아 있는 버전만). 한 번 더 호출하면 다시 직전 활성으로 돌아갑니다. 활성화 이력의 사유는 `rollback` 으로 기록됩니다(사유 인자 없음).
- **Query Params**: `ft=traffic|optical`
- **Response**: `{"status": "success", "track": "optical", "previous": "v2", "active": "v1", "warnings": []}`
- **오류**: `409` 되돌릴 이전 활성 버전이 없음

> **재학습 동작 변경 (2026-10-01)**: `POST /api/model/train` 과 드리프트 자동 재학습의 결과는 더 이상 즉시 활성화되지 않고 후보로만 저장됩니다. (활성 모델이 아직 없는 최초 학습만 자동 활성화)

### 2.5. RCA 도메인 룰 관리 (Rule Engine)

#### 2.5.1. 등록된 도메인 룰 조회
- **Endpoint**: `GET /api/rca/rules`
- **Description**: 현재 등록된 RCA 룰 목록을 우선순위(Priority) 정렬 상태로 반환합니다.
- **Response**:
```json
{
  "count": 1,
  "rules": [
    {
      "id": "IN-001",
      "track": "integrated",
      "priority": 20,
      "contributions": {
        "error_packet": 60
      },
      "raw_conditions": {
        "max_rx_avg_power": -20
      },
      "diagnosis": "물리 계층 심각한 장애",
      "action": "즉각적인 광 선로 단선 점검"
    }
  ]
}
```

#### 2.5.2. 신규 도메인 룰 추가
- **Endpoint**: `POST /api/rca/rules`
- **Description**: Fully Structured JSON(기여도 및 원시 데이터 조건) 기반의 새 도메인 룰을 주입합니다. 즉시 평가 엔진에 반영됩니다.
- **Body Example**:
```json
{
  "id": "TR-001",
  "track": "traffic",
  "priority": 10,
  "contributions": {
    "error_packet": 60
  },
  "diagnosis": "CRC/비트 오류 의심 (에러 패킷 급증)",
  "action": "광 커넥터 청소 또는 케이블 점검 권고"
}
```

#### 2.5.3. 도메인 룰 삭제
- **Endpoint**: `DELETE /api/rca/rules/{rule_id}`
- **Description**: 지정된 ID의 룰을 메모리 및 파일에서 삭제합니다.

### 2.6. Data Drift & Auto-Retraining

#### 2.6.1. 드리프트 상태 조회
- **Endpoint**: `GET /api/drift/status`
- **Description**: 최근 수행된 데이터 드리프트(Data Drift) 감지 결과를 반환합니다. 24시간 단위의 평균 MSE와 베이스라인 비교값을 포함합니다.
- **Response**:
```json
{
  "status": "success",
  "last_result": {
    "traffic": {
      "mean_mse": 0.0035,
      "baseline_mse": 0.0024,
      "drift_ratio": 1.45,
      "status": "normal"
    },
    "optical": {
      "mean_mse": 0.0042,
      "baseline_mse": 0.0020,
      "drift_ratio": 2.10,
      "status": "drifted"
    },
    "drift_detected": true,
    "drifted_tracks": ["optical"],
    "auto_retrain_triggered": true
  }
}
```

#### 2.6.2. 수동 드리프트 검사 및 재학습 트리거
- **Endpoint**: `POST /api/drift/check`
- **Description**: 즉시 데이터 드리프트를 검사하고, 임계치 초과 시 백그라운드 재학습 파이프라인을 가동합니다.
- **Response**:
```json
{
  "traffic": {
    "mean_mse": 0.0035,
    "baseline_mse": 0.0024,
    "drift_ratio": 1.45,
    "status": "normal"
  },
  "optical": {
    "mean_mse": 0.0042,
    "baseline_mse": 0.0020,
    "drift_ratio": 2.10,
    "status": "drifted"
  },
  "drift_detected": true,
  "drifted_tracks": ["optical"],
  "auto_retrain_triggered": true
}
```

## 3. 실시간 알림 스트림 (SSE)
이상 발생 시 서버에서 클라이언트로 즉시 푸시 알림을 전달합니다.

- **URL**: `/api/stream/alarms`
- **Method**: `GET` (Headers: `Accept: text/event-stream`)
- **Event Type**: `alarm`
- **Data Example**:
```json
{
  "type": "ALARM", // 해제시 "CLEAR"
  "event_time": "2026-05-14 10:15:05",
  "ip_addr": "192.168.99.226",
  "slot_id": 1,
  "port_id": 5,
  "severity": "CRITICAL",
  "message": "Traffic (TX:24702658, RX:24702607)" // 해제시  "Alarm cleared"
}
```

### 2.7. 시스템 내부 통신 (Internal Webhook)
Kafka Consumer가 추론 결과를 바탕으로 실시간 알람을 발송하기 위해 API 서버에 호출하는 내부용 엔드포인트입니다. 외부 노출을 권장하지 않습니다.

- **URL**: `/api/internal/alarm`
- **Method**: `POST`
- **Body Example**:
```json
[
  {
    "occur_date": "2026-05-14 10:15:00",
    "ip_addr": "192.168.99.226",
    "cid": 1,
    "lid": 5,
    "alarm_label": "CRITICAL",
    "anomaly_reason": "Traffic (TX:1200, RX:0)"
  }
]
```

---
*최종 업데이트: 2026-09-21*
