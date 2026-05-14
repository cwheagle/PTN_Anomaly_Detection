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
    "is_optical_anomaly": 0
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

### 2.3. 스케줄러 상태 및 제어
추론 엔진의 가동 상태를 조회하거나 제어합니다.

- **URL**: `/api/scheduler/status`
- **Method**: `GET` (상태 조회) / `POST` (제어 명령)
- **GET Response**:
```json
{
  "status": "running", // 또는 "stopped"
  "next_run_time": "2026-05-14 10:30:00"
}
```
- **POST Body**:
```json
{ 
  "action": "start"  // "start", "stop"
}
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

#### 2.4.2 추론 설정 업데이트 (재학습 없음)
- **Endpoint**: `POST /api/model/inference-config`
- **Description**: 모델 재학습 없이 임계값이나 감도 설정만 즉시 업데이트합니다.
- **Query Params**: `ft=traffic|optical`
- **Body**:
```json
{
  "threshold": 0.15,
  "slope_threshold": 2.0
}
```

#### 2.4.3 모델 학습 실행
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

#### 2.4.4 모델 학습 중지
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

---
*최종 업데이트: 2026-05-14*
