# PTN Anomaly Detection API Specification (v1.0)

본 문서는 PTN EMS 이상탐지 엔진 연동을 위한 REST API 및 SSE 규격을 정의합니다.

## 1. Global Settings
- **Base URL**: `http://localhost:8000`
- **Versioning**: HTTP Response Header `X-API-Version: 1.0` 포함

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
    "occur_date": "2026-05-11 10:15:00",
    "ip_addr": "192.168.99.226",
    "slot_id": 1,
    "port_id": 5,
    "severity": 95.5,
    "alarm_level": 3,
    "alarm_label": "CRITICAL",
    "slope": 1.5,
    "slope_label": "RISING",
    "ttf_minutes": 30.0,
    "expected_fatal_time": "2026-05-11 10:45:00",
    "anomaly_reason": "T:Traffic Anomaly (TX:1200, RX:0)"
  }
]
```

### 2.2. 스케줄러 상태 및 제어
추론 엔진의 가동 상태를 조회하거나 제어합니다.

- **URL**: `/api/scheduler/status`
- **Method**: `GET` (상태 조회) / `POST` (제어 명령)
- **GET Response**:
```json
{
  "status": "running", // 또는 "stopped"
  "next_run_time": "2026-05-11 10:30:00"
}
```
- **POST Body**:
```json
{ 
  "action": "start"  // "start", "stop"
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
  "event_time": "2026-05-11 10:15:05",
  "ip_addr": "192.168.99.226",
  "slot_id": 1,
  "port_id": 5,
  "severity": "CRITICAL",
  "reason": "T:Traffic (TX:24702658, RX:24702607)",
}
```

---
*최종 업데이트: 2026-05-11*
