# PTN / 광 전송 장비 도메인 RCA 룰셋 (예지 정비 최종안)

본 문서는 트래픽(Traffic) 및 광 파워(Optical) 데이터를 활용한 AI 기반 장애 예측 솔루션의 핵심 룰셋입니다.
AI가 도출한 **'기여도(Contribution)'**와 **'추세 기울기(Slope)'**를 활용하여 하드웨어 장애(LOS) 도달 전에 선제적으로 대응하는 것을 목표로 합니다.

## 1. Traffic Track (이더넷 / 데이터 계층)
데이터 링크 계층의 병목, 에러, 외부 공격 등을 탐지합니다.

1. **L2 Loop / Broadcast Storm (Priority: 100)**
   - **조건**: `rx_packet` 기여도 ≥ 35%, `tx_packet` 기여도 ≥ 35%, `rx_packet_ratio` ≥ 1.5, `tx_packet_ratio` ≥ 1.5 (JSON: `min_rx_packet_ratio: 1.5, min_tx_packet_ratio: 1.5`)
   - **Diagnosis**: 양방향 트래픽 동시 급증 (L2 브로드캐스트 스톰 전조 증상 또는 루핑 발생 의심)
   - **Action**: 하위 스위치 루프(Loop) 구성 및 MAC Address 플래핑 이력 긴급 확인
2. **DDoS / 마이크로버스트 폭주 (Priority: 80)**
   - **조건**: `rx_packet` 기여도 ≥ 80%, `rx_packet_ratio` ≥ 2.0 (JSON: `min_rx_packet_ratio: 2.0`)
   - **Diagnosis**: 단방향 수신 트래픽 비정상 폭주. 과다 트래픽 유입(DDoS/플러딩) 공격 의심.
   - **Action**: 유입 트래픽 NetFlow 분석(출발지 IP 추적) 및 대역폭 제한(QoS) 적용
3. **상위 네트워크 논리적 완전 단절 (Silent Drop) (Priority: 70)**
   - **조건**: `rx_packet` 기여도 ≥ 70%, `rx_packet` == 0 (JSON: `max_rx_packet: 0`)
   - **Diagnosis**: 물리 연결은 정상이나 논리적 수신 트래픽 100% 단절 (블랙홀 발생)
   - **Action**: 상위 라우터 논리적 포트 셧다운 여부 및 방화벽/ACL 차단 정책 즉시 확인
4. **비트 에러(CRC) 폭주 징후 (Priority: 60)**
   - **조건**: `error_packet` 기여도 ≥ 60%, `error_packet_ratio` ≥ 2.0 (JSON: `min_error_packet_ratio: 2.0`)
   - **Diagnosis**: 대량의 비트 에러 폭주 (심각한 패킷 로스 및 통신 품질 저하 발생 중)
   - **Action**: 양단 포트 Speed/Duplex 미스매치 확인 또는 SFP 모듈 불량으로 인한 긴급 교체 요망
5. **[예지정비] 트래픽 급감 전조 징후 (Priority: 50)**
   - **조건**: `rx_packet` 기여도 ≥ 70%, `rx_packet` > 0, `rx_packet_ratio` ≤ 0.5 (JSON: `max_rx_packet_ratio: 0.5, min_rx_packet: 1`)
   - **Diagnosis**: 수신 트래픽 비정상적 급감 추세 (상위 네트워크 병목 또는 라우팅 플래핑 의심)
   - **Action**: 업스트림(상위) 노드의 라우팅 테이블 및 네이버(Neighbor) 세션 안정성 예방 점검
6. **[예지정비] 비트 에러 점진적 증가 (Priority: 40)**
   - **조건**: `error_packet` 기여도 ≥ 60%, `error_packet_trend_slope` > 0 (JSON: `min_error_trend_slope: 0.1`)
   - **Diagnosis**: 이더넷 프레임 에러(FCS/CRC) 점진적 증가 징후 포착
   - **Action**: 케이블 차폐 상태 점검 및 SFP 전기 포트 이물질 예방 점검 요망

## 2. Optical Track (광 / 물리 계층)
물리 계층의 하드웨어(SFP 모듈) 수명, 선로 오염 및 물리적 단선을 탐지합니다.

1. **물리 선로 완전 단선 (Hard LOS) (Priority: 100)**
   - **조건**: `rx_avg_power` 기여도 ≥ 40%, `rx_avg_power` ≤ -24 dBm (JSON: `max_rx_avg_power: -24`)
   - **Diagnosis**: 수신 광전력 한계치 미달 (LOS 발생 확정 - 선로 완전 단선)
   - **Action**: 현장 선로 복구반 즉시 출동 및 우회(Protection) 경로 절체 상태 확인
2. **모듈 자체 하드웨어 결함 (Priority: 80)**
   - **조건**: `rx_avg_power` 기여도 ≥ 40%, `tx_avg_power` 기여도 ≥ 40%, `rx_avg_power` ≤ -24 dBm, `tx_avg_power` ≤ -10 dBm
   - **Diagnosis**: 송수신 광 전력 동시 급감 (광 모듈 자체 하드웨어 Fault 또는 슬롯 전원 이상)
   - **Action**: 광 모듈 재장착(Re-seat) 및 교체, 슬롯 불량 여부 점검
3. **[예지정비] 광 수신 신호 급격한 열화 (Priority: 60)**
   - **조건**: `rx_avg_power` 기여도 ≥ 60%, `rx_power_trend_slope` ≤ -1.0 (JSON: `max_rx_power_trend_slope: -1.0`)
   - **Diagnosis**: 광 수신 전력 급격한 저하 (패치코드 꺾임, 무거운 물체에 눌림 등 단기 내 LOS 위험)
   - **Action**: 통신실 내 광 패치코드 결선 상태 및 굴곡(Macrobending) 즉시 확인
4. **[예지정비] 광 송신 레이저 급사 징후 (Priority: 60)**
   - **조건**: `tx_avg_power` 기여도 ≥ 60%, `tx_power_trend_slope` ≤ -1.0 (JSON: `max_tx_power_trend_slope: -1.0`)
   - **Diagnosis**: 광 송신 전력 급격한 저하 (송신 레이저 다이오드 불량 또는 과열 의심)
   - **Action**: 광 모듈 즉시 교체 준비 및 라인카드 온도 점검
5. **[예지정비] 송수신 모듈 점진적 에이징 (Priority: 50)**
   - **조건**: `tx_avg_power` (또는 rx_avg_power) 기여도 ≥ 60%, 해당 지표의 `power_trend_slope` < 0 (JSON: `max_rx_power_trend_slope: -0.1` 등)
   - **Diagnosis**: 광 전력 점진적 저하 (레이저 바이어스 전류 노후화 또는 커넥터 오염 진행 중)
   - **Action**: 다음 계획 예방 정비(PM) 시 커넥터 클리닝 및 모듈 교체 스케줄 수립

### 2.3. Integrated Track (물리+데이터 복합 장애)
물리 계층의 이상이 논리 계층에 미치는 파급 효과를 분석하여 근본 원인을 특정합니다.

1. **장비 셧다운 및 하드웨어 패닉 (Priority: 100)**
   - **조건**: `rx_packet`, `tx_packet`, `rx_avg_power`, `tx_avg_power` 기여도 각각 ≥ 15%, `traffic_severity` ≥ 80, `optical_severity` ≥ 80
   - **Diagnosis**: 물리 및 데이터 계층 전 지표 동시 치명적 붕괴 (포트/라인카드 패닉 임박)
   - **Action**: 해당 라인카드 상태 및 듀얼 전원(Redundancy) 긴급 점검. 즉각적인 시스템 백업 준비
2. **물리 선로 단선에 의한 수신 트래픽 100% 단절 (Priority: 90)**
   - **조건**: `rx_avg_power` 기여도 ≥ 25%, `rx_packet` 기여도 ≥ 25%, `rx_avg_power` ≤ -24 dBm, `rx_packet` == 0
   - **Diagnosis**: 광 선로 완전 단선으로 인한 수신 트래픽 100% 단절 (원인: 물리 선로)
   - **Action**: 상위 라우팅 점검 불필요. 즉시 현장 선로 복구반 출동
3. **[예지정비] 신호 미약으로 인한 데이터 손실 (Priority: 80)**
   - **조건**: `rx_avg_power` 기여도 ≥ 25%, `error_packet` 기여도 ≥ 25%, `rx_power_trend_slope` < 0, `error_packet_trend_slope` > 0
   - **Diagnosis**: 수신 광전력 저하로 인한 비트 오류(BER) 급증 진행 중
   - **Action**: 단순 트래픽 장애 아님. 광 선로 로스(OTDR) 측정 및 커넥터 클리닝 실시
4. **[예지정비] 송신 열화로 인한 병목 현상 (Priority: 70)**
   - **조건**: `tx_avg_power` 기여도 ≥ 25%, `tx_packet` 기여도 ≥ 25%, `tx_power_trend_slope` < 0, `tx_packet_ratio` ≤ 0.5
   - **Diagnosis**: 송신 광전력 저하로 인한 송신 트래픽 병목/드랍 발생
   - **Action**: 내 장비 송신 모듈 상태 점검 및 대향국(Remote) 수신 에러 상태 크로스 체크

---
**비고:** 
명시된 룰에 해당하지 않는 복합 징후(사각지대)는 `DefaultRuleSet`으로 넘어가 가장 높은 기여도를 가진 Feature를 기반으로 기본 이상 진단을 내립니다.
