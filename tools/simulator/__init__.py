"""
Simulator Package

- 테스트 데이터 생성 시뮬레이터.
- generate_history.py : 과거 데이터 주입
  - 현재 시간 기준 과거 30일치의 데이터를 자동으로 생성하여 DB에 주입.
  - 시간: 15분 단위
  - 데이터: 시스템 10 * 포트 10 = 총 100개
  - 테이블: cowptn_noti_pm_YYYY_MM_DD_hh (매 시간마다 1개씩, 30일치)
    - 컬럼: occur_date, ip_addr, cid, lid, signal_type, es, ses, bbe_in_error
      ** es <- tx_packet,   ses <- rx_packet,   bbe_in_error <- error_packet
  - 테이블: cowptn_noti_pm_optic_power_YYYY_MM_DD_hh (현재 시간 기준)
    - 컬럼: occur_date, ip_addr, cid, lid, tx_avg_power, rx_avg_power
- realtime_injector.py : 
  - 15분마다 동작하는 스케줄러를 가동하여 실시간 데이터 주입
  - 시간: 15분 단위
  - 데이터: 시스템 10 * 포트 10 = 총 100개
  - 테이블: cowptn_noti_pm_YYYY_MM_DD_hh (현재 시간 기준)
    - 컬럼: occur_date, ip_addr, cid, lid, signal_type, es, ses, bbe_in_error
      ** es <- tx_packet,   ses <- rx_packet,   bbe_in_error <- error_packet
  - 테이블: cowptn_noti_pm_optic_power_YYYY_MM_DD_hh (현재 시간 기준)
    - 컬럼: occur_date, ip_addr, cid, lid, tx_avg_power, rx_avg_power
"""
