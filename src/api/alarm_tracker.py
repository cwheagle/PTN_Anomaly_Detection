"""
알람 발생/해제 이벤트 산출 (순수 로직, DB/네트워크 의존 없음)

Consumer는 포트 1건씩 결과를 웹훅으로 보낸다. 따라서 "이번 payload에 없는 포트 = 복구"로
간주하면 안 되고, payload에 **포함된 포트**의 상태만으로 해당 포트의 발생/해제를 판단해야 한다.

재알림 간격(renotify_minutes): 같은 포트가 CRITICAL 로 계속 지속될 때 ALARM 을 다시 보내는 간격.
  - 15 (기본): 스텝(15분)마다 매번 ALARM  (기존 동작)
  - 60 등   : 진입 시 1회 + 해당 분마다 재알림
  - 0       : 진입 시 1회만
CLEAR 는 재알림 설정과 무관하게 복구 시점에 항상 전달된다.
"""
import pandas as pd

ALARM_LABEL = 'CRITICAL'
DEFAULT_RENOTIFY_MINUTES = 15


def plan_alarm_events(rows: list, active: dict, renotify_minutes: int = DEFAULT_RENOTIFY_MINUTES) -> list:
    """
    Args:
        rows: 웹훅 payload 행 목록 (occur_date, ip_addr, cid, lid, alarm_label, anomaly_reason ...)
        active: 현재 활성 알람 상태 {(ip, cid, lid): 마지막으로 ALARM 을 보낸 occur_date 문자열}
                (이 함수는 변경하지 않음)
        renotify_minutes: 지속 중인 CRITICAL 의 재알림 간격(분). 0 이면 진입 시 1회만.

    Returns:
        [(event_dict, key, occur_date_str), ...]
        - ALARM: 포트가 CRITICAL 이고 (활성 상태가 아니거나, 재알림 간격이 지났을 때)
        - CLEAR: 포트가 활성 알람 상태였는데 이번 행이 CRITICAL 이 아닐 때
        상태 갱신(발송 성공 시에만 active 반영)은 호출자가 수행한다.
    """
    events = []
    pending = dict(active)  # 같은 payload 안의 중복 행 처리를 위한 사본

    for row in rows:
        key = (row['ip_addr'], int(row['cid']), int(row['lid']))
        occur = str(row['occur_date'])

        if row.get('alarm_label') == ALARM_LABEL:
            last = pending.get(key)
            if last is not None:
                # 이미 활성: 재알림 간격이 지났을 때만 다시 보냄 (동일 시점 중복/역순 도착은 간격 미달로 걸러짐)
                if renotify_minutes <= 0:
                    continue
                elapsed = (pd.Timestamp(occur) - pd.Timestamp(last)).total_seconds() / 60.0
                if elapsed < renotify_minutes:
                    continue
            events.append(({
                "type": "ALARM",
                "event_time": occur,
                "ip_addr": key[0],
                "slot_id": key[1],
                "port_id": key[2],
                "severity": row['alarm_label'],
                "message": row.get('anomaly_reason', ''),
            }, key, occur))
            pending[key] = occur
        elif key in pending:
            events.append(({
                "type": "CLEAR",
                "event_time": occur,
                "ip_addr": key[0],
                "slot_id": key[1],
                "port_id": key[2],
                "message": "Alarm cleared",
            }, key, occur))
            del pending[key]

    return events
