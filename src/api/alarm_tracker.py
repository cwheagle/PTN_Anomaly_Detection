"""
알람 발생/해제 이벤트 산출 (순수 로직, DB/네트워크 의존 없음)

Consumer는 포트 1건씩 결과를 웹훅으로 보낸다. 따라서 "이번 payload에 없는 포트 = 복구"로
간주하면 안 되고, payload에 **포함된 포트**의 상태만으로 해당 포트의 발생/해제를 판단해야 한다.
"""

ALARM_LABEL = 'CRITICAL'


def plan_alarm_events(rows: list, active: dict) -> list:
    """
    Args:
        rows: 웹훅 payload 행 목록 (occur_date, ip_addr, cid, lid, alarm_label, anomaly_reason ...)
        active: 현재 활성 알람 상태 {(ip, cid, lid): occur_date_str}  (이 함수는 변경하지 않음)

    Returns:
        [(event_dict, key, occur_date_str), ...]
        - ALARM: 해당 포트가 CRITICAL이고, 같은 발생 시점으로 이미 보고된 적이 없을 때
        - CLEAR: 해당 포트가 활성 알람 상태였는데 이번 행이 CRITICAL이 아닐 때
        상태 갱신(발송 성공 시에만 active 반영)은 호출자가 수행한다.
    """
    events = []
    pending = dict(active)  # 같은 payload 안의 중복 행 처리를 위한 사본

    for row in rows:
        key = (row['ip_addr'], int(row['cid']), int(row['lid']))
        occur = str(row['occur_date'])

        if row.get('alarm_label') == ALARM_LABEL:
            if pending.get(key) == occur:
                continue  # 동일 포트·동일 시점 중복 발송 방지
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
