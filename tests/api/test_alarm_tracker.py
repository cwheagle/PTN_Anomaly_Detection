"""
알람 발생/해제 검증: 포트 단위로 판단 (다른 포트의 웹훅이 이 포트의 알람을 해제하면 안 됨)
"""
from src.api.alarm_tracker import plan_alarm_events



# ──────────────────────────────────────────────
# 1. 알람 발생/해제 (포트 단위)
# ──────────────────────────────────────────────

def _row(ip, label, occur='2026-10-01 10:00:00', cid=1, lid=1):
    return {'occur_date': occur, 'ip_addr': ip, 'cid': cid, 'lid': lid,
            'alarm_label': label, 'anomaly_reason': 'reason'}


def _apply(events, active):
    """호출자(alarm_callback)와 동일하게 상태 반영"""
    for ev, key, occur in events:
        if ev['type'] == 'ALARM':
            active[key] = occur
        else:
            active.pop(key, None)


def test_other_port_webhook_does_not_clear_active_alarm():
    """회귀: 포트 B의 웹훅(1건 payload)이 포트 A의 알람을 CLEAR 하던 버그"""
    active = {}
    _apply(plan_alarm_events([_row('A', 'CRITICAL')], active), active)
    events = plan_alarm_events([_row('B', 'CRITICAL')], active)
    _apply(events, active)

    assert [e['type'] for e, _, _ in events] == ['ALARM']          # CLEAR 없음
    assert set(k[0] for k in active) == {'A', 'B'}


def test_port_recovery_emits_clear_only_for_that_port():
    active = {('A', 1, 1): 't0', ('B', 1, 1): 't0'}

    events = plan_alarm_events([_row('A', 'NORMAL', occur='t1')], active)

    assert [(e['type'], e['ip_addr']) for e, _, _ in events] == [('CLEAR', 'A')]
    _apply(events, active)
    assert set(k[0] for k in active) == {'B'}


def test_duplicate_alarm_same_occur_date_is_suppressed():
    active = {('A', 1, 1): '2026-10-01 10:00:00'}
    assert plan_alarm_events([_row('A', 'CRITICAL')], active) == []


def test_new_occur_date_re_alarms():
    active = {('A', 1, 1): '2026-10-01 10:00:00'}
    events = plan_alarm_events([_row('A', 'CRITICAL', occur='2026-10-01 10:15:00')], active)
    assert [e['type'] for e, _, _ in events] == ['ALARM']


def test_non_critical_without_active_alarm_emits_nothing():
    assert plan_alarm_events([_row('A', 'MAJOR'), _row('B', 'NORMAL')], {}) == []


def test_state_not_mutated_by_planning():
    """발송 성공 시에만 상태를 반영하므로, 계획 단계에서는 active를 바꾸지 않아야 함"""
    active = {('A', 1, 1): 't0'}
    plan_alarm_events([_row('A', 'NORMAL'), _row('B', 'CRITICAL')], active)
    assert active == {('A', 1, 1): 't0'}
