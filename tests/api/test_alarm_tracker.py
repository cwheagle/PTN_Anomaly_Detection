"""
알람 발생/해제 검증: 포트 단위로 판단 (다른 포트의 웹훅이 이 포트의 알람을 해제하면 안 됨)
"""
import pandas as pd

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


# --- 재알림 간격 (ALARM_RENOTIFY_MINUTES) ---

def _steps(n, start='2026-10-01 10:00:00', label='CRITICAL', ip='A'):
    """15분 간격 n 스텝의 행 목록"""
    t0 = pd.Timestamp(start)
    return [_row(ip, label, occur=str(t0 + pd.Timedelta(minutes=15 * i))) for i in range(n)]


def _replay(rows, renotify, active=None):
    """Consumer 가 포트 1건씩 웹훅을 보내는 방식으로 순차 재생하며 ALARM/CLEAR 건수를 센다"""
    active = {} if active is None else active
    counts = {'ALARM': 0, 'CLEAR': 0}
    for r in rows:
        ev = plan_alarm_events([r], active, renotify)
        _apply(ev, active)
        for e, _, _ in ev:
            counts[e['type']] += 1
    return counts, active


def test_default_interval_keeps_existing_behavior_alarm_every_step():
    """기본값(15분)은 기존 동작: 지속되는 CRITICAL 10스텝 -> ALARM 10건"""
    assert plan_alarm_events.__defaults__ == (15,)
    counts, _ = _replay(_steps(10), renotify=15)
    assert counts['ALARM'] == 10


def test_renotify_60_alarms_on_entry_then_hourly():
    """60분: 진입(0) + 4스텝째 + 8스텝째 = 3건"""
    counts, _ = _replay(_steps(10), renotify=60)
    assert counts['ALARM'] == 3


def test_renotify_zero_alarms_only_on_entry():
    counts, _ = _replay(_steps(10), renotify=0)
    assert counts['ALARM'] == 1


def test_clear_is_unaffected_by_renotify_setting():
    """재알림을 줄여도(조용히 지속 중이어도) 복구 시점의 CLEAR 는 항상 전달되어야 함"""
    rows = _steps(6) + _steps(1, start='2026-10-01 11:30:00', label='NORMAL')
    for renotify in (15, 60, 0):
        counts, active = _replay(rows, renotify)
        assert counts['CLEAR'] == 1 and active == {}, f"renotify={renotify}"


def test_silent_steps_do_not_update_last_notified_time():
    """재알림 간격 계산은 '마지막으로 ALARM 을 보낸 시각' 기준이어야 함 (조용한 스텝이 기준을 밀어내면 영원히 재알림이 안 감)"""
    _, active = _replay(_steps(3), renotify=60)       # 진입 1건, 이후 2스텝은 조용함
    assert active == {('A', 1, 1): '2026-10-01 10:00:00'}


def test_escalation_to_critical_always_alarms_even_with_zero_interval():
    """MAJOR 등 비CRITICAL -> CRITICAL 진입은 새 인시던트이므로 간격 설정과 무관하게 즉시 ALARM"""
    rows = _steps(3, label='MAJOR') + _steps(1, start='2026-10-01 10:45:00', label='CRITICAL')
    counts, _ = _replay(rows, renotify=0)
    assert counts['ALARM'] == 1


def test_duplicate_same_timestamp_never_realarms():
    active = {('A', 1, 1): '2026-10-01 10:00:00'}
    assert plan_alarm_events([_row('A', 'CRITICAL', occur='2026-10-01 10:00:00')], active, 15) == []
