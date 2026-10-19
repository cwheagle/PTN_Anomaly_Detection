"""U3: 이름 조회 규칙(7.2)·표 없는 모델(7.3)·타겟 매핑 로더(7.4)"""
import pandas as pd
import pytest

from validation.realdata.alarm_names import (AlarmNameError, AlarmNames, TargetMap, UNCLASSIFIED, DEFAULT_TARGET_MAP,
                                              load_override)


def names(**kw):
    t9 = [f"N{i}" for i in range(256)]
    t9[233], t9[232], t9[231], t9[100], t9[255] = "HI-PWR", "HI-PWR3", "HI-PWR2", "", ""
    data = {"code_offset": 367, "sys_id": {"31": "CE9001", "10": "CE9002", "27": "CE9003", "99": "CE9004"},
            "tables": {"CE9001": t9, "CE9004": ["a"]}, "table_alias": {"CE9002": "CE9001"},
            "models_without_table": ["CE9003"]}
    return AlarmNames(data, **kw)


def test_normal_lookup_and_alias():
    n = names()
    assert n.resolve(31, 600) == ("HI-PWR", "ok", "CE9001", 233)
    assert n.resolve(31, 599)[0] == "HI-PWR3" and n.resolve(31, 598)[0] == "HI-PWR2"
    assert n.resolve(10, 600)[:2] == ("HI-PWR", "ok")                       # 별칭 표 사용


def test_status_table_rules():
    n = names()
    assert n.resolve(31, 367 + 256)[:2] == ("ALM-ID-256", "out_of_range")
    assert n.resolve(31, 100)[:2] == ("ALM-ID--267", "out_of_range")        # 음수 code
    assert n.resolve(31, 367 + 100)[:2] == ("ALM-EMPTY-100", "empty")
    assert n.resolve(31, 367 + 255)[1] == "empty"
    assert n.resolve(77, 600)[:2] == ("UNKNOWN-SYS-77-600", "unknown_model")
    assert n.resolve(27, 600)[:2] == ("NOTABLE-CE9003-600", "no_table")      # 다른 모델 표를 빌려 쓰지 않음
    assert n.resolve(None, 600)[1] == "unknown_model" and n.resolve(31, "x")[1] == "unknown_model"
    assert n.resolve(31, 367 + 0)[0] == "N0"


def test_display_format_setting():
    assert names(display_fmt="X{code}").resolve(31, 999)[0] == "X632"


def test_override_applies_with_distinct_status(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("model,alarm_type,alarm_name\nCE9003,600,MY-NAME\nCE9001,367,FORCED\n", encoding="utf-8")
    n = names(override=load_override(str(p)))
    assert n.resolve(27, 600)[:2] == ("MY-NAME", "override") and n.resolve(31, 367)[:2] == ("FORCED", "override")
    assert n.resolve(31, 600)[1] == "ok"


def test_resolve_frame_columns_and_dtypes():
    df = pd.DataFrame({"system_type": [31, 27, 31], "alarm_type": [600, 600, 598], "x": [1, 2, 3]})
    out = names().resolve_frame(df)
    assert list(out["alarm_name"]) == ["HI-PWR", "NOTABLE-CE9003-600", "HI-PWR2"]
    assert list(out["name_status"]) == ["ok", "no_table", "ok"] and "model" in out and "code" in out
    assert "alarm_name" not in df.columns                               # 입력 불변
    assert names().resolve_frame(df.iloc[0:0]).empty


# ---- 타겟 매핑 ----------------------------------------------------------------------
def rows(*items):
    return [dict(zip(("pattern", "target_type", "grade", "source"), i)) for i in items]


def test_exact_beats_glob_and_two_rows_for_one_name():
    tm = TargetMap(rows(("ENC-*", "제외", "사용안함", "D"), ("ENC-LINK-FAIL", "트래픽단절", "보조", "B"),
                        ("OPT-LOS", "트래픽단절", "1차", "B"), ("OPT-LOS", "광관찰", "관찰참고", "C")))
    assert tm.classify("ENC-LINK-FAIL") == [("트래픽단절", "보조", "B")]          # 정확한 이름이 글롭을 대체
    assert tm.classify("ENC-KEY-FAIL") == [("제외", "사용안함", "D")]
    assert [t for t, _, _ in tm.classify("OPT-LOS")] == ["트래픽단절", "광관찰"]
    assert tm.classify("NOPE") == [(UNCLASSIFIED, "", "")]


def test_two_globs_on_one_name_is_error_at_load_validation():
    tm = TargetMap(rows(("FAN-*", "제외", "사용안함", "D"), ("*-FAIL", "제외", "사용안함", "D")))
    with pytest.raises(AlarmNameError, match="글롭이 둘 이상"):
        tm.validate_names(["FAN-1-FAIL"])
    tm.validate_names(["FAN-1"])                                         # 하나만 걸리면 통과
    with pytest.raises(AlarmNameError):
        tm.classify("FAN-1-FAIL")
    tm2 = TargetMap(rows(("FAN-*", "제외", "사용안함", "D"), ("*-FAIL", "제외", "사용안함", "D"), ("FAN-1-FAIL", "제외", "보조", "D")))
    tm2.validate_names(["FAN-1-FAIL"])                                   # 정확한 이름이 있으면 모호하지 않음


def test_invalid_rows_rejected():
    with pytest.raises(AlarmNameError):
        TargetMap(rows(("A", "엉뚱", "1차", "A")))
    with pytest.raises(AlarmNameError):
        TargetMap(rows(("A", "제외", "엉뚱", "A")))
    with pytest.raises(AlarmNameError, match="중복"):
        TargetMap(rows(("A", "제외", "보조", "A"), ("A", "제외", "보조", "A")))


def test_classify_frame_only_classifies_ok_or_override_names():
    tm = TargetMap(rows(("HI-PWR", "광관찰", "관찰참고", "C")))
    df = pd.DataFrame({"alarm_name": ["HI-PWR", "HI-PWR", "ALM-ID-1"], "name_status": ["ok", "no_table", "ok"]})
    out = tm.classify_frame(df)
    assert list(out["target_types"]) == ["광관찰", UNCLASSIFIED, UNCLASSIFIED]


def test_packaged_map_loads_and_has_documented_rows():
    tm = TargetMap.load(DEFAULT_TARGET_MAP)
    assert tm.classify("ETH-ERR") == [("CRC증가", "1차", "A")]
    assert {t for t, _, _ in tm.classify("OPT-LOS")} == {"트래픽단절", "광관찰"}
    assert tm.classify("ETH-NO-RX-TRAFFIC")[0][:2] == ("트래픽단절", "1차")
    assert tm.classify("LINK-FAIL-MANUAL")[0][:2] == ("해당없음", "사용안함")
    assert tm.classify("RMON-ALARM")[0][1] == "사용안함"                  # 5.1 규칙 2: 정답으로 쓰지 않음
    assert tm.classify("ENC-LINK-FAIL")[0][:2] == ("트래픽단절", "보조") and tm.classify("ENC-GMAC-ERR")[0][0] == "제외"
    assert tm.classify("FAN-A-1-FAIL")[0][0] == "제외" and tm.classify("OPT-PWR-LOW-3")[0][0] == "광관찰"
    assert tm.classify("SOMETHING-NEW") == [(UNCLASSIFIED, "", "")]


def test_packaged_map_has_no_glob_ambiguity_on_real_names_if_available():
    import os
    from validation.realdata.alarm_names import DEFAULT_NAMES_PATH
    if not os.path.exists(DEFAULT_NAMES_PATH):
        pytest.skip("validation/runs/ref/alarm_names.json 없음 (변환기로 생성, git 제외)")
    TargetMap.load(DEFAULT_TARGET_MAP, known_names=AlarmNames.load().all_names())
