"""U3: 알람 이름 표 변환기 (합성 C 소스 픽스처 — 실제 장비 소스는 저장소에 두지 않는다)"""
import json

import pytest

from validation.cli import convert_alarm_tables as cv
from validation.cli.convert_alarm_tables import ConvertError, convert, explain, main, parse_arrays, parse_sys_enum


def table_src(name, special=None, n=256, comment=True, ctype="const char *"):
    """name 표: 위치 i 의 기본 이름 'X<i>', special={위치: (이름, 주석값 또는 None)}."""
    special = special or {}
    lines = [f"{ctype}{name}[] =", "{"]
    for i in range(n):
        nm, ev = special.get(i, (f"X{i}", i))
        ev = i if ev is None else ev
        cm = f"   // SOMF_{name}_ID_{i}   = {ev:>6}, // 0x{ev:02x}" if comment else ""
        lines.append(f'    "{nm}"  ,{cm}')
    lines.append("};")
    return "\n".join(lines) + "\n"


FAKE_ENUM = "typedef enum { SYS_CE9001 = 31, SYS_CE9002, SYS_CE9003 = 40 } enum_sys_id; // 합성\n"
D6 = {231: ("HI-PWR2", None), 232: ("HI-PWR3", None), 233: ("HI-PWR", None), 255: ("", None)}


def test_d6_example_reproduced_with_synthetic_table():
    text = FAKE_ENUM + table_src("lpsztbl_CE9001_ALM_RSN", D6)
    res, warns, _ = convert([], texts=[text])
    t = res["tables"]["CE9001"]
    assert [t[600 - 367], t[599 - 367], t[598 - 367]] == ["HI-PWR", "HI-PWR3", "HI-PWR2"]
    assert res["sys_id"] == {"31": "CE9001", "32": "CE9002", "40": "CE9003"} and res["sys_id_source"] == "input_enum"
    assert "'HI-PWR'" in explain(res, ["31:600"])
    assert res["code_offset"] == 367 and len(res["source_sha256"]) == 1


def test_enum_positions_verified_and_mismatch_warned():
    text = table_src("lpsztbl_CE9001_ALM_RSN", {10: ("A", 12), 20: ("B", 20)})
    _, warns, stats = convert([], texts=[text])
    assert any("불일치 1건" in w and "위치 10" in w and "enum 12" in w for w in warns)
    assert stats["CE9001"]["enum_mismatch"] == 1


def test_hex_comment_mismatch_empty_duplicate_and_length_warned():
    lines = ['const char *lpsztbl_CE9001_ALM_RSN[] = {',
             '"A", // X = 0, // 0x00', '"A", // X = 1, // 0x09', '"", // X = 2, // 0x02', '};']
    _, warns, stats = convert([], texts=["\n".join(lines)])
    joined = "\n".join(warns)
    assert "주석의 10진/16진 값 불일치 1건" in joined and "빈 이름 1개" in joined and "중복 이름 1종" in joined
    assert "원소 수 3" in joined
    assert stats["CE9001"]["duplicate_names"] == {"A": [0, 1]}


def test_no_comment_entries_are_reported_not_verified():
    _, warns, _ = convert([], texts=[table_src("lpsztbl_CE9001_ALM_RSN", comment=False)])
    assert any("enum 주석 없는 원소 256개" in w for w in warns)


def test_adjacent_string_concat_escapes_and_comments():
    text = ('const char* lpsztbl_CE9001_ALM_RSN[] = { "AB" "CD" , /* blk , "x" */ "E\\"F" , // c , "Y"\n'
            ' "" };')
    arr = parse_arrays(text)["lpsztbl_CE9001_ALM_RSN"]
    assert [e["value"] for e in arr] == ["ABCD", 'E"F', ""]


@pytest.mark.parametrize("bad", ["NULL", "SOME_MACRO", "[3] = \"x\"", "0", "\"A\" MACRO"])
def test_non_string_element_stops_conversion_with_position(bad):
    text = f'const char *lpsztbl_CE9001_ALM_RSN[] = {{\n "A",\n {bad} ,\n "C" }};'
    with pytest.raises(ConvertError) as e:
        convert([], texts=[text])
    assert "lpsztbl_CE9001_ALM_RSN[1]" in str(e.value) and "줄 3" in str(e.value)


def test_empty_element_and_unclosed_array_fail():
    with pytest.raises(ConvertError, match="빈 원소"):
        parse_arrays('const char *lpsztbl_CE9001_ALM_RSN[] = { "A", , "B" };')
    with pytest.raises(ConvertError, match="닫히지 않"):
        parse_arrays('const char *lpsztbl_CE9001_ALM_RSN[] = { "A", "B"')


def test_4step_and_other_arrays_ignored_and_not_in_tables():
    text = (table_src("lpsztbl_CE9001_ALM_RSN", n=3) + table_src("lpsztbl_CE9001_ALM_RSN_4STEP", {1: ("DIFFERENT", None)}, n=3, ctype="const char* ")
            + 'const char *lpsztbl_CE9001_UID[] = { "u" };\n')
    res, _, _ = convert([], texts=[text])
    assert list(res["tables"]) == ["CE9001"] and res["tables"]["CE9001"][1] == "X1"
    assert res["ignored_arrays"] == ["lpsztbl_CE9001_ALM_RSN_4STEP"] and res["ignored_other_array_count"] == 1


def test_models_without_table_alias_and_unknown_alias_target():
    text = FAKE_ENUM + table_src("lpsztbl_CE9001_ALM_RSN", n=4)
    res, _, _ = convert([], aliases={"CE9002": "CE9001"}, texts=[text])
    assert res["models_without_table"] == ["CE9003"] and res["table_alias"] == {"CE9002": "CE9001"}
    with pytest.raises(ConvertError, match="--alias"):
        convert([], aliases={"CE9002": "NOPE"}, texts=[text])


def test_array_without_sys_id_and_d7_difference_warned():
    text = FAKE_ENUM + table_src("lpsztbl_CEZZZZ_ALM_RSN", n=2)
    res, warns, _ = convert([], texts=[text])
    assert any("sys_id 에 없는 모델: CEZZZZ" in w for w in warns) and any("D7 과 다름" in w for w in warns)
    res2, w2, _ = convert([], texts=[table_src("lpsztbl_CE7080_ALM_RSN", n=2)])           # enum 없음 → D7 사용
    assert res2["sys_id_source"] == "design_D7" and res2["sys_id"]["31"] == "CE7080" and "CE8310" in res2["models_without_table"]


def test_duplicate_array_across_inputs_fails():
    t = table_src("lpsztbl_CE9001_ALM_RSN", n=2)
    with pytest.raises(ConvertError, match="중복"):
        convert([], texts=[t, t])


def test_parse_sys_enum_implicit_values():
    assert parse_sys_enum("enum enum_sys_id { A_CE1 = 5, A_CE2, A_CE3 = 9, A_CE4 };") == {5: "CE1", 6: "CE2", 9: "CE3", 10: "CE4"}
    assert parse_sys_enum("int x;") is None


def test_main_writes_json_and_returns_error_code(tmp_path, capsys):
    src = tmp_path / "t.c"
    src.write_text(FAKE_ENUM + table_src("lpsztbl_CE9001_ALM_RSN", D6), encoding="utf-8")
    out = tmp_path / "ref" / "names.json"
    assert main([str(src), "--out", str(out), "--observed", "31:600"]) == 0
    assert json.load(open(out, encoding="utf-8"))["tables"]["CE9001"][233] == "HI-PWR"
    assert "HI-PWR" in capsys.readouterr().out
    bad = tmp_path / "bad.c"
    bad.write_text('const char *lpsztbl_CE9001_ALM_RSN[] = { NULL };', encoding="utf-8")
    assert main([str(bad), "--out", str(tmp_path / "x.json")]) == 2 and not (tmp_path / "x.json").exists()


def test_default_output_is_git_ignored_runs_dir():
    assert cv.DEFAULT_OUT.replace("\\", "/").endswith("validation/runs/ref/alarm_names.json")
