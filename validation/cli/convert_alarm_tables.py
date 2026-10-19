"""
EMS 알람 이름 표(C 소스) → JSON 변환기 (P1-2 U3, 설계서 7.1)

입력: 장비 소스의 `lpsztbl_<모델>_ALM_RSN[]` 배열이 든 C 텍스트 파일(1개 이상). 원문은 장비 소스 일부이므로
저장소에 커밋하지 않고, 결과 JSON 은 `validation/runs/ref/alarm_names.json`(git 제외)에 둔다.

- 배열 원소가 문자열 리터럴이 아니면(NULL·매크로·지정 초기화자) 변환 실패로 중단한다(조용히 건너뛰면 인덱스가 밀림).
- 원소 뒤 주석의 `= <값>`(enum 값)이 배열 위치와 같은지 검증한다. 불일치·중복 이름·빈 이름은 경고.
- `lpsztbl_<MODEL>_ALM_RSN` 정확 일치만 채택. `_4STEP` 등 그 밖의 `*_ALM_RSN*` 배열은 무시 목록.
- enum_sys_id 가 입력에 없으면 설계서 2.1 D7 의 목록을 쓴다(출력에 `sys_id_source` 로 구분).

사용법:
  python validation/cli/convert_alarm_tables.py <c_source>... [--out validation/runs/ref/alarm_names.json]
        [--alias CE7300S=CE7300] [--observed 31:600 12:390]
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

CODE_OFFSET = 367
DEFAULT_OUT = os.path.join(root_dir, "validation", "runs", "ref", "alarm_names.json")
EXPECTED_TABLE_LEN = 256

# 설계서 2.1 D7 (사용자 확인)
D7_SYS_ID = {7: "CE7200", 8: "CE7300", 9: "CE7400", 10: "CE7300S", 11: "CE740C", 12: "CE73K0", 13: "CE730M",
             14: "CE7510", 15: "CE7533", 16: "CE7511", 17: "CE7534", 18: "CE7515", 20: "CE7611", 21: "CE7610",
             22: "CE8300", 23: "CE8310", 24: "CE740E", 25: "CE5100", 26: "CE5200", 27: "CE5300", 28: "CE74C2",
             29: "CE5500", 30: "CE841P", 31: "CE7080", 32: "CE1830", 33: "CE7700Q", 34: "CE7700", 35: "CE7700Q2",
             42: "CE7700Q2V", 43: "CE7510Q", 48: "CE8600"}

_ARRAY_RE = re.compile(r"const\s+char\s*\*\s*(lpsztbl_\w+)\s*\[\s*\]\s*=\s*\{")
_TABLE_NAME_RE = re.compile(r"^lpsztbl_([A-Za-z0-9]+)_ALM_RSN$")
_ESC = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "0": "\0"}


class ConvertError(RuntimeError):
    pass


def _line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def scan_array(text, start, table):
    """`{` 바로 뒤(start)부터 짝이 되는 `}` 까지 원소를 읽는다. 반환: (원소 목록, 끝 위치).
    원소 = {"value": str, "comment": str, "line": int}. 문자열 리터럴이 아닌 토큰이 있으면 ConvertError."""
    elems, parts, bad, cur_line = [], [], [], None
    i, n = start, len(text)

    def finish(pos):
        nonlocal parts, bad, cur_line
        if bad:
            raise ConvertError(f"{table}[{len(elems)}] (줄 {cur_line}): 문자열 리터럴이 아닌 원소 {''.join(bad)!r} — 변환 중단")
        if parts:
            elems.append({"value": "".join(parts), "comment": "", "line": cur_line})
        parts, bad, cur_line = [], [], None

    last_end_line = None
    while i < n:
        c = text[i]
        if c == '"':
            j, buf = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    buf.append(_ESC.get(text[j + 1], text[j + 1]))
                    j += 2
                else:
                    buf.append(text[j])
                    j += 1
            if j >= n:
                raise ConvertError(f"{table}: 닫히지 않은 문자열 (줄 {_line_of(text, i)})")
            if cur_line is None:
                cur_line = _line_of(text, i)
            parts.append("".join(buf))
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            if elems and last_end_line == _line_of(text, i) and not elems[-1]["comment"]:
                elems[-1]["comment"] = text[i + 2:j].strip()
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise ConvertError(f"{table}: 닫히지 않은 주석 (줄 {_line_of(text, i)})")
            i = j + 2
        elif c == ",":
            if not parts and not bad:
                raise ConvertError(f"{table}[{len(elems)}] (줄 {_line_of(text, i)}): 빈 원소(쉼표만 있음) — 변환 중단")
            finish(i)
            last_end_line = _line_of(text, i)
            i += 1
        elif c == "}":
            finish(i)
            return elems, i
        elif c.isspace():
            i += 1
        else:
            if cur_line is None:
                cur_line = _line_of(text, i)
            bad.append(c)
            i += 1
    raise ConvertError(f"{table}: 배열이 닫히지 않았습니다")


def parse_arrays(text):
    """반환: {배열 전체 이름: [원소...]} (파일 내 순서 유지)."""
    out = {}
    for m in _ARRAY_RE.finditer(text):
        name = m.group(1)
        elems, _ = scan_array(text, m.end(), name)
        if name in out:
            raise ConvertError(f"배열 {name} 이 중복 정의되어 있습니다")
        out[name] = elems
    return out


_ENUM_VAL_RE = re.compile(r"=\s*(\d+)")
_ENUM_HEX_RE = re.compile(r"0[xX]([0-9a-fA-F]+)")


def check_table(model, elems):
    """enum 주석 검증. 반환 (경고 목록, 통계 dict)."""
    warns, mism, nocomment, hexbad = [], [], 0, []
    names = [e["value"].strip() for e in elems]
    for idx, e in enumerate(elems):
        cm = e["comment"]
        mv = _ENUM_VAL_RE.search(cm)
        if not mv:
            nocomment += 1
            continue
        if int(mv.group(1)) != idx:
            mism.append((idx, int(mv.group(1)), names[idx]))
        mh = _ENUM_HEX_RE.search(cm)
        if mh and int(mh.group(1), 16) != int(mv.group(1)):
            hexbad.append((idx, cm))
    if mism:
        warns.append(f"{model}: enum 값과 배열 위치 불일치 {len(mism)}건 (예: 위치 {mism[0][0]} ← enum {mism[0][1]} {mism[0][2]!r})")
    if hexbad:
        warns.append(f"{model}: 주석의 10진/16진 값 불일치 {len(hexbad)}건 (예: 위치 {hexbad[0][0]})")
    if nocomment:
        warns.append(f"{model}: enum 주석 없는 원소 {nocomment}개 (위치 검증 못 함)")
    empties = [i for i, nme in enumerate(names) if nme == ""]
    if empties:
        warns.append(f"{model}: 빈 이름 {len(empties)}개 (위치 {empties[:5]}{'…' if len(empties) > 5 else ''})")
    if len(elems) != EXPECTED_TABLE_LEN:
        warns.append(f"{model}: 원소 수 {len(elems)} (기대 {EXPECTED_TABLE_LEN})")
    seen, dups = {}, {}
    for i, nme in enumerate(names):
        if nme:
            seen.setdefault(nme, []).append(i)
    for nme, idxs in seen.items():
        if len(idxs) > 1:
            dups[nme] = idxs
    if dups:
        warns.append(f"{model}: 중복 이름 {len(dups)}종 (예: {next(iter(dups))!r} 위치 {next(iter(dups.values()))})")
    stats = {"entries": len(elems), "empty": len(empties), "enum_mismatch": len(mism), "no_comment": nocomment,
             "duplicate_names": dups}
    return warns, stats


_ENUM_SYS_RE = re.compile(r"enum\s+enum_sys_id\s*\{(.*?)\}|\{([^{}]*)\}\s*enum_sys_id", re.DOTALL)


def parse_sys_enum(text):
    """enum_sys_id → {값: 모델명}. 없으면 None. 값 생략은 직전 값 + 1. 모델명은 항목 이름 안의 CE... 토큰."""
    stripped = re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.DOTALL)
    m = _ENUM_SYS_RE.search(stripped)
    if not m:
        return None
    body = m.group(1) if m.group(1) is not None else m.group(2)
    out, val = {}, -1
    for item in [x.strip() for x in body.split(",") if x.strip()]:
        nm, _, expr = item.partition("=")
        if expr.strip():
            try:
                val = int(expr.strip(), 0)
            except ValueError:
                raise ConvertError(f"enum_sys_id: 해석할 수 없는 값 {item!r}")
        else:
            val += 1
        mm = re.search(r"(CE[0-9A-Za-z]+)", nm.strip().upper())
        if mm:
            out[val] = mm.group(1)
    return out


def convert(paths, aliases=None, texts=None):
    """변환 본체. 반환 (result dict, warnings list, table stats)."""
    aliases = aliases or {}
    texts = texts if texts is not None else [open(p, encoding="utf-8", errors="replace").read() for p in paths]
    shas = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    arrays, enum_found = {}, None
    for t in texts:
        for k, v in parse_arrays(t).items():
            if k in arrays:
                raise ConvertError(f"배열 {k} 이 여러 입력에 중복됩니다")
            arrays[k] = v
        e = parse_sys_enum(t)
        if e:
            enum_found = {**(enum_found or {}), **e}
    tables, ignored, warns, stats = {}, [], [], {}
    for name, elems in arrays.items():
        m = _TABLE_NAME_RE.match(name)
        if not m:
            ignored.append(name)
            continue
        model = m.group(1)
        w, st = check_table(model, elems)
        warns += w
        stats[model] = st
        tables[model] = [e["value"].strip() for e in elems]
    sys_id = enum_found if enum_found else dict(D7_SYS_ID)
    sys_source = "input_enum" if enum_found else "design_D7"
    if enum_found:
        diff = [f"{k}: 입력={enum_found.get(k)} 설계 D7={D7_SYS_ID.get(k)}"
                for k in sorted(set(enum_found) | set(D7_SYS_ID)) if enum_found.get(k) != D7_SYS_ID.get(k)]
        if diff:
            warns.append("enum_sys_id 가 설계서 D7 과 다름: " + "; ".join(diff))
    for a, target in aliases.items():
        if target not in tables:
            raise ConvertError(f"--alias {a}={target}: 대상 표 {target} 이 없습니다")
    models = set(sys_id.values())
    no_table = sorted(m for m in models if m not in tables and aliases.get(m) not in tables)
    without_enum = sorted(m for m in tables if m not in models)
    if without_enum:
        warns.append("배열은 있는데 sys_id 에 없는 모델: " + ", ".join(without_enum))
    result = {
        "source_sha256": shas, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "code_offset": CODE_OFFSET, "sys_id_source": sys_source,
        "sys_id": {str(k): v for k, v in sorted(sys_id.items())}, "tables": tables,
        "table_alias": dict(aliases), "ignored_arrays": [i for i in ignored if "ALM_RSN" in i],
        "ignored_other_array_count": len([i for i in ignored if "ALM_RSN" not in i]),
        "models_without_table": no_table, "warnings": warns,
        "table_stats": {m: {k: v for k, v in s.items() if k != "duplicate_names"} | {"duplicate_name_count": len(s["duplicate_names"])}
                        for m, s in stats.items()},
    }
    return result, warns, stats


def explain(result, observed):
    """관측된 `system_type:alarm_type` 을 code = alarm_type − 367 로 풀어 이름을 보여 준다 (오프셋 근거 확인용)."""
    lines = []
    for item in observed:
        st, at = (int(x) for x in item.split(":"))
        model = result["sys_id"].get(str(st))
        table = result["tables"].get(result["table_alias"].get(model, model)) if model else None
        code = at - result["code_offset"]
        nm = table[code] if table and 0 <= code < len(table) else None
        lines.append(f"  system_type {st} ({model}) alarm_type {at} → code {code} → {nm!r}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="C 소스의 알람 이름 표 → JSON")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--alias", action="append", default=[], metavar="MODEL=TARGET", help="예: CE7300S=CE7300")
    ap.add_argument("--observed", nargs="*", default=[], metavar="SYS:ALARM_TYPE")
    args = ap.parse_args(argv)
    aliases = dict(a.split("=", 1) for a in args.alias)
    try:
        result, warns, _ = convert(args.sources, aliases)
    except ConvertError as e:
        print(f"[변환 실패] {e}", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"표 {len(result['tables'])}개 → {args.out}")
    print(f"  sys_id 출처: {result['sys_id_source']}, 무시한 *_ALM_RSN* 배열: {result['ignored_arrays']}")
    print(f"  표 없는 모델: {result['models_without_table']}")
    for w in warns:
        print(f"[경고] {w}")
    if args.observed:
        print(explain(result, args.observed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
