"""
P1-2 알람명 해석 (설계서 7.2~7.4)

- `AlarmNames`: system_type + alarm_type → 알람명. code = alarm_type − code_offset(367) 로 모델별 표를 조회한다.
  표 없는 모델·범위 밖·빈 이름·모르는 system_type 은 이름을 추측하지 않고 `name_status` 로 구분한다(7.3).
- `TargetMap`: 알람명 → 타겟 유형 매핑 CSV. 정확한 이름이 글롭보다 우선하고, 한 이름에 글롭이 둘 이상 걸리면 오류.
"""
import csv
import fnmatch
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_NAMES_PATH = os.path.join(ROOT, "validation", "runs", "ref", "alarm_names.json")
DEFAULT_TARGET_MAP = os.path.join(HERE, "alarm_target_map.csv")

UNCLASSIFIED = "미분류"
TARGET_TYPES = {"CRC증가", "트래픽단절", "광관찰", "제외", "해당없음"}
GRADES = {"1차", "보조", "사용안함", "관찰참고"}


class AlarmNameError(RuntimeError):
    pass


def load_override(path):
    """수동 매핑 CSV (열 model, alarm_type, alarm_name) → {(model, alarm_type): name}."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["model"].strip(), int(r["alarm_type"]))] = r["alarm_name"].strip()
    return out


class AlarmNames:
    def __init__(self, data, override=None, display_fmt="ALM-ID-{code}"):
        self.offset = int(data.get("code_offset", 367))
        self.sys_id = {int(k): v for k, v in data["sys_id"].items()}
        self.tables = data["tables"]
        self.alias = data.get("table_alias", {})
        self.override = override or {}
        self.display_fmt = display_fmt
        self.models_without_table = list(data.get("models_without_table", []))

    @classmethod
    def load(cls, path=DEFAULT_NAMES_PATH, override_csv=None, display_fmt="ALM-ID-{code}"):
        if not os.path.exists(path):
            raise AlarmNameError(f"알람 이름 JSON 이 없습니다: {path} — validation/cli/convert_alarm_tables.py 로 먼저 변환하세요")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, load_override(override_csv) if override_csv else None, display_fmt)

    def resolve(self, system_type, alarm_type):
        """반환 (alarm_name, name_status, model, code)."""
        try:
            st, at = int(system_type), int(alarm_type)
        except (TypeError, ValueError):
            return f"UNKNOWN-SYS-{system_type}-{alarm_type}", "unknown_model", None, None
        model = self.sys_id.get(st)
        code = at - self.offset
        if model is None:
            return f"UNKNOWN-SYS-{st}-{at}", "unknown_model", None, code
        if (model, at) in self.override:
            return self.override[(model, at)], "override", model, code
        table = self.tables.get(self.alias.get(model, model))
        if table is None:
            return f"NOTABLE-{model}-{at}", "no_table", model, code
        if not (0 <= code < len(table)):
            return self.display_fmt.format(code=code), "out_of_range", model, code
        name = table[code].strip()
        if name == "":
            return f"ALM-EMPTY-{code}", "empty", model, code
        return name, "ok", model, code

    def resolve_frame(self, df):
        """df(system_type, alarm_type) → model, code, alarm_name, name_status 열을 붙인 사본. (system_type, alarm_type) 조합별로 한 번만 조회."""
        out = df.copy()
        if out.empty:
            for c in ("model", "code", "alarm_name", "name_status"):
                out[c] = pd.Series(dtype="object")
            return out
        keys = out[["system_type", "alarm_type"]].drop_duplicates()
        res = {(a, b): self.resolve(a, b) for a, b in zip(keys["system_type"], keys["alarm_type"])}
        vals = [res[(a, b)] for a, b in zip(out["system_type"], out["alarm_type"])]
        out["alarm_name"] = [v[0] for v in vals]
        out["name_status"] = [v[1] for v in vals]
        out["model"] = [v[2] for v in vals]
        out["code"] = [v[3] for v in vals]
        return out

    def all_names(self):
        return {n for t in self.tables.values() for n in t if n}


class TargetMap:
    """알람명 → [(target_type, grade, source), ...]. 매핑에 없으면 [(미분류, "", "")]."""

    def __init__(self, rows):
        self.exact, self.globs = {}, {}
        for r in rows:
            if r["target_type"] not in TARGET_TYPES:
                raise AlarmNameError(f"알 수 없는 target_type {r['target_type']!r} (pattern {r['pattern']})")
            if r["grade"] not in GRADES:
                raise AlarmNameError(f"알 수 없는 grade {r['grade']!r} (pattern {r['pattern']})")
            entry = (r["target_type"], r["grade"], r["source"])
            target = self.globs if "*" in r["pattern"] else self.exact
            lst = target.setdefault(r["pattern"], [])
            if entry in lst:
                raise AlarmNameError(f"같은 행이 중복되었습니다: {r}")
            lst.append(entry)

    @classmethod
    def load(cls, path=DEFAULT_TARGET_MAP, known_names=None):
        with open(path, encoding="utf-8", newline="") as f:
            tm = cls(list(csv.DictReader(f)))
        if known_names is not None:
            tm.validate_names(known_names)
        return tm

    def validate_names(self, names):
        """정확한 이름이 없는 이름에 글롭이 둘 이상 걸리면 오류 (모호성은 데이터를 보기 전에 해소)."""
        for n in names:
            if n in self.exact:
                continue
            hits = [g for g in self.globs if fnmatch.fnmatchcase(n, g)]
            if len(hits) > 1:
                raise AlarmNameError(f"알람명 {n!r} 에 글롭이 둘 이상 걸립니다: {hits}")

    def classify(self, name):
        if name in self.exact:
            return list(self.exact[name])
        hits = [g for g in self.globs if fnmatch.fnmatchcase(name, g)]
        if len(hits) > 1:
            raise AlarmNameError(f"알람명 {name!r} 에 글롭이 둘 이상 걸립니다: {hits}")
        if hits:
            return list(self.globs[hits[0]])
        return [(UNCLASSIFIED, "", "")]

    def classify_frame(self, df, name_col="alarm_name", status_col="name_status"):
        """알람명이 'ok'(또는 override) 인 행만 분류, 나머지는 미분류. 반환: 입력 사본 + target_types(;로 연결)·grades 열."""
        out = df.copy()
        cache = {}

        def one(name, status):
            if status not in ("ok", "override"):
                return [(UNCLASSIFIED, "", "")]
            if name not in cache:
                cache[name] = self.classify(name)
            return cache[name]

        res = [one(n, s) for n, s in zip(out[name_col], out[status_col])]
        out["target_types"] = [";".join(t for t, _, _ in r) for r in res]
        out["grades"] = [";".join(g for _, g, _ in r) for r in res]
        return out
