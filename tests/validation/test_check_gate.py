"""validation/cli/check_gate.py 테스트 (P1-6 U2-G2: 공통 + G2, T-P6-U2a~h 의 G2 해당분)

시드 목록·금지 목록·고정 경로·판정 기준은 설계서 6장의 사전 고정값이며, 소형 데이터로 g2-train → g2 → g2-verify 가
끝까지 도는지, 1회 잠금·순서·출력 억제 가드가 학습·측정 전에 작동하는지 확인한다. 실제 근거 시드·사례는 열지 않는다.
"""
import argparse
import inspect
import json
import os

import pandas as pd
import pytest

from validation.cli import check_gate as cg
from validation.cli import check_promotion as cp

FORBIDDEN = (23, 31, 47, 53, 59, 61, 101, 102, 103, 105, 211, 900, 33, 34, 35)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """고정 경로(RUN_ROOT)를 임시 폴더로 돌린다 — 실제 validation/runs/p1_6 은 건드리지 않는다"""
    monkeypatch.setattr(cg, "RUN_ROOT", str(tmp_path / "p1_6"))
    return tmp_path / "p1_6"


def _train_args(train_seed, data_seeds="107,109,113", cases="N,K,W", active="models", nodes=2, days=6, epochs=1):
    return argparse.Namespace(train_seed=train_seed, data_seeds=data_seeds, cases=cases, active_models=active,
                              nodes=nodes, days=days, epochs=epochs)


def _write_dev(root):
    os.makedirs(root, exist_ok=True)
    (root / cg.G2_DEV_NAME).write_text("{}")


def _boom(*a, **k):
    raise AssertionError("거부되어야 할 요청이 학습·측정까지 갔음")


# ── 사전 고정 상수 ──
def test_constants_are_the_pre_registered_values():
    assert cg.P16_SEEDS == {"g3_dev": (7, 11), "g3_evidence": (79, 83, 89), "g2_data": (107, 109, 113)}
    assert set(cg.FORBIDDEN_SEEDS) == set(FORBIDDEN)
    assert cg.G2_CASES == ("N", "K", "W") and cg.G2_EPOCHS_W == 1
    assert cg.G2_MODES == {"A": "both_fail", "B": "lower_warn", "현행": "worse"}
    assert cg.G2_LOCK_NAME == "p1_6_g2_evidence.json" and cg.G2_DEV_NAME == "g2_dev.json"


# ── T-P6-U2b: 시드 가드 ──
@pytest.mark.parametrize("seed", FORBIDDEN)
def test_g2_train_rejects_forbidden_seeds_before_anything_is_created(root, monkeypatch, seed):
    monkeypatch.setattr(cg, "train_cases", _boom)
    for ts in (0, 1):
        with pytest.raises(SystemExit, match="사용 금지"):
            cg.cmd_g2_train(_train_args(ts, data_seeds=str(seed)))
    assert not root.exists()                                                  # 폴더·잠금 미생성


@pytest.mark.parametrize("seed", [7, 11, 79, 83, 89, 5, 108])               # G3 시드(교차 금지)·임의 시드
def test_g2_train_rejects_cross_use_and_unknown_seeds(root, monkeypatch, seed):
    monkeypatch.setattr(cg, "train_cases", _boom)
    with pytest.raises(SystemExit, match="허용 집합"):
        cg.cmd_g2_train(_train_args(0, data_seeds=f"107,{seed}"))
    assert not root.exists()


def test_check_seeds_rejects_g2_seeds_for_g3_sets_too():
    for s in cg.P16_SEEDS["g2_data"]:
        with pytest.raises(SystemExit, match="교차"):
            cg.check_seeds([s], cg.P16_SEEDS["g3_dev"] + cg.P16_SEEDS["g3_evidence"])
    for s in FORBIDDEN:
        with pytest.raises(SystemExit, match="사용 금지"):
            cg.check_seeds([s], cg.P16_SEEDS["g3_dev"])
    assert cg.check_seeds([11, 7], cg.P16_SEEDS["g3_dev"]) == [7, 11]


@pytest.mark.parametrize("data_seeds,cases", [("107,109", "N,K,W"), ("107,109,113", "N,K"), ("113", "W")])
def test_g2_evidence_cases_must_be_the_whole_set(root, monkeypatch, data_seeds, cases):
    monkeypatch.setattr(cg, "train_cases", _boom)
    _write_dev(root)
    with pytest.raises(SystemExit, match="전체만"):
        cg.cmd_g2_train(_train_args(1, data_seeds=data_seeds, cases=cases))
    assert not (root / "locks").exists()                                      # 거부는 잠금을 만들지 않음


def test_g2_train_rejects_unknown_case_and_train_seed(root, monkeypatch):
    monkeypatch.setattr(cg, "train_cases", _boom)
    with pytest.raises(SystemExit):
        cg.cmd_g2_train(_train_args(0, cases="N,X"))
    with pytest.raises(SystemExit):
        cg.cmd_g2_train(_train_args(2))


# ── T-P6-U2e: 순서 (g2_dev.json 없이는 난수 1 학습 거부) ──
def test_rand1_training_requires_dev_record_and_creates_no_lock_without_it(root, monkeypatch):
    monkeypatch.setattr(cg, "train_cases", _boom)
    with pytest.raises(SystemExit, match="개발 기록"):
        cg.cmd_g2_train(_train_args(1))
    assert not root.exists()


# ── T-P6-U2c: 잠금 ──
def test_rand1_lock_is_fixed_path_exclusive_before_training_and_survives_deleted_runs(root, monkeypatch):
    calls = []
    monkeypatch.setattr(cg, "train_cases", lambda *a, **k: calls.append(a))
    _write_dev(root)
    lock = root / "locks" / "p1_6_g2_evidence.json"
    assert cg.g2_lock_path() == str(lock)                                    # 고정 경로 — --record·모델 폴더와 무관(인자에 없음)
    cg.cmd_g2_train(_train_args(1))
    rec = json.loads(lock.read_text())
    assert len(calls) == 1 and rec["status"] == "trained" and rec["data_seeds"] == [107, 109, 113] and rec["cases"] == ["N", "K", "W"]
    os.makedirs(root / "g2" / "rand1" / "N_s107", exist_ok=True)
    import shutil
    shutil.rmtree(root / "g2")                                               # 실행 폴더를 지워도
    with pytest.raises(SystemExit, match="1회만"):                            # 같은 사례 집합 재학습은 거부
        cg.cmd_g2_train(_train_args(1))
    assert len(calls) == 1


def test_rand1_lock_is_created_before_training_so_a_failed_training_still_locks(root, monkeypatch):
    def fail(*a, **k):
        assert (root / "locks" / "p1_6_g2_evidence.json").exists()           # 학습 시작 전에 이미 잠금
        raise RuntimeError("학습 실패")
    monkeypatch.setattr(cg, "train_cases", fail)
    _write_dev(root)
    with pytest.raises(RuntimeError):
        cg.cmd_g2_train(_train_args(1))
    assert json.loads((root / "locks" / "p1_6_g2_evidence.json").read_text())["status"] == "started"
    with pytest.raises(SystemExit, match="1회만"):
        cg.cmd_g2_train(_train_args(1))
    verify = argparse.Namespace(active_models="models", no_g2c=True, record=None)
    with pytest.raises(SystemExit, match="trained"):                          # 학습 미완료 잠금은 읽기도 거부
        cg.cmd_g2_verify(verify)


def test_g2_verify_reads_evidence_once(root, monkeypatch):
    monkeypatch.setattr(cg, "g2_cases_table", lambda *a, **k: [])
    monkeypatch.setattr(cg, "judge_g2", lambda rows: {"N": {}, "pass": True})
    monkeypatch.setattr(cg, "_print_g2", lambda *a, **k: None)
    os.makedirs(root / "locks")
    lock = root / "locks" / cg.G2_LOCK_NAME
    lock.write_text(json.dumps({"status": "trained"}))
    rec = root / "copy.json"
    verify = argparse.Namespace(active_models="models", no_g2c=True, record=str(rec))
    cg.cmd_g2_verify(verify)
    assert json.loads(lock.read_text())["status"] == "verified" and rec.exists()
    with pytest.raises(SystemExit, match="1회만"):
        cg.cmd_g2_verify(argparse.Namespace(active_models="models", no_g2c=True, record=None))


def test_g2_verify_without_lock_is_rejected(root):
    with pytest.raises(SystemExit, match="잠금"):
        cg.cmd_g2_verify(argparse.Namespace(active_models="models", no_g2c=True, record=None))
    assert not root.exists()


# ── T-P6-U2d: --record 가드 ──
def test_g2_verify_record_guard_runs_before_any_state_change(root, monkeypatch):
    monkeypatch.setattr(cg, "g2_cases_table", _boom)
    os.makedirs(root / "locks")
    lock = root / "locks" / cg.G2_LOCK_NAME
    lock.write_text(json.dumps({"status": "trained"}))
    existing = root / "old.json"
    existing.write_text("{}")
    for record, msg in ((str(existing), "이미 있습니다"), (str(root / "nodir" / "x.json"), "폴더"), (str(lock), "이미 있습니다")):
        with pytest.raises(SystemExit, match=msg):
            cg.cmd_g2_verify(argparse.Namespace(active_models="models", no_g2c=True, record=record))
    assert json.loads(lock.read_text())["status"] == "trained"                # 거부는 잠금 상태를 바꾸지 않음
    assert not (root / "nodir").exists()


def test_g2_dev_record_is_not_overwritten(root, monkeypatch):
    monkeypatch.setattr(cg, "g2_cases_table", _boom)
    _write_dev(root)
    with pytest.raises(SystemExit, match="이미 있습니다"):
        cg.cmd_g2(argparse.Namespace(active_models="models", no_g2c=True))


def test_train_cases_refuses_existing_case_folders_without_overwriting(root, monkeypatch):
    monkeypatch.setattr(cp, "train_operational", _boom)
    os.makedirs(cg.case_dir(0, "K", 109))
    with pytest.raises(SystemExit, match="덮어쓰지"):
        cg.train_cases(0, (107, 109, 113), cg.G2_CASES, "models")


# ── T-P6-U2f: 난수 1 학습의 출력·로그에 값 없음 ──
def test_rand1_training_prints_and_logs_no_values(root, monkeypatch, capsys):
    def fake_train(out, **kw):
        print("[SAVE] Best model updated (Val Loss: 0.123456) threshold 0.987654")
        import sys
        print("val_loss 0.123456", file=sys.stderr)
        os.makedirs(out, exist_ok=True)
        return {}
    monkeypatch.setattr(cp, "train_operational", fake_train)
    _write_dev(root)
    cg.cmd_g2_train(_train_args(1))
    out = capsys.readouterr()
    assert "0.123456" not in out.out + out.err and "0.987654" not in out.out + out.err
    assert "학습 완료" in out.out
    logs = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(".log")]
    assert logs == []                                                         # 근거 사례는 로그 파일도 만들지 않음
    # 개발(난수 0)은 값을 화면에 내지 않고 로그 파일에만 남긴다
    cg.cmd_g2_train(_train_args(0))
    out = capsys.readouterr()
    assert "0.123456" not in out.out + out.err
    assert "0.123456" in (root / "g2" / "rand0" / "train.log").read_text()


# ── T-P6-U2g: 판정 함수 (설계서 6.4 기준) ──
def _row(kind, vr, b_status, causes=(), a_status="PASS"):
    return {"kind": kind, "track": "traffic", "val_ratio": vr, "B_status": b_status, "B_causes": list(causes), "A_status": a_status}


def _ok_rows():
    return ([_row("N", 1.0, "PASS")] * 3 + [_row("L", 0.01, "WARN", [])] * 2 +
            [_row("K", 0.2, "WARN")] + [_row("W", 4.0, "FAIL", ["val_loss"])])


def test_judge_g2_pass_case():
    j = cg.judge_g2(_ok_rows())
    assert j["N"]["ok"] and j["L"]["ok"] and j["K"]["ok"] is True and j["W"]["ok"] is True and j["pass"] is True


def test_judge_g2_n_fails_on_any_warn_or_fail():
    for status in ("WARN", "FAIL"):
        j = cg.judge_g2(_ok_rows() + [_row("N", 0.2, status)])
        assert j["N"]["ok"] is False and j["pass"] is False


def test_judge_g2_l_requires_every_case_to_be_flagged():
    j = cg.judge_g2(_ok_rows() + [_row("L", 0.5, "PASS")])
    assert j["L"]["ok"] is False and j["pass"] is False


def test_judge_g2_k_not_judgeable_when_no_leak_case_below_one_third():
    rows = [r for r in _ok_rows() if r["kind"] != "K"] + [_row("K", 0.34, "PASS"), _row("K", 1.0, "PASS")]
    j = cg.judge_g2(rows)
    assert j["K"]["ok"] is None and j["K"]["ratio_below_third"] == 0 and j["pass"] is True     # K 면책: 값을 낮춰 맞추지 않음
    # 경계: 정확히 1/3 은 '미만'이 아니다
    assert cg.judge_g2([r for r in _ok_rows() if r["kind"] != "K"] + [_row("K", 1 / 3, "PASS")])["K"]["ratio_below_third"] == 0
    # 비율 < 1/3 인데 WARN 이 아니면 K 위반(판정식은 결정적이라 단위 테스트용 입력)
    bad = cg.judge_g2([r for r in _ok_rows() if r["kind"] != "K"] + [_row("K", 0.1, "PASS")])
    assert bad["K"]["ok"] is False


def test_judge_g2_w_exemption_and_cause_separation():
    base = [r for r in _ok_rows() if r["kind"] != "W"]
    j = cg.judge_g2(base + [_row("W", 3.0, "PASS")])                           # 비율 ≤ 3(경계 포함) → 구성 실패, 판정 제외
    assert j["W"]["ok"] is None and j["W"]["excluded_not_worse"] == 1 and j["pass"] is True
    j = cg.judge_g2(base + [_row("W", 4.0, "FAIL", ["threshold"])])             # 비율 > 3 인데 FAIL 원인에 검증 손실이 없음
    assert j["W"]["ok"] is False and j["W"]["not_fail_or_no_val_cause"] == 1 and j["pass"] is False
    j = cg.judge_g2(base + [_row("W", 4.0, "FAIL", ["threshold", "val_loss"])])
    assert j["W"]["ok"] is True
    j = cg.judge_g2(base + [_row("W", 2.0, "FAIL", ["threshold"]), _row("W", 4.0, "FAIL", ["val_loss"])])
    assert j["W"]["threshold_only_fail"] == 1 and j["W"]["ok"] is True          # 임계치 항목만으로 FAIL 은 별도 기록
    j = cg.judge_g2(base + [_row("W", 4.0, "PASS")])
    assert j["W"]["ok"] is False


def test_g2_row_records_three_modes_with_cause_tags():
    act = {"threshold": 0.3, "final_val_loss": 3.01}
    low = {"threshold": 0.3, "final_val_loss": 0.0259}
    r = cg.g2_row("L", "traffic", act, low)
    assert (r["현행_status"], r["A_status"], r["B_status"]) == ("PASS", "FAIL", "WARN")
    assert r["A_causes"] == ["val_loss"] and r["B_causes"] == [] and r["val_ratio"] == pytest.approx(0.0259 / 3.01)
    worse = cg.g2_row("W", "traffic", {"threshold": 0.3, "final_val_loss": 0.02}, {"threshold": 1.0, "final_val_loss": 0.07})
    assert worse["B_status"] == "FAIL" and worse["B_causes"] == ["threshold", "val_loss"]


# ── T-P6-U2h: 추가 인자의 기본값 = 현행 ──
def test_new_arguments_default_to_current_behavior():
    assert inspect.signature(cp.train_operational).parameters["leak_val_into_train"].default is False
    assert inspect.signature(cp.v3_table).parameters["return_alarms"].default is False
    # 기존 인자의 기본값은 그대로
    p = inspect.signature(cp.train_operational).parameters
    assert (p["nodes"].default, p["days"].default, p["split"].default, p["exclusion"].default, p["val_fraction"].default) == \
           (4, 21, "port", "aub", None)


def test_v3_table_return_alarms_keeps_the_table_identical(tiny_env):
    t0 = cp.v3_table(str(tiny_env), str(tiny_env), seeds=[7], nodes=5, densities={"기준": 7.0})
    t1, alarms = cp.v3_table(str(tiny_env), str(tiny_env), seeds=[7], nodes=5, densities={"기준": 7.0}, return_alarms=True)
    pd.testing.assert_frame_equal(t0, t1)
    assert set(alarms) == {(7, "기준", ft, c[0]) for ft in cp.TRACKS for c in cp.V3_POOL}
    assert all({"alarm", "occur_date"} <= set(df.columns) for df in alarms.values())


# ── T-P6-U2a: 소형 데이터로 g2-train → g2 → g2-verify 끝까지 ──
@pytest.fixture(scope="module")
def e2e(tmp_path_factory, tiny_model_dir):
    root = tmp_path_factory.mktemp("p1_6")
    mp = pytest.MonkeyPatch()
    mp.setattr(cg, "RUN_ROOT", str(root / "p1_6"))
    try:
        active = str(tiny_model_dir)
        cg.cmd_g2_train(_train_args(0, active=active))
        cg.cmd_g2(argparse.Namespace(active_models=active, no_g2c=False))
        cg.cmd_g2_train(_train_args(1, active=active))
        record = str(root / "evidence_copy.json")
        cg.cmd_g2_verify(argparse.Namespace(active_models=active, no_g2c=True, record=record))
        yield root / "p1_6", record
    finally:
        mp.undo()


def test_g2_end_to_end_tables_and_files(e2e):
    root, record = e2e
    dev = json.load(open(root / "g2_dev.json"))
    assert dev["phase"] == "dev" and dev["train_seed"] == 0
    kinds = pd.Series([r["kind"] for r in dev["rows"]]).value_counts().to_dict()
    assert kinds == {"N": 12, "L": 6, "K": 6, "W": 6}                          # 트랙 2개 × (정상 순서쌍 6 · 레거시 3 · 누출 3 · 1에폭 3)
    assert all("g2c_mse_median_ratio" in r for r in dev["rows"])               # G2-C 는 기록만
    lock = json.load(open(root / "locks" / cg.G2_LOCK_NAME))
    assert lock["status"] == "verified" and "judgement" in lock
    ev = json.load(open(record))
    assert ev["phase"] == "evidence" and ev["train_seed"] == 1 and len(ev["rows"]) == len(dev["rows"])
    assert all("g2c_mse_median_ratio" not in r for r in ev["rows"])            # --no-g2c


def test_leak_case_trains_on_train_plus_holdout_and_other_cases_do_not(e2e):
    root, _ = e2e
    for ft in cp.TRACKS:
        k = root / "g2" / "rand0" / "K_s107" / "train_data"
        n = root / "g2" / "rand0" / "N_s107" / "train_data"
        tr, te = len(pd.read_csv(k / f"{ft}_train.csv")), len(pd.read_csv(k / f"{ft}_test.csv"))
        assert len(pd.read_csv(k / f"{ft}_train_leak.csv")) == tr + te
        assert not (n / f"{ft}_train_leak.csv").exists()
    w = json.load(open(root / "g2" / "rand0" / "W_s107" / "traffic_ae_v1.json"))
    assert w["config"]["epochs"] == 1


def test_rand1_run_leaves_no_log_and_a_second_run_is_refused(e2e):
    root, _ = e2e
    assert not any(f.endswith(".log") for d, _, fs in os.walk(root / "g2" / "rand1") for f in fs)
    with pytest.raises(SystemExit):
        cg.cmd_g2_train(_train_args(1))


# ══════════════════════════════════════════════
# U2-G3: g3 · g3-dev · g3-verify
# ══════════════════════════════════════════════
def _g3_args(**kw):
    base = dict(models=None, active_models="models", nodes=2, seeds="79,83,89", record=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _write_g3_dev(root, selected="G3-A"):
    os.makedirs(root, exist_ok=True)
    (root / cg.G3_DEV_NAME).write_text(json.dumps({"selected": selected}))


def test_g3_constants_are_the_pre_registered_values():
    assert cg.G3_LOCK_NAME == "p1_6_g3_evidence.json" and cg.G3_DEV_NAME == "g3_dev.json"
    assert set(cg.G3_MODELS) == {"s101", "s103"} and all(p.endswith(("promo_s101", "promo_s103")) for p in cg.G3_MODELS.values())
    assert cg.G3_ALTS == {"G3-A": "G3_A", "G3-C": "G3_C"}                      # G3-B 는 Q1 으로 제외


# ── T-P6-U2b: 시드 가드 (G3) ──
@pytest.mark.parametrize("seed", FORBIDDEN)
def test_g3_commands_reject_forbidden_seeds_before_measurement(root, monkeypatch, seed):
    monkeypatch.setattr(cg, "_g3_tables", _boom)
    _write_g3_dev(root)
    for fn, kw in ((cg.cmd_g3, {"seeds": str(seed)}), (cg.cmd_g3_verify, {"seeds": str(seed)})):
        with pytest.raises(SystemExit, match="사용 금지"):
            fn(_g3_args(**kw))
    assert not (root / "locks").exists()


@pytest.mark.parametrize("seeds", ["107", "7,79", "79,83,89", "13"])        # G2 시드·근거 혼입·임의 시드는 탐색(g3)에서 거부
def test_g3_explore_accepts_only_dev_seeds(root, monkeypatch, seeds):
    monkeypatch.setattr(cg, "_g3_tables", _boom)
    with pytest.raises(SystemExit, match="허용 집합"):
        cg.cmd_g3(_g3_args(seeds=seeds))


@pytest.mark.parametrize("seeds", ["79", "79,83", "83,89", "7,11", "107", "7,79,83,89"])
def test_g3_verify_requires_the_whole_evidence_set(root, monkeypatch, seeds):
    monkeypatch.setattr(cg, "_g3_tables", _boom)
    _write_g3_dev(root)
    with pytest.raises(SystemExit):
        cg.cmd_g3_verify(_g3_args(seeds=seeds))
    assert not (root / "locks").exists()                                     # 거부는 잠금도 폴더도 남기지 않음


# ── T-P6-U2e: 순서 ──
def test_g3_verify_requires_dev_record_with_selected_alternative(root, monkeypatch):
    monkeypatch.setattr(cg, "_g3_tables", _boom)
    with pytest.raises(SystemExit, match="개발 기록"):
        cg.cmd_g3_verify(_g3_args())
    for sel in (None, "G3-B", "x"):
        _write_g3_dev(root, sel)
        with pytest.raises(SystemExit, match="선택된 대안이 없"):
            cg.cmd_g3_verify(_g3_args())
    assert not (root / "locks").exists()


def test_g3_dev_record_is_not_overwritten(root, monkeypatch):
    monkeypatch.setattr(cg, "_g3_tables", _boom)
    _write_g3_dev(root)
    with pytest.raises(SystemExit, match="이미 있습니다"):
        cg.cmd_g3_dev(_g3_args())


# ── T-P6-U2d: 기록 가드 ──
def test_g3_verify_record_guard_runs_before_lock(root, monkeypatch):
    monkeypatch.setattr(cg, "_g3_tables", _boom)
    _write_g3_dev(root)
    old = root / "old.json"
    old.write_text("{}")
    lock_path = root / "locks" / cg.G3_LOCK_NAME
    for record, msg in ((str(old), "이미 있습니다"), (str(root / "nodir" / "x.json"), "폴더"), (str(lock_path), "폴더|잠금 파일")):
        with pytest.raises(SystemExit, match=msg):
            cg.cmd_g3_verify(_g3_args(record=record))
    assert not (root / "locks").exists() and not (root / "nodir").exists()


# ── T-P6-U2c: 잠금 ──
def test_g3_verify_lock_is_fixed_path_exclusive_before_measurement_and_one_shot(root, monkeypatch, tmp_path):
    seen = []

    def fake_tables(models, active, seeds, nodes):
        seen.append(((root / "locks" / cg.G3_LOCK_NAME).exists(), sorted(seeds)))      # 측정 시작 전에 잠금이 이미 있음
        return {m: _mk_table() for m in models}
    monkeypatch.setattr(cg, "_g3_tables", fake_tables)
    _write_g3_dev(root, "G3-A")
    rec1 = tmp_path / "r1.json"
    cg.cmd_g3_verify(_g3_args(record=str(rec1)))
    assert seen == [(True, [79, 83, 89])]
    lock = json.loads((root / "locks" / cg.G3_LOCK_NAME).read_text())
    assert lock["status"] == "verified" and lock["seeds"] == [79, 83, 89] and lock["selected"] == "G3-A" and "verdict" in lock
    assert cg.g3_lock_path() == str(root / "locks" / cg.G3_LOCK_NAME)
    for kw in ({}, {"record": str(tmp_path / "other_name.json")}, {"models": f"x={tmp_path}"}):   # 기록 이름·모델 폴더를 바꿔도 우회 불가
        with pytest.raises(SystemExit, match="1회만"):
            cg.cmd_g3_verify(_g3_args(**kw))
    assert len(seen) == 1 and rec1.exists()
    assert not (tmp_path / "other_name.json").exists()


def test_g3_verify_failed_measurement_still_leaves_the_lock(root, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("측정 실패")
    monkeypatch.setattr(cg, "_g3_tables", fail)
    _write_g3_dev(root)
    with pytest.raises(RuntimeError):
        cg.cmd_g3_verify(_g3_args())
    assert json.loads((root / "locks" / cg.G3_LOCK_NAME).read_text())["status"] == "started"
    with pytest.raises(SystemExit, match="1회만"):
        cg.cmd_g3_verify(_g3_args())


# ── T-P6-U2g: G3 판정 함수 ──
A_, B_ = cp.V3_ACTIVES[0][0], cp.V3_ACTIVES[1][0]


def _t(rows):
    return pd.DataFrame([{"seed": 79, "density": "d", "track": "traffic", "candidate": "c", "truth_fp": 0.0, "total": 0.0,
                          "active_total": 0.0, "delta": 0.0, "new_rate": 0.0, "n_new": 0, "caught_new_eps": 0,
                          "label": l, "active": a, "G3_A": x, "G3_C": y, "G3_C4": z} for l, a, x, y, z in rows])


def _mk_table():
    return _t([("정상", A_, "PASS", "PASS", "PASS"), ("정상", B_, "PASS", "PASS", "WARN"),
               ("과다", A_, "WARN", "WARN", "WARN"), ("과다", B_, "FAIL", "FAIL", "WARN")])


def test_judge_alt_rules_and_boundaries():
    j = cg.judge_alt(_mk_table(), "G3_A")
    assert (j["rule1"], j["rule2"], j["rule3"], j["ok"]) == (True, True, True, True)
    c4 = cg.judge_alt(_mk_table(), "G3_C4")                                   # 현행: 과다(b) 가 WARN → 규칙 2 위반
    assert (c4["rule2_excess_b_not_fail"], c4["rule2"], c4["ok"]) == (1, False, False)
    bad1 = _t([("정상", B_, "FAIL", "PASS", "PASS"), ("과다", B_, "FAIL", "FAIL", "FAIL"), ("과다", A_, "WARN", "WARN", "WARN")])
    assert cg.judge_alt(bad1, "G3_A")["rule1_normal_fail"] == 1 and cg.judge_alt(bad1, "G3_A")["ok"] is False
    assert cg.judge_alt(bad1, "G3_C")["ok"] is True


def test_judge_alt_rule3_is_a_safety_check_and_same_for_both_alternatives():
    """규칙 3: 과다 후보의 (a) 판정은 절대 상한 WARN 경로가 결정하므로 대안 간 같다 — PASS 가 나오면 두 대안 모두 위반"""
    t = _t([("정상", A_, "PASS", "PASS", "PASS"), ("과다", A_, "PASS", "PASS", "PASS"), ("과다", B_, "FAIL", "FAIL", "FAIL")])
    for col in ("G3_A", "G3_C"):
        j = cg.judge_alt(t, col)
        assert j["rule3_excess_a_pass"] == 1 and j["rule3"] is False and j["ok"] is False


def test_judge_alt_undetermined_without_excess_labels():
    j = cg.judge_alt(_t([("정상", A_, "PASS", "PASS", "PASS"), ("정상", B_, "PASS", "PASS", "PASS")]), "G3_A")
    assert j["rule2"] is None and j["rule3"] is None and j["ok"] is None


def test_judge_g3_models_selection_rule_requires_both_models_and_prefers_g3a():
    ok, bad = _mk_table(), _t([("정상", B_, "FAIL", "PASS", "PASS"), ("과다", B_, "FAIL", "FAIL", "WARN"), ("과다", A_, "WARN", "WARN", "WARN")])
    assert cg.judge_g3_models({"s101": ok, "s103": ok})["selected"] == "G3-A"
    j = cg.judge_g3_models({"s101": ok, "s103": bad})                           # s103 에서 G3-A 가 정상 후보를 FAIL → G3-A 탈락, G3-C 통과
    assert j["passed"] == {"G3-A": False, "G3-C": True} and j["selected"] == "G3-C"
    worse = _t([("정상", B_, "FAIL", "FAIL", "PASS"), ("과다", B_, "FAIL", "FAIL", "WARN"), ("과다", A_, "WARN", "WARN", "WARN")])
    j = cg.judge_g3_models({"s101": ok, "s103": worse})                         # 둘 다 미충족 → 선택 없음(① FAIL), 값 재선택·조합 없음
    assert j["selected"] is None and j["passed"] == {"G3-A": False, "G3-C": False}
    assert cg.judge_g3_models({"s101": ok})["reference_c4"]["s101"]["ok"] is False


def _s(total, pd_=180):
    return {"ports": pd_ / 3, "port_days": pd_, "incidents_per_port_day": total}


@pytest.mark.parametrize("cand,act,expected", [
    (0.07, 0.05, "PASS"),      # Δ = 0.02 (부동소수 0.020000000000000004) → 초과 아님, 절대 상한 0.05 이내 → PASS 는 아님: 0.07 > 0.05 → WARN
    (0.07, 0.04, "FAIL"),
    (0.04, 0.02, "PASS"),
    (0.0625, 0.0625, "WARN"),  # Δ 0, 절대 상한 0.05 초과 → WARN
])
def test_alt_status_delta_keeps_absolute_warn_and_boundary(cand, act, expected):
    if (cand, act) == (0.07, 0.05):
        expected = "WARN"
    assert cg.alt_status("delta", _s(cand), _s(act), None) == expected


@pytest.mark.parametrize("rate,total,expected", [(0.02, 0.01, "PASS"), (0.02 + 1e-6, 0.01, "FAIL"), (0.0, 0.06, "WARN"), (0.0, 0.05, "PASS")])
def test_alt_status_new_alarms(rate, total, expected):
    assert cg.alt_status("new_alarms", _s(total), _s(0.0), rate) == expected


def test_alt_status_guard_skip_and_unknown_rule():
    assert cg.alt_status("delta", _s(0.9, 149), _s(0.0, 149), None) == "SKIP"          # 최소 포트·일 가드(150)
    assert cg.alt_status("delta", _s(0.9, 150), _s(0.0, 150), None) == "FAIL"
    assert cg.alt_status("delta", None, _s(0.0), None) == "SKIP"
    assert cg.alt_status("delta", _s(0.9), None, None) == "WARN"                      # 활성 없음: 상대 판정 안 함, 절대 상한 WARN
    with pytest.raises(ValueError):
        cg.alt_status("ratio", _s(0.1), _s(0.1), None)


# ── T-P6-U2a: 소형 데이터로 g3 → g3-dev → g3-verify, 라벨은 P1-3 V3 정의 그대로 ──
@pytest.fixture(scope="module")
def g3_e2e(tmp_path_factory, tiny_model_dir):
    root = tmp_path_factory.mktemp("p1_6_g3")
    mp = pytest.MonkeyPatch()
    mp.setattr(cg, "RUN_ROOT", str(root / "p1_6"))
    try:
        tiny = str(tiny_model_dir)
        models = f"s101={tiny},s103={tiny}"
        explore = root / "explore.json"
        cg.cmd_g3(_g3_args(models=models, active_models=tiny, seeds="7", record=str(explore), nodes=5))
        cg.cmd_g3_dev(_g3_args(models=models, active_models=tiny, nodes=5))
        yield root / "p1_6", explore, tiny, models
    finally:
        mp.undo()


def test_g3_end_to_end_table_columns_and_labels_equal_p1_3(g3_e2e):
    root, explore, tiny, _ = g3_e2e
    t = cg.g3_table(tiny, tiny, [7], nodes=5, densities={"기준": 7.0})
    base = cp.v3_table(tiny, tiny, [7], nodes=5, densities={"기준": 7.0})
    pd.testing.assert_series_equal(t["label"], base["label"])                  # 라벨 = P1-3 V3 정의(v3_table 그대로)
    pd.testing.assert_series_equal(t["G3_C4"], base["G3"], check_names=False)  # C4' 열 = src check_g3
    assert {"G3_A", "G3_C", "delta", "new_rate", "n_new", "caught_new_eps"} <= set(t.columns) and len(t) == 20
    assert set(t.G3_A) <= {"PASS", "WARN", "FAIL", "SKIP"} and (t.n_new >= 0).all() and (t.caught_new_eps >= 0).all()
    assert json.load(open(explore))["phase"] == "explore" and os.path.exists(str(explore) + ".csv")


def test_g3_dev_end_to_end_writes_record_with_both_models(g3_e2e):
    root, _, _, _ = g3_e2e
    rec = json.load(open(root / "g3_dev.json"))
    assert rec["phase"] == "dev" and rec["seeds"] == [7, 11] and rec["models"] == ["s101", "s103"]
    assert set(rec["judgement"]["per_alt"]) == {"G3-A", "G3-C"} and rec["selected"] in (None, "G3-A", "G3-C")
    assert set(rec["judgement"]["per_alt"]["G3-A"]) == {"s101", "s103"}


def test_g3_verify_end_to_end_runs_once_on_evidence_seeds_with_tiny_models(g3_e2e, tmp_path):
    root, _, tiny, models = g3_e2e
    (root / cg.G3_DEV_NAME).unlink()
    _write_g3_dev(root, "G3-A")                                                 # 소형 모델의 개발 결과와 무관하게 순서 가드를 통과시킨다
    rec = tmp_path / "ev.json"
    cg.cmd_g3_verify(_g3_args(models=models, active_models=tiny, record=str(rec), nodes=2))
    lock = json.load(open(root / "locks" / cg.G3_LOCK_NAME))
    assert lock["status"] == "verified" and lock["seeds"] == [79, 83, 89]
    ev = json.load(open(rec))
    assert ev["phase"] == "evidence" and ev["verdict"]["selected"] == "G3-A"
    pd.read_csv(str(rec) + ".csv")
    with pytest.raises(SystemExit, match="1회만"):
        cg.cmd_g3_verify(_g3_args(models=models, active_models=tiny, nodes=2))
