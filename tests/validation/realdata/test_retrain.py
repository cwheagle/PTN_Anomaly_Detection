"""개정 6 U6b: 판정/탐색 모델 재학습 실행기 (설계 9.7.5a R1~R6) — 작은 합성 스냅샷과 축소 에폭(테스트 전용 패치)으로 실제 Trainer 를 돌린다"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from tests.validation.realdata.test_replay_report import FakeDetector
from tests.validation.realdata.test_alarm_episodes import names as mk_names
from validation.realdata import commands as cmds
from validation.realdata import periods as pr
from validation.realdata import replay as rp
from validation.realdata import retrain as rt

D0 = pd.Timestamp("2026-07-10")
def _pick_ports():
    """홀드아웃 해시(salt ptn, 10%)에서 검증 포트가 2개 이상 되도록 14 + 2 포트를 고른다 (결정적)."""
    from src.data.train_window import is_val_port
    val = [c for c in range(1, 400) if is_val_port("1.1.1.1", c, 1, 0.10, "ptn")][:2]
    train = [c for c in range(1, 400) if not is_val_port("1.1.1.1", c, 1, 0.10, "ptn")][:14]
    return [(c, 1) for c in sorted(train + val)]


PORTS = _pick_ports()


@pytest.fixture(scope="module")
def small_cfg():
    """테스트 전용: MODEL_CONFIG 의 에폭/배치만 줄인다 (실제 실행에서는 MODEL_CONFIG 를 그대로 쓴다 — retrain.train_model 은 config_override 를 받지 않음)."""
    import src.models.trainer as tr
    orig = tr.MODEL_CONFIG
    tr.MODEL_CONFIG = {**orig, "epochs": 1, "batch_size": 512, "patience": 1}
    yield
    tr.MODEL_CONFIG = orig


@pytest.fixture
def snap(tmp_path):
    runs = tmp_path / "runs"
    rd = runs / "r1"
    (rd / "raw" / "eth").mkdir(parents=True)
    (rd / "raw" / "alarm").mkdir(parents=True)
    rng = np.random.RandomState(0)
    for i in range(50):
        day = D0 + pd.Timedelta(days=i)
        rows = []
        for c, l in PORTS:
            for k in range(1, 97):
                t = day + pd.Timedelta(minutes=15 * k)
                rows.append({"ip_addr": "1.1.1.1", "cid": c, "lid": l, "system_type": 31, "address": f"X-S.{c} -P{l}", "tx_packet": 1000 + rng.randint(0, 50),
                             "rx_packet": 900 + rng.randint(0, 50), "error_packet": 0, "occur_date": t, "bin_end": t, "src_table": "t", "id": k})
        pd.DataFrame(rows).to_csv(rd / "raw" / "eth" / f"{day:%Y-%m-%d}.csv.gz", index=False, compression="gzip")
    pd.DataFrame(columns=["id", "ip_addr", "cid", "lid", "system_type", "alarm_type", "severity_type", "occur_date", "release_date", "address", "src_table"]
                 ).to_csv(rd / "raw" / "alarm" / "all.csv.gz", index=False, compression="gzip")
    json.dump({"complete": True, "range_start": str(D0), "range_end": str(D0 + pd.Timedelta(days=50))}, open(rd / "manifest.json", "w"))
    nm = tmp_path / "names.json"
    n = mk_names()
    json.dump({"code_offset": 367, "sys_id": {str(k): v for k, v in n.sys_id.items()}, "tables": n.tables, "models_without_table": []}, open(nm, "w"))
    return cmds.Context(str(rd), str(runs), names_path=str(nm))


# ---- R1: 스냅샷 어댑터 ----------------------------------------------------------------------------------
def test_snapshot_adapter_enforces_half_open_window_and_empty_optical_and_alarms(snap):
    j0 = snap.periods["J0"]
    db = rt.SnapshotDB(snap.eth, snap.periods["S"], j0)
    wide = db.fetch_traffic(D0 - pd.Timedelta(days=30), j0 + pd.Timedelta(days=30))                  # 더 넓은 범위를 요청해도 [S, J0) 밖은 안 줌
    assert wide["occur_date"].min() > snap.periods["S"] and wide["occur_date"].max() < j0
    assert not (wide["occur_date"] >= j0).any() and (wide["occur_date"] == j0).sum() == 0              # J0 00:00 행은 판정 쪽
    assert db.fetch_optical("a", "b").empty and db.fetch_alarm_rows("a", "b").empty
    assert list(wide.columns) == ["occur_date", "ip_addr", "cid", "lid", "tx_packet", "rx_packet", "error_packet"]


def test_training_windows_are_fixed_by_data_days(snap):
    s, e, role = rt.training_window(snap, "judge")
    assert (s, e, role) == (snap.periods["S"], snap.periods["J0"], "judge")
    s2, x, role2 = rt.training_window(snap, "explore")
    assert role2 == "explore" and x == snap.data_days[20] + pd.Timedelta(days=1) < snap.periods["J0"]
    with pytest.raises(ValueError):
        rt.training_window(snap, "other")


# ---- 학습 (실제 Trainer, 축소 에폭) ---------------------------------------------------------------------------
def test_train_judge_model_records_everything_and_leaves_operational_models_untouched(snap, small_cfg):
    from src.config import PATHS
    before = dict(PATHS["traffic"])
    run = rt.train_model(snap, "judge")
    out = os.path.join(snap.runs_dir, "models", "judge_seed0")
    assert run["status"] == "completed" and run["train_seed"] == 0 and run["retries"] == 0 and run["role"] == "judge"
    assert run["operational_models_unchanged"] is True and PATHS["traffic"] == before                  # PATHS 복구, 운영 models/ 해시 불변
    assert os.path.exists(os.path.join(out, "traffic_registry.json")) and os.path.exists(os.path.join(out, "train_run.json"))
    tw = run["training_window"]
    assert tw["mode"] == "port_split" and tw["source"] == "snapshot" and tw["end_exclusive"] is True and pd.Timestamp(tw["end"]) == snap.periods["J0"]
    assert pr.check_training_window({"trained_at": run["trained_at"], "training_window": tw}, snap.periods["J0"])[0]       # 9.7.5 (b) 통과
    assert pd.Timestamp(run["train_csv"]["max_occur_date"]) < snap.periods["J0"] and run["train_csv"]["ports"] > 0 and run["val_csv"]["ports"] >= 1
    assert len(run["train_csv"]["sha256"]) == 64 and run["suspect_alarm_history"].startswith("empty") and run["alert_policy"] is None
    assert run["val_port_fraction"] == 0.10 and run["holdout_ports"] and run["torch"] and run["model_config_sha"]
    info = rt.model_info_from_dir(out)
    assert info["role"] == "judge" and info["alert_policy"] is None and info["train_csv_sha"] == run["train_csv"]["sha256"]
    assert pr.read_ledger(snap.runs_dir)[-1]["command"].startswith("train-model") and pd.Timestamp(pr.read_ledger(snap.runs_dir)[-1]["end"]) <= snap.periods["J0"]


def test_existing_model_folder_is_never_overwritten_or_retrained_with_another_seed(snap, small_cfg):
    rt.train_model(snap, "judge")
    with pytest.raises(rt.TrainFailure, match="덮어쓰지"):
        rt.train_model(snap, "judge")
    with pytest.raises(rt.TrainFailure, match="덮어쓰지"):
        rt.train_model(snap, "judge", seed=1, run_id="judge_seed0")


def test_technical_failure_can_rerun_same_seed_and_counts_retries(snap, small_cfg, monkeypatch):
    import src.models.trainer as tr
    real = tr.Trainer.train
    calls = {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real(self, *a, **k)

    monkeypatch.setattr(tr.Trainer, "train", flaky)
    with pytest.raises(OSError):
        rt.train_model(snap, "judge")
    out = os.path.join(snap.runs_dir, "models", "judge_seed0")
    assert json.load(open(os.path.join(out, "train_run.json")))["status"] == "technical_failure"
    run = rt.train_model(snap, "judge")                                                              # 같은 시드로 다시
    assert run["status"] == "completed" and run["retries"] == 1 and run["train_seed"] == 0


def test_rule_failure_means_no_judge_model_and_no_retry(snap, monkeypatch):
    from src.pipeline.retrain_policy import RetrainPolicy
    monkeypatch.setattr(rt, "collect", lambda ctx, s, e, d: ({"skipped": "suspect_fraction=0.342", "suspect_stats": {}}, RetrainPolicy()))
    with pytest.raises(rt.TrainFailure, match="AI 판정 불가"):
        rt.train_model(snap, "judge")
    out = os.path.join(snap.runs_dir, "models", "judge_seed0")
    run = json.load(open(os.path.join(out, "train_run.json")))
    assert run["status"] == "failed_rule" and "완화" in run["note"]
    with pytest.raises(rt.TrainFailure, match="덮어쓰지"):                                              # 기준 완화·재설정으로 다시 학습하지 않는다
        rt.train_model(snap, "judge")


def test_zero_holdout_or_empty_sequences_are_rule_failures():
    with pytest.raises(rt.TrainFailure, match="홀드아웃 포트가 0"):
        rt._check_collect({"train": 100, "test": 0})
    with pytest.raises(rt.TrainFailure):
        rt._check_collect({})
    rt._check_collect({"train": 10, "test": 3})
    assert not rt._finite(float("nan"), 1.0) and not rt._finite(None) and rt._finite(0.1, 2)


def test_train_model_has_no_config_override_parameter():
    import inspect
    assert set(inspect.signature(rt.train_model).parameters) == {"ctx", "kind", "run_id", "seed"}      # MODEL_CONFIG 그대로, 탐색 금지


# ---- 탐색 모델 + 재현 검사 ----------------------------------------------------------------------------------
def test_explore_model_window_and_reproducibility_check(snap, small_cfg):
    run = rt.train_model(snap, "explore")
    x = snap.data_days[20] + pd.Timedelta(days=1)
    assert pd.Timestamp(run["training_window"]["end"]) == x and run["role"] == "explore"
    out = os.path.join(snap.runs_dir, "models", "explore_seed0")
    assert rt.verify_reproducible(snap, out) == []                                                   # 같은 절차로 다시 만든 CSV 해시 일치
    csvp = os.path.join(out, "train_data", "traffic_train.csv")
    with open(csvp, "a", encoding="utf-8") as f:
        f.write("\n")
    # 파일을 건드려도 기록된 해시는 그대로이므로 재현(재생성) 검사는 통과 — 파일 변조는 폴더 해시(train_csv_sha 는 기록값, 파일 자체는 전후 대조)로 별도로 잡는다
    assert rt.verify_reproducible(snap, out) == []
    with open(os.path.join(snap.run_dir, "manifest.json"), "a") as f:
        f.write(" ")
    assert any("매니페스트" in p for p in rt.verify_reproducible(snap, out))


# ---- 잠금·판정·--models-dir (R6) --------------------------------------------------------------------------
def test_lock_with_models_dir_then_judge_and_dev_reports(snap, small_cfg):
    rt.train_model(snap, "judge")
    rt.train_model(snap, "explore")
    jdir = os.path.join(snap.runs_dir, "models", "judge_seed0")
    edir = os.path.join(snap.runs_dir, "models", "explore_seed0")
    cmds.cmd_join_check(snap, "dev")
    with pytest.raises(pr.PeriodError, match="판정 모델 폴더가 아닙니다"):
        cmds.cmd_lock(snap, models_dir=edir)                                                         # 탐색 모델은 잠글 수 없다
    # 개발 기간 보고: 판정 모델을 개발 구간에서 재생하면 in-sample 표시, 탐색 모델의 시간 밖 구간은 exploration 이지만 in-sample 아님
    x = snap.data_days[20] + pd.Timedelta(days=1)
    rep_in = cmds.cmd_report(snap, f"{(D0 + pd.Timedelta(days=21)):%Y-%m-%d}..{snap.periods['J0']:%Y-%m-%d}", "dev", models_dir=jdir)
    assert rep_in["header"]["in_sample"] is True and rep_in["header"]["exploration"] is True and rep_in["header"]["model_role"] == "judge"
    rep_out = cmds.cmd_report(snap, f"{x:%Y-%m-%d}..{snap.periods['J0']:%Y-%m-%d}", "dev", models_dir=edir)
    assert rep_out["header"]["in_sample"] is False and rep_out["header"]["exploration"] is True and rep_out["header"]["model_role"] == "explore"
    lock = cmds.cmd_lock(snap, models_dir=jdir)
    L = pr.load_lock(snap.runs_dir, snap.periods)["models"]["traffic"]
    assert L["models_dir"] == jdir and L["model_sha"] and L["scaler_sha"] and L["meta_sha"] and L["train_run_sha"] and L["train_csv_sha"] and L["snapshot_manifest_sha"]
    assert pr.load_lock(snap.runs_dir, snap.periods)["training_policy"]["alert_policy"] is None and "lock_path" in lock
    # 폴더 변조 → 표식 전 거부
    with pytest.raises(pr.PeriodError, match="--models-dir"):
        cmds.cmd_report(snap, "judge", "judge", models_dir=edir)
    with open(os.path.join(jdir, "traffic_scaler_v1.joblib"), "ab") as f:
        f.write(b"x")
    with pytest.raises(pr.PeriodError, match="scaler_sha"):
        cmds.cmd_report(snap, "judge", "judge")
    assert not os.path.exists(pr.run_marker_path(snap.runs_dir, snap.periods))


def test_judge_runs_with_retrained_model_after_hash_and_reproduction_checks(snap, small_cfg):
    rt.train_model(snap, "judge")
    jdir = os.path.join(snap.runs_dir, "models", "judge_seed0")
    cmds.cmd_join_check(snap, "dev")
    cmds.cmd_lock(snap, models_dir=jdir)
    rep = cmds.cmd_report(snap, "judge", "judge")                                                    # detector=None: 폴더 해시·학습 CSV 재현·스냅샷 해시 대조 후 실제 모델 로드
    h = rep["header"]
    assert h["purpose"] == "judge" and h["models_dir"] == jdir and h["model_role"] == "judge" and h["in_sample"] is False and h["exploration"] is False
    assert json.load(open(pr.run_marker_path(snap.runs_dir, snap.periods)))["phase"] == "completed"
    assert rep["header"]["slots"]["ai"]["playable"] > 0                                              # 실제 재학습 모델이 판정 구간을 재생함
