"""
시드 고정형 평가용 시나리오 생성기 (DB 불필요)

기존 generate_rca_history.py 의 문제점을 보완한 평가 전용 생성기:
  - 시드 고정으로 재현 가능 (같은 seed/설정 -> 같은 데이터, 같은 정답)
  - 장애가 '기간이 한정된 에피소드'(램프 -> 지속 -> 복구)이며 복구 시점까지 정답에 기록됨
  - 스텝 단위 정답(label): 0=정상, 1=열화 진행(ramp), 2=장애 지속(plateau)
  - 장애 강도가 다양함 (경미 ~ 심각) -> 자명한 임계치로는 잡히지 않는 사례 포함
  - 정상 데이터에도 현실적인 노이즈 포함: 산발적 에러 패킷, 트래픽 버스트, 광 흔들림, 결측 행
  - 유병률이 낮음 (기본 약 4~5%) -> 정밀도/오탐이 의미를 가짐

출력(디렉토리):
  traffic.csv, optical.csv   : DB 조회 결과와 동일한 스키마 (occur_date, ip_addr, cid, lid, ...)
  labels.csv                 : 스텝 단위 정답 (state, episode_id, scenario, nuisance)
  episodes.csv               : 에피소드 정답 (start / fail / end 시각, 시나리오, 강도)
  meta.json                  : 생성 설정 (seed 등)
"""
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

SCENARIOS = ("crc_error", "optical_degradation", "traffic_drop")
INTERVAL_MIN = 15
STEPS_PER_DAY = 24 * 60 // INTERVAL_MIN


@dataclass
class ScenarioConfig:
    seed: int = 7
    nodes: int = 4
    ports_per_node: int = 10
    days: int = 14
    start: str = "2026-01-05 00:00:00"
    ip_prefix: str = "10.20.0."

    # 에피소드 (포트별): 평균 mean_gap_days 마다 1건, 열화 3~8h -> 장애 지속 1~6h -> 복구
    mean_gap_days: float = 7.0
    ramp_steps: tuple = (12, 32)
    plateau_steps: tuple = (4, 24)
    warmup_days: int = 1                 # 초반 정상 구간 (첫 에피소드는 이후에 시작)
    scenario_weights: tuple = (0.4, 0.4, 0.2)

    # 정상 노이즈
    benign_error_prob: float = 0.02      # 정상에서도 산발적으로 발생하는 소량 에러
    burst_prob: float = 0.003            # 일시적 트래픽 버스트 (장애 아님)
    blip_prob: float = 0.003             # 일시적 광 파워 흔들림 (장애 아님)
    missing_prob: float = 0.005          # 결측 행

    # 장애 강도 범위
    crc_final_errors: tuple = (30, 5000)     # log-uniform
    optical_final_drop_db: tuple = (3.0, 25.0)
    traffic_drop_fraction: tuple = (0.5, 0.95)


def _port_profile(rng):
    tx_base = rng.uniform(-6.0, -1.0)
    return {
        "scale": rng.uniform(0.1, 2.0),
        "tx_pwr": tx_base,
        "rx_pwr": tx_base - rng.uniform(1.0, 4.0),
        "opt_noise": 0.1 * rng.uniform(0.8, 1.5),
        "phase": rng.uniform(-2, 2),
    }


def _traffic_baseline(rng, prof, hours):
    """24시간 주기 + 가우시안 노이즈 (기존 DataGenerator 와 동일한 분포)"""
    shifted = hours - prof["phase"]
    mult = (np.cos((shifted - 14) / 24.0 * 2 * np.pi) + 1.5) / 2.5
    mean = 100000 * prof["scale"] * mult
    std = 5000 * prof["scale"]
    tx = np.maximum(0, rng.normal(mean, std)).astype(np.int64)
    rx = np.maximum(0, rng.normal(mean, std)).astype(np.int64)
    return tx, rx


def _plan_episodes(rng, cfg, n_steps):
    """한 포트의 에피소드 목록 [(start, ramp, plateau, scenario, magnitude)] — 서로 겹치지 않음"""
    episodes = []
    t = cfg.warmup_days * STEPS_PER_DAY
    mean_gap = cfg.mean_gap_days * STEPS_PER_DAY
    while True:
        t += int(rng.exponential(mean_gap))
        ramp = int(rng.integers(cfg.ramp_steps[0], cfg.ramp_steps[1] + 1))
        plateau = int(rng.integers(cfg.plateau_steps[0], cfg.plateau_steps[1] + 1))
        if t + ramp + plateau >= n_steps:
            break
        scenario = str(rng.choice(SCENARIOS, p=np.array(cfg.scenario_weights) / sum(cfg.scenario_weights)))
        if scenario == "crc_error":
            lo, hi = np.log(cfg.crc_final_errors[0]), np.log(cfg.crc_final_errors[1])
            mag = float(np.exp(rng.uniform(lo, hi)))
        elif scenario == "optical_degradation":
            mag = float(rng.uniform(*cfg.optical_final_drop_db))
        else:
            mag = float(rng.uniform(*cfg.traffic_drop_fraction))
        episodes.append((t, ramp, plateau, scenario, mag))
        t += ramp + plateau + 12          # 복구 후 최소 3시간은 정상
    return episodes


def generate(cfg: ScenarioConfig):
    """Returns: dict(traffic, optical, labels, episodes) 의 DataFrame 들"""
    rng = np.random.default_rng(cfg.seed)
    n_steps = cfg.days * STEPS_PER_DAY
    times = pd.date_range(cfg.start, periods=n_steps, freq=f"{INTERVAL_MIN}min")
    hours = times.hour.values + times.minute.values / 60.0

    traffic_parts, optical_parts, label_parts, episode_rows = [], [], [], []
    ep_counter = 0

    for n in range(cfg.nodes):
        ip = f"{cfg.ip_prefix}{10 + n}"
        for p in range(cfg.ports_per_node):
            cid, lid = p, p + 1
            prof = _port_profile(rng)

            tx, rx = _traffic_baseline(rng, prof, hours)
            err = np.where(rng.random(n_steps) < cfg.benign_error_prob,
                           rng.integers(1, 6, n_steps), 0).astype(np.int64)
            otx = rng.normal(prof["tx_pwr"], prof["opt_noise"], n_steps)
            orx = rng.normal(prof["rx_pwr"], prof["opt_noise"], n_steps)

            state = np.zeros(n_steps, dtype=np.int8)
            ep_id = np.full(n_steps, -1, dtype=np.int64)
            scen = np.array([""] * n_steps, dtype=object)
            nuisance = np.zeros(n_steps, dtype=np.int8)

            # --- 노이즈(장애 아님): 트래픽 버스트 / 광 흔들림 ---
            for i in np.where(rng.random(n_steps) < cfg.burst_prob)[0]:
                k = int(rng.integers(1, 3))
                f = rng.uniform(1.8, 3.0)
                tx[i:i + k] = (tx[i:i + k] * f).astype(np.int64)
                rx[i:i + k] = (rx[i:i + k] * f).astype(np.int64)
                nuisance[i:i + k] = 1
            for i in np.where(rng.random(n_steps) < cfg.blip_prob)[0]:
                orx[i] += rng.choice([-1, 1]) * rng.uniform(0.8, 1.5)
                nuisance[i] = 1

            # --- 장애 에피소드 ---
            for (s, ramp, plateau, scenario, mag) in _plan_episodes(rng, cfg, n_steps):
                idx = np.arange(s, s + ramp + plateau)
                prog = np.minimum(1.0, (idx - s + 1) / ramp)        # 0->1 진행도 (지속 구간은 1.0)
                if scenario == "crc_error":
                    err[idx] = np.maximum(err[idx], (mag * prog ** 2).astype(np.int64))
                    tx[idx] = (tx[idx] * (1.0 - 0.2 * prog)).astype(np.int64)
                    rx[idx] = (rx[idx] * (1.0 - 0.2 * prog)).astype(np.int64)
                elif scenario == "optical_degradation":
                    orx[idx] -= mag * prog
                    otx[idx] -= 0.1 * mag * prog
                else:  # traffic_drop
                    keep = 1.0 - mag * prog
                    tx[idx] = (tx[idx] * keep).astype(np.int64)
                    rx[idx] = (rx[idx] * keep).astype(np.int64)
                state[s:s + ramp] = 1
                state[s + ramp:s + ramp + plateau] = 2
                ep_id[idx] = ep_counter
                scen[idx] = scenario
                nuisance[idx] = 0
                episode_rows.append({
                    "episode_id": ep_counter, "ip_addr": ip, "cid": cid, "lid": lid,
                    "scenario": scenario, "magnitude": round(mag, 3),
                    "t_start": times[s], "t_fail": times[s + ramp],
                    "t_end": times[s + ramp + plateau - 1],
                })
                ep_counter += 1

            # --- 결측 행 (트래픽/광 각각 독립) ---
            keep_t = rng.random(n_steps) >= cfg.missing_prob
            keep_o = rng.random(n_steps) >= cfg.missing_prob

            key = {"ip_addr": ip, "cid": cid, "lid": lid}
            traffic_parts.append(pd.DataFrame({"occur_date": times, **key, "tx_packet": tx, "rx_packet": rx,
                                               "error_packet": err})[keep_t])
            optical_parts.append(pd.DataFrame({"occur_date": times, **key, "tx_avg_power": np.round(otx, 2),
                                               "rx_avg_power": np.round(orx, 2)})[keep_o])
            label_parts.append(pd.DataFrame({"occur_date": times, **key, "state": state, "episode_id": ep_id,
                                             "scenario": scen, "nuisance": nuisance}))

    return {
        "traffic": pd.concat(traffic_parts, ignore_index=True),
        "optical": pd.concat(optical_parts, ignore_index=True),
        "labels": pd.concat(label_parts, ignore_index=True),
        "episodes": pd.DataFrame(episode_rows),
    }


def save(data: dict, cfg: ScenarioConfig, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for name, df in data.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg),
                   "ports": cfg.nodes * cfg.ports_per_node,
                   "episodes": int(len(data["episodes"])),
                   "fault_step_ratio": float((data["labels"]["state"] > 0).mean())}, f, indent=2)


def load(out_dir: str):
    parse = {"traffic": ["occur_date"], "optical": ["occur_date"], "labels": ["occur_date"],
             "episodes": ["t_start", "t_fail", "t_end"]}
    data = {n: pd.read_csv(os.path.join(out_dir, f"{n}.csv"), parse_dates=cols) for n, cols in parse.items()}
    data["labels"]["scenario"] = data["labels"]["scenario"].fillna("")
    return data
