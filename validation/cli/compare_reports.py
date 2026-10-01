"""
평가 리포트 비교표 (report_<tag>.json 여러 개를 한 표로)

사용법:
  python validation/cli/compare_reports.py validation/runs/seed7_n6_d14 validation/runs/seed7_n6_d14_clean
  (각 폴더의 모든 report_*.json 을 읽어 AI 모델별 + 기준선(공통)을 한 표로 출력)
"""
import glob
import json
import os
import sys

COLS = [("early_detection_rate", "조기탐지", "pct"), ("episode_detection_rate", "탐지", "pct"),
        ("lead_time_median_min", "리드(분)", "num"), ("false_incidents_per_port_day", "오탐/포트일", "f3"),
        ("incident_precision", "이벤트P", "pct"), ("event_f1", "이벤트F1", "f3")]


def fmt(v, kind):
    if v is None:
        return "    -"
    return {"pct": f"{v * 100:5.1f}%", "num": f"{v:6.0f}", "f3": f"{v:6.3f}"}[kind]


def main(dirs):
    for d in dirs:
        reports = {os.path.basename(p)[len("report_"):-len(".json")]: json.load(open(p, encoding="utf-8"))
                   for p in sorted(glob.glob(os.path.join(d, "report_*.json")))}
        if not reports:
            print(f"[!] {d}: report_*.json 없음")
            continue
        meta = next(iter(reports.values()))["meta"]
        print(f"\n=== {d} | 포트 {meta['ports']} 에피소드 {meta['episodes']} 유병률 {meta['prevalence']:.2%} ===")
        print(f"{'방법':<28}" + "".join(f"{c[1]:>11}" for c in COLS) + f"{'AUPRC':>9}")
        for tag, rep in reports.items():
            r = rep["results"]["AI (최종 알람)"]
            print(f"{'AI [' + tag + ']':<28}" + "".join(f"{fmt(r['event'][c[0]], c[2]):>11}" for c in COLS)
                  + f"{fmt(r['step']['auprc'], 'f3'):>9}")
        # 기준선은 모델과 무관하게 동일하므로 첫 리포트에서 한 번만
        first = next(iter(reports.values()))["results"]
        for name, r in first.items():
            if name.startswith("[기준]"):
                print(f"{name:<28}" + "".join(f"{fmt(r['event'][c[0]], c[2]):>11}" for c in COLS)
                      + f"{fmt(r['step']['auprc'], 'f3'):>9}")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_dirs = [os.path.join(root_dir, "validation", "runs", d) for d in ("seed7_n6_d14", "seed7_n6_d14_clean")]
    main(sys.argv[1:] or default_dirs)
