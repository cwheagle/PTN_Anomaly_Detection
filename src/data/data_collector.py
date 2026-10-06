import os
import pandas as pd
from src.data import train_window
from src.data.db_connector import DBConnector

PORT_KEYS = train_window.PORT_KEYS

class DataCollector:
    """운영 환경에서의 배치 데이터 수집 엔진"""
    def __init__(self):
        self.db = DBConnector()

    @staticmethod
    def load_exclusions(path):
        """학습에서 제외할 구간 목록(CSV)을 로드. path가 없으면 None (제외 없음).

        CSV 컬럼: ip_addr, cid, lid, start_time, end_time  (구형 컬럼명 failure_time 도 end_time 으로 인정)
        운영 환경에는 정답지가 없으므로 기본은 제외 없음이며, 시뮬레이션·EMS 알람 이력·사람 피드백 등
        명시적으로 제외 구간을 알고 있을 때만 주입한다. (평가용 정답지를 학습에 암묵적으로 쓰지 않기 위함)
        """
        if not path:
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"Exclusion file not found: {path}")
        gt_df = pd.read_csv(path)
        if 'end_time' not in gt_df.columns and 'failure_time' in gt_df.columns:
            gt_df['end_time'] = gt_df['failure_time']
        gt_df['start_time'] = pd.to_datetime(gt_df['start_time'])
        gt_df['end_time'] = pd.to_datetime(gt_df['end_time'])
        if 'failure_time' in gt_df.columns:
            gt_df['failure_time'] = pd.to_datetime(gt_df['failure_time'])
        return gt_df

    @staticmethod
    def filter_excluded(df, exclusions):
        """exclusions의 (포트, start_time ~ end_time[=failure_time]) 구간(양끝 포함)에 해당하는 행을 제거"""
        if df is None or df.empty or exclusions is None or exclusions.empty:
            return df
        df = df.copy()
        df['occur_date'] = pd.to_datetime(df['occur_date'])
        drop_mask = train_window.exclusion_mask(df, train_window.merge_intervals(exclusions, touch=pd.Timedelta(0)))
        print(f"[*] DataCollector: Dropped {int(drop_mask.sum())} excluded records.")
        return df[~drop_mask].copy()

    def fetch_alarm_intervals(self, start, end, policy):
        """자기 알람 이력(anomaly_detection)에서 장애 의심 구간 산출 (DB 오류는 예외로 전파)"""
        rows = self.db.fetch_alarm_rows(start, end, policy.alarm_min_level)
        return train_window.intervals_from_alarms(
            rows, policy.suspect_pre_steps, policy.suspect_post_steps, policy.suspect_gap_steps)

    @staticmethod
    def _bound(value, end_of_day):
        """'YYYY-MM-DD' 는 하루의 시작/끝으로, datetime/전체 타임스탬프는 그대로"""
        if isinstance(value, str) and len(value) <= 10:
            return f"{value} {'23:59:59' if end_of_day else '00:00:00'}"
        return pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S')

    def collect_and_save(self, train_start, train_end, test_start=None, test_end=None, feature_type=None,
                         output_dir="data", stop_checker=None, exclude_path=None,
                         exclusions=None, val_port_fraction=None, split_salt="ptn", suspect_policy=None):
        """데이터를 수집하여 학습/테스트용 CSV로 저장 (Trainer 호출용)

        Args:
            train_start/end, test_start/end: 명시적 날짜 지정 방식 ('YYYY-MM-DD'; 포트 분할 모드에서는 datetime 도 가능)
            feature_type: 'traffic' 또는 'optical'. None이면 둘 다 수집.
            stop_checker: 중지 요청 여부를 확인할 콜백 함수
            exclude_path: 학습 제외 구간 CSV 경로. None이면 환경변수 TRAIN_EXCLUDE_CSV, 둘 다 없으면 제외 없음.
            exclusions: 추가로 제외할 구간 DataFrame (exclude_path 와 합집합)
            val_port_fraction: 지정하면 **포트 분할 모드** — train_start~train_end 한 기간을 포트 단위로
                학습/검증(`<ft>_train.csv`/`<ft>_test.csv`)으로 나눈다 (test_* 는 무시). None 이면 기존 날짜 분할.
            suspect_policy: RetrainPolicy. 지정하면 장애 의심 구간(자기 알람 이력 ∪ 규칙)을 자동 제외하고,
                의심 비율이 한도를 넘으면 해당 트랙 학습 데이터를 저장하지 않고 건너뜀(`skipped`).
        Returns:
            {ft: {'train': n, 'test': n, 'suspect_stats': {...}}} 또는 중단 시 {'skipped': 사유, 'suspect_stats': ...}
        """
        port_split = val_port_fraction is not None
        if port_split:
            fetch_start, fetch_end = self._bound(train_start, False), self._bound(train_end, True)
            t_start, t_end, v_start, v_end = fetch_start, fetch_end, fetch_start, fetch_end
        else:
            fetch_start = min(train_start, test_start)
            fetch_end = max(train_end, test_end)
            t_start, t_end = f"{train_start} 00:00:00", f"{train_end} 23:59:59"
            v_start, v_end = f"{test_start} 00:00:00", f"{test_end} 23:59:59"

        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # 학습 제외 구간: 명시 CSV(exclude_path/TRAIN_EXCLUDE_CSV) ∪ 주입된 exclusions (+ 자동 의심 구간)
        explicit = [self.load_exclusions(exclude_path or os.getenv("TRAIN_EXCLUDE_CSV")), exclusions]
        explicit = [e for e in explicit if e is not None and len(e)]
        explicit_iv = train_window.merge_intervals(*explicit, touch=pd.Timedelta(0)) if explicit else None
        if explicit_iv is not None:
            print(f"[*] DataCollector: {len(explicit_iv)} exclusion intervals will be removed from training data.")

        alarm_iv = None
        if suspect_policy is not None and suspect_policy.exclude_from_alarms:
            alarm_iv = self.fetch_alarm_intervals(fetch_start, fetch_end, suspect_policy)
            print(f"[*] DataCollector: {len(alarm_iv)} alarm-history suspect intervals.")

        def collect_track(ft, fetch):
            raw = fetch(fetch_start, fetch_end, stop_checker=stop_checker)
            if stop_checker and stop_checker():
                return False                                   # 중지 시 조기 리턴
            if raw is None or raw.empty:
                return True
            raw = raw.copy()
            raw['occur_date'] = pd.to_datetime(raw['occur_date'])

            stats = None
            if suspect_policy is not None:
                parts = [explicit_iv, alarm_iv]
                if suspect_policy.exclude_from_rules:
                    parts.append(train_window.intervals_from_rules(raw, suspect_policy))
                intervals = train_window.merge_intervals(*parts)
                stats = train_window.suspect_stats(raw, intervals, suspect_policy.port_drop_fraction)
                if train_window.exceeds_suspect_limit(stats, suspect_policy):
                    results[ft] = {'skipped': f"suspect_fraction={stats['fraction']:.3f}", 'suspect_stats': stats}
                    print(f"[!] {ft}: 의심 구간 비율 {stats['fraction']:.1%} > 한도 — 학습 데이터 저장 안 함")
                    return True
                raw, _ = train_window.apply_suspect_filter(raw, intervals, suspect_policy.port_drop_fraction, stats)
            elif explicit_iv is not None:
                raw = self.filter_excluded(raw, explicit_iv)

            raw = raw.sort_values(PORT_KEYS + ['occur_date'])
            if port_split:
                train, test = train_window.split_ports(raw, val_port_fraction, split_salt)
            else:
                train = raw[(raw['occur_date'] >= t_start) & (raw['occur_date'] <= t_end)]
                test = raw[(raw['occur_date'] >= v_start) & (raw['occur_date'] <= v_end)]

            train.to_csv(os.path.join(output_dir, f"{ft}_train.csv"), index=False)
            test.to_csv(os.path.join(output_dir, f"{ft}_test.csv"), index=False)
            results[ft] = {'train': len(train), 'test': len(test)}
            if stats is not None:
                results[ft]['suspect_stats'] = stats
            print(f"[*] {ft.capitalize()} collected: Train({len(train)}), Test({len(test)})")
            return True

        if feature_type is None or feature_type == 'traffic':
            if not collect_track('traffic', self.db.fetch_traffic):
                return results
        if feature_type is None or feature_type == 'optical':
            if not collect_track('optical', self.db.fetch_optical):
                return results
        return results
