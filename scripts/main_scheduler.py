import os
import sys

# 프로젝트 루트 디렉토리를 path에 추가 (scripts 폴더에서 실행 시 대응)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.scheduler import PTNAnomalyScheduler

def main():
    print("="*50)
    print("   PTN Anomaly Detection System (v1.0)   ")
    print("="*50)
    print("Initializing Scheduler...")
    
    try:
        scheduler = PTNAnomalyScheduler()
        # 스케줄러 시작 (즉시 첫 실행 후 설정된 INTERVAL_MINUTES 주기마다 반복)
        scheduler.start()
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
