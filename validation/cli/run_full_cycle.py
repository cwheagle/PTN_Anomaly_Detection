import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
os.chdir(root_dir)  # data/, models/ 등 상대 경로 기준을 프로젝트 루트로 고정

from run_training import run_training
from run_inference_check import run_inference_test

def run_full_cycle():
    """학습부터 추론 검증까지 전체 사이클 통합 실행"""
    print("\n" + "#"*70)
    print("FULL PIPELINE INTEGRATION TEST (TRAIN + INFERENCE)")
    print("#"*70)

    # 1. 모델 학습
    print("[*] Starting training phase...")
    run_training()

    # 2. 추론 및 검증
    run_inference_test()

    print("\n" + "#"*70)
    print("FULL PIPELINE TEST COMPLETE")
    print("#"*70)

if __name__ == "__main__":
    run_full_cycle()
