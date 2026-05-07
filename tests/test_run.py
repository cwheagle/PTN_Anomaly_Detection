import sys
import os

# 현재 디렉토리를 경로에 추가하여 src 모듈을 찾을 수 있게 함
sys.path.append(os.getcwd())

from tests.test_train import run_training
from tests.test_inference import run_inference_test

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
