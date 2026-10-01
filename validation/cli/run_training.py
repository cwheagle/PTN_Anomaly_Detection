import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
os.chdir(root_dir)  # data/, models/ 등 상대 경로 기준을 프로젝트 루트로 고정
from src.models.trainer import Trainer

def run_training():
    print("="*70)
    print("MODEL TRAINING PHASE")
    print("="*70)
    for ft in ['traffic', 'optical']:
        print(f"[*] Training {ft} model...")
        Trainer(ft).train()
    print("="*70)

if __name__ == "__main__":
    run_training()
