import sys
import os
sys.path.append(os.getcwd())
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
