# main.py
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from src.physics import calculate_baselines, run_worker_wrapper
from src.ml_pipeline import train_model, analyze_candidates
from src.visualize import plot_physics_manifold, plot_diagnostics

DATA_FILE = "data/thesis_dataset.csv"

def main():
    # 1. Setup Physics
    baselines = calculate_baselines()
    
    # 2. Data Generation
    if os.path.exists(DATA_FILE):
        print(f"\nLoading existing data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
    else:
        print("\n--- Step 1: Generating Training Data (Parallel) ---")
        BATCH = 50
        TOTAL_SAMPLES = 4000
        tasks = []
        for i in range(TOTAL_SAMPLES // BATCH):
            t_type = 'hadronic' if i % 2 == 0 else 'quark'
            tasks.append((t_type, BATCH, i, i))

        res = Parallel(n_jobs=-1)(delayed(run_worker_wrapper)(t, baselines) for t in tqdm(tasks))
        df = pd.DataFrame([item for sublist in res for item in sublist], 
                          columns=['Mass', 'Radius', 'Lambda', 'Label', 'Curve_ID'])
        df.to_csv(DATA_FILE, index=False)
        print(f"Generated {len(df)} samples.")

    # 3. Machine Learning
    model, X_test, y_test = train_model(df)
    
    # 4. Visualization
    plot_diagnostics(model, X_test, y_test)
    plot_physics_manifold(1000, baselines)
    
    # 5. Inference
    analyze_candidates(model)

if __name__ == "__main__":
    main()