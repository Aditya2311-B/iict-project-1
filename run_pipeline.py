import subprocess
import sys
import os

def run_script(script_name, args=[]):
    print("\n" + "="*60)
    print(f" RUNNING: {script_name}")
    print("="*60)
    try:
        # Run script using current Python interpreter
        result = subprocess.run([sys.executable, script_name] + args, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred while running {script_name}: {e}")
        return False

def main():
    print("============================================================")
    print("     SUMMER INTERNSHIP: FAKE NEWS DETECTION PIPELINE        ")
    print("============================================================")
    
    # Step 0: Download dataset if not present
    if not os.path.exists("train.csv"):
        print("Dataset 'train.csv' not found. Starting download...")
        if not run_script("download_dataset.py"):
            print("Failed to download dataset. Pipeline aborted.")
            sys.exit(1)
    else:
        print("Dataset 'train.csv' already exists. Skipping download.")
        
    # Step 1: Preprocessing
    if not run_script("week1_preprocessing.py"):
        print("Preprocessing failed. Pipeline aborted.")
        sys.exit(1)
        
    # Step 2: Feature Engineering & EDA
    if not run_script("week2_feature_engineering.py"):
        print("Feature engineering failed. Pipeline aborted.")
        sys.exit(1)
        
    # Step 3: Model Building
    if not run_script("week3_model_training.py"):
        print("Model training failed. Pipeline aborted.")
        sys.exit(1)
        
    # Step 4: Evaluation & Visualizations
    if not run_script("week4_evaluation.py"):
        print("Evaluation failed. Pipeline aborted.")
        sys.exit(1)
        
    print("\n" + "="*60)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("All models are trained, plots are generated, and metrics saved.")
    print("Opening the Interactive Fake News Prediction Tool...\n")
    
    # Start the interactive testing tool
    run_script("week4_evaluation.py", ["--interactive"])

if __name__ == "__main__":
    main()
