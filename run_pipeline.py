# run_pipeline.py
import subprocess
import sys
import os

# List of scripts to run in order
pipeline_scripts = [
    "scripts/01_data_collection.py",
    "scripts/02_data_processing.py",
    "scripts/03_label_and_split.py",
    "scripts/04_train_model.py",
]

def run_script(script_path):
    """Executes a script and checks for errors."""
    print("="*60)
    print(f"▶️  RUNNING: {script_path}")
    print("="*60)
    try:
        # Use sys.executable to ensure the correct python environment is used
        result = subprocess.run(
            [sys.executable, script_path], 
            check=True, text=True, capture_output=True
        )
        print(result.stdout)
        print(f"✅ SUCCESS: {script_path} completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in {script_path}:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ ERROR: Script not found at {script_path}")
        return False


if __name__ == "__main__":
    for script in pipeline_scripts:
        if not run_script(script):
            print(f"\nPipeline stopped due to an error in {script}.")
            break
    else:
        print("\n🎉🎉🎉 All pipeline scripts executed successfully! 🎉🎉🎉")