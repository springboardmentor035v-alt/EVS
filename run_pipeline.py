# run_pipeline.py
import subprocess
import sys
import os

# List of scripts to run in order - CORRECTED FILE NAMES
pipeline_scripts = [
    "scripts/01_data_collection.py",
    "scripts/02_data_processing.py", 
    "scripts/03_label_and_split.py",
    "scripts/04_train_model.py",  # Fixed from 04_train_model.py
]

def run_script(script_path):
    """Executes a script and checks for errors."""
    print("="*60)
    print(f"▶️  RUNNING: {script_path}")
    print("="*60)
    try:
        # Check if script exists
        if not os.path.exists(script_path):
            print(f"❌ ERROR: Script not found at {script_path}")
            return False
            
        # Use sys.executable to ensure the correct python environment is used
        result = subprocess.run(
            [sys.executable, script_path], 
            check=True, 
            text=True, 
            capture_output=True,
            timeout=300  # 5 minute timeout per script
        )
        print(result.stdout)
        if result.stderr:
            print(f"⚠️  Warnings: {result.stderr}")
        print(f"✅ SUCCESS: {script_path} completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in {script_path}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ ERROR: Script not found at {script_path}")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR in {script_path}: {e}")
        return False

def check_requirements():
    """Check if required files exist before running pipeline"""
    print("🔍 Checking requirements...")
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("❌ 'data' directory not found")
        return False
        
    # Check if locations file exists
    if not os.path.exists("data/locations.csv"):
        print("❌ 'data/locations.csv' not found")
        return False
        
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ '.env' file not found")
        return False
        
    print("✅ All requirements found")
    return True

if __name__ == "__main__":
    print("🚀 Starting EnviroScan AI Pipeline")
    print("This will run the complete data processing and model training pipeline.")
    print()
    
    # Check requirements first
    if not check_requirements():
        print("\n❌ Please fix the above issues before running the pipeline.")
        sys.exit(1)
    
    # Run each script in sequence
    success_count = 0
    for script in pipeline_scripts:
        if run_script(script):
            success_count += 1
        else:
            print(f"\n💥 Pipeline stopped due to error in {script}")
            break
    else:
        # This runs only if loop completes without break
        print("\n" + "="*60)
        print(f"🎉 PIPELINE COMPLETE: {success_count}/{len(pipeline_scripts)} scripts successful")
        print("="*60)
        
        # Final check for model files
        print("\n🔍 Checking generated model files...")
        expected_files = [
            "outputs/pollution_source_model.joblib",
            "outputs/label_encoder.joblib", 
            "outputs/scaler.joblib"
        ]
        
        for file in expected_files:
            if os.path.exists(file):
                print(f"✅ {file}")
            else:
                print(f"❌ {file} - MISSING")