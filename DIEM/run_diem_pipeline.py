import subprocess
import sys

# List the scripts you want to run
scripts = ["parse_gaze.py", "build_heatmaps.py", "extract_frames.py"]

for script in scripts:
    print(f"Starting {script}...")
    
    # sys.executable ensures it uses the same Python environment
    result = subprocess.run([sys.executable, script])
    
    # Check if the script executed successfully
    if result.returncode == 0:
        print(f"{script} finished successfully.\n")
    else:
        print(f"{script} failed with exit code {result.returncode}.\n")