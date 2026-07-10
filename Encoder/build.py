import subprocess
import sys

def build():
    cmd = [
        "C:\\msys64\\mingw64\\bin\\gcc.exe",
        "-O3",
         "-Wall",
        "encoder.c",
        "-o",
        "encoder.exe",
        "-lavcodec",
        "-lavformat",
        "-lavutil",
        "-lswscale",
        "-lx264"
    ]
    print(f"Running build command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"Exit code: {result.returncode}")
    sys.exit(result.returncode)

if __name__ == "__main__":
    build()
