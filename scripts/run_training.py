import os
import subprocess
import sys

base = "C:/Users/EliuaK_Csy/Plants-Disease-Detection"
os.chdir(base)

print("=" * 60)
print("EVA-02 Training Script")
print("=" * 60)

print("\nStarting training...")
print("Model: EVA-02 Large")
print("Epochs: 50")
print("Batch size: 16")
print("=" * 60)

cmd = [
    sys.executable,
    "main.py",
    "train",
    "--epochs", "50",
    "--batch-size", "16",
    "--no-prepare",
    "--no-wandb"
]

print(f"\nRunning: {' '.join(cmd)}")
print("=" * 60)

# Run training
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Write output to log file and print
log_file = os.path.join(base, "training.log")
with open(log_file, 'w') as f:
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        f.write(line)
        f.flush()

process.wait()
print("\n" + "=" * 60)
print(f"Training finished with exit code: {process.returncode}")
print(f"Log file: {log_file}")
print("=" * 60)