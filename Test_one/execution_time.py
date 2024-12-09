import time
import subprocess

# Define the script to benchmark
SCRIPT_NAME = "MxMultiGalois.py"  # Replace with the actual name of the script

def execute_script():
    """Run the script and measure its execution time."""
    start_time = time.time()
    subprocess.run(["python3", SCRIPT_NAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time

if __name__ == "__main__":
    execution_time = execute_script()
    print(f"Execution Time: {execution_time:.4f} seconds")
