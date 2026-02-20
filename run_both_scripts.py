#!/usr/bin/env python3
"""
Wrapper script to run both analysis scripts and capture output.
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_name, args=None):
    """Run a Python script and capture output."""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Print stdout
        if result.stdout:
            print(result.stdout)
        
        # Print stderr if there are errors
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"ERROR: Script {script_name} timed out after 300 seconds")
        return 1
    except Exception as e:
        print(f"ERROR running {script_name}: {e}")
        return 1

if __name__ == "__main__":
    print("Starting dataset analysis...")
    
    # Run analyze_dataset.py
    rc1 = run_script("analyze_dataset.py")
    
    # Run validate_dataset.py with --no-tokenizer
    rc2 = run_script("validate_dataset.py", ["--no-tokenizer"])
    
    print(f"\n{'='*70}")
    print(f"Summary: analyze_dataset.py returned {rc1}, validate_dataset.py returned {rc2}")
    print(f"{'='*70}")
    
    sys.exit(max(rc1, rc2))
