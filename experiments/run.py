#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Python script to run two evaluation scripts sequentially.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_path):
    """
    Runs the specified Python script.
    
    Args:
        script_path (str): The full path to the script.
    
    Returns:
        bool: True if the script ran successfully, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Starting to run script: {script_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check if the script file exists
    if not os.path.exists(script_path):
        print(f"Error: Script file not found - {script_path}")
        return False
    
    try:
        # Run the script using the same Python interpreter
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,  # Display output directly in the console
            text=True,
            check=False, # Do not raise exception on non-zero exit code
            cwd=os.path.dirname(script_path)  # Set the working directory to the script's directory
        )
        
        if result.returncode == 0:
            print(f"\n✅ Script {os.path.basename(script_path)} ran successfully.")
            return True
        else:
            print(f"\n❌ Script {os.path.basename(script_path)} failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ An error occurred while running the script: {str(e)}")
        return False

def main():
    """Main function to orchestrate script execution."""
    print("🚀 Starting OpenVLA MFM evaluation script sequence")
    
    # Define the paths to the scripts to be run
    scripts = [
        "/root/autodl-tmp/openvla/MFM/experiments/eval_agentview.py",
        "/root/autodl-tmp/openvla/MFM/experiments/eval_agentview_dark.py"
    ]
    
    success_count = 0
    total_scripts = len(scripts)
    
    # Run each script in sequence
    for i, script_path in enumerate(scripts, 1):
        print(f"\n🔄 Progress: {i}/{total_scripts}")
        
        if run_script(script_path):
            success_count += 1
        else:
            # If a script fails, ask the user whether to continue
            print(f"\n⚠️ Script {os.path.basename(script_path)} failed. Continue to the next script?")
            user_input = input("Enter 'y' to continue, 'n' to stop: ").strip().lower()
            if user_input != 'y':
                print("User chose to stop execution.")
                break
    
    # Print the final summary
    print(f"\n{'='*60}")
    print("📊 Execution Summary")
    print(f"{'='*60}")
    print(f"Total scripts: {total_scripts}")
    print(f"Successfully run: {success_count}")
    print(f"Failed: {total_scripts - success_count}")
    print(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_scripts:
        print("🎉 All scripts ran successfully!")
        return 0
    else:
        print("⚠️ Some scripts failed to run.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ User interrupted the script execution.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred in the program: {str(e)}")
        sys.exit(1)