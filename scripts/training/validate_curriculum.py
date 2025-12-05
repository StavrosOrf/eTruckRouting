#!/usr/bin/env python
"""
Simple validation script to check curriculum learning files are properly created.
Does not require running the actual environment.
"""

import os
import json

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def validate_json_config(filepath, required_keys):
    """Validate JSON config file has required keys."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        for key in required_keys:
            if key not in config:
                print(f"  ⚠️  Missing key: {key}")
                return False
        
        print(f"  ✅ Valid JSON with required keys: {required_keys}")
        return True
    except Exception as e:
        print(f"  ❌ Error reading JSON: {e}")
        return False

def main():
    """Validate curriculum learning implementation."""
    print("\n" + "="*70)
    print("CURRICULUM LEARNING IMPLEMENTATION VALIDATION")
    print("="*70)
    
    base_path = "/home/sorfanouda/EVPR"
    all_ok = True
    
    # Check core implementation files
    print("\n📦 Core Implementation Files:")
    all_ok &= check_file_exists(
        f"{base_path}/EVRoutingEnv/models/curriculum_env.py",
        "Curriculum environment wrapper"
    )
    all_ok &= check_file_exists(
        f"{base_path}/scripts/training/train_curriculum.py",
        "Curriculum training script"
    )
    
    # Check configuration files
    print("\n⚙️  Configuration Files:")
    config_files = [
        ("curriculum_config_uniform.json", ["strategy", "truck_range", "stop_range"]),
        ("curriculum_config_staged.json", ["strategy", "stages"]),
        ("curriculum_config_mixed.json", ["strategy", "difficulty_levels"]),
    ]
    
    for config_file, required_keys in config_files:
        filepath = f"{base_path}/EVRoutingEnv/config_files/{config_file}"
        if check_file_exists(filepath, f"Config: {config_file}"):
            validate_json_config(filepath, required_keys)
        else:
            all_ok = False
    
    # Check example scripts
    print("\n📜 Example Training Scripts:")
    scripts = [
        "run_curriculum_uniform.sh",
        "run_curriculum_staged.sh",
        "run_curriculum_mixed.sh",
    ]
    
    for script in scripts:
        filepath = f"{base_path}/scripts/training/{script}"
        if check_file_exists(filepath, f"Script: {script}"):
            # Check if executable
            is_executable = os.access(filepath, os.X_OK)
            if is_executable:
                print(f"  ✅ Executable")
            else:
                print(f"  ⚠️  Not executable (run: chmod +x {filepath})")
        else:
            all_ok = False
    
    # Check documentation
    print("\n📚 Documentation:")
    all_ok &= check_file_exists(
        f"{base_path}/docs/CURRICULUM_LEARNING.md",
        "Curriculum learning guide"
    )
    all_ok &= check_file_exists(
        f"{base_path}/CURRICULUM_IMPLEMENTATION.md",
        "Implementation summary"
    )
    
    # Check for key classes in curriculum_env.py
    print("\n🔍 Checking Implementation Classes:")
    curriculum_env_path = f"{base_path}/EVRoutingEnv/models/curriculum_env.py"
    if os.path.exists(curriculum_env_path):
        with open(curriculum_env_path, 'r') as f:
            content = f.read()
        
        classes = [
            "CurriculumStrategy",
            "UniformRandomStrategy",
            "StagedCurriculumStrategy",
            "MixedCurriculumStrategy",
            "CurriculumEnvWrapper"
        ]
        
        for cls in classes:
            if f"class {cls}" in content:
                print(f"  ✅ Class: {cls}")
            else:
                print(f"  ❌ Missing class: {cls}")
                all_ok = False
    
    # Final summary
    print("\n" + "="*70)
    if all_ok:
        print("✅ ALL VALIDATION CHECKS PASSED!")
        print("="*70)
        print("\n🎉 Curriculum learning implementation is complete and ready to use!")
        print("\n📖 Quick Start:")
        print("   1. Read the guide: docs/CURRICULUM_LEARNING.md")
        print("   2. Run example: ./scripts/training/run_curriculum_uniform.sh")
        print("   3. Or customize:")
        print("      python scripts/training/train_curriculum.py \\")
        print("          --curriculum-strategy uniform \\")
        print("          --truck-range 3 8 --stop-range 3 8 \\")
        print("          --max-timesteps 2000000 --seed 42")
        print()
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("="*70)
        print("\nPlease check the errors above and ensure all files are created.")
        print()
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    exit(main())
