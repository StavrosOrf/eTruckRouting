"""
Master test runner for all unit tests.
Runs environment and TD3 component tests.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_environment
import test_td3_components


def main():
    """Run all test suites."""
    print("\n" + "="*80)
    print("RUNNING ALL UNIT TESTS")
    print("="*80)
    
    all_passed = True
    
    # Run environment tests
    print("\n" + "="*80)
    print("SUITE 1: ENVIRONMENT TESTS")
    print("="*80)
    env_passed = test_environment.run_all_tests()
    all_passed = all_passed and env_passed
    
    # Run TD3 component tests
    print("\n" + "="*80)
    print("SUITE 2: TD3 COMPONENT TESTS")
    print("="*80)
    td3_passed = test_td3_components.run_all_tests()
    all_passed = all_passed and td3_passed
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    print(f"Environment Tests: {'✓ PASSED' if env_passed else '✗ FAILED'}")
    print(f"TD3 Component Tests: {'✓ PASSED' if td3_passed else '✗ FAILED'}")
    print()
    
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
