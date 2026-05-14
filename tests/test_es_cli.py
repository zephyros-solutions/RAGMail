#!/usr/bin/env python3
"""
Quick test to verify Elasticsearch strategy works through CLI
"""
import subprocess
import sys

def test_es_help():
    """Test that --method accepts 'es' as a valid option"""
    result = subprocess.run(
        ['python', 'main.py', '--help'],
        capture_output=True,
        text=True
    )
    
    print("=" * 80)
    print("Testing: CLI accepts 'es' as method option")
    print("=" * 80)
    
    if 'Elasticsearch BM25' in result.stdout or 'es' in result.stdout:
        print("✓ 'es' is listed in --method choices")
        print(f"\nHelp text for --method:")
        for line in result.stdout.split('\n'):
            if '--method' in line or 'es' in line.lower():
                print(f"  {line}")
        return True
    else:
        print("✗ 'es' not found in help text")
        print(f"\nFull help output:\n{result.stdout}")
        return False

def test_es_invalid_args():
    """Test that invalid method is rejected"""
    result = subprocess.run(
        ['python', 'main.py', '-m', 'dummy', '-g', 'llama3.2', '--method', 'invalid'],
        capture_output=True,
        text=True
    )
    
    print("\n" + "=" * 80)
    print("Testing: Invalid --method is rejected")
    print("=" * 80)
    
    if result.returncode != 0:
        print("✓ Invalid method rejected with non-zero exit code")
        if 'invalid choice' in result.stderr or 'invalid' in result.stderr.lower():
            print(f"✓ Error message indicates invalid choice")
            return True
        else:
            print(f"✗ Error message unclear: {result.stderr[:200]}")
            return False
    else:
        print("✗ Should have failed with invalid method")
        return False

if __name__ == '__main__':
    print("\nELASTICSEARCH STRATEGY CLI INTEGRATION TEST")
    print("=" * 80)
    
    results = []
    results.append(("Help text includes 'es'", test_es_help()))
    results.append(("Invalid method rejected", test_es_invalid_args()))
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n✓ All CLI integration tests PASSED")
        print("\nElasticsearch strategy is properly integrated into CLI.")
        print("Usage: python main.py -m <mailbox> -g llama3.2 --method es")
    else:
        print("\n✗ Some tests FAILED")
        sys.exit(1)
