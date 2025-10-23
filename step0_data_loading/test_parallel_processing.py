#!/usr/bin/env python3
"""
Test script for parallel processing refactoring

This script tests the refactored process_trial.py with different worker counts
to demonstrate the performance improvements.

Usage:
    python test_parallel_processing.py <trial_path> [camera_id]
"""

import os
import sys
import time
from process_trial import process_trial

def test_performance(trial_path: str, camera_id=None, test_workers=[1, 2, 4, 8]):
    """
    Test processing performance with different worker counts

    Args:
        trial_path: Path to trial folder
        camera_id: Camera ID (None for single-camera)
        test_workers: List of worker counts to test
    """
    print("="*80)
    print("PARALLEL PROCESSING PERFORMANCE TEST")
    print("="*80)
    print(f"\nTrial: {trial_path}")
    if camera_id:
        print(f"Camera: {camera_id}")
    print("\nTesting different worker counts...\n")

    results = []

    for num_workers in test_workers:
        print("\n" + "="*80)
        print(f"Testing with {num_workers} worker(s)")
        print("="*80)

        try:
            # Process trial
            start = time.time()
            output_path = process_trial(
                trial_path,
                camera_id,
                output_base="trial_input_test",
                num_workers=num_workers
            )
            elapsed = time.time() - start

            # Read metadata to get stats
            import json
            metadata_path = os.path.join(output_path, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            results.append({
                'workers': num_workers,
                'time': elapsed,
                'fps': metadata['processing_fps'],
                'total_frames': metadata['total_frames'],
                'successful': metadata['successful']
            })

        except Exception as e:
            print(f"\nERROR with {num_workers} workers: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\n{'Workers':<10} {'Time (s)':<12} {'FPS':<12} {'Speedup':<12} {'Frames':<10}")
    print("-"*80)

    baseline_time = results[0]['time'] if results else 0

    for result in results:
        speedup = baseline_time / result['time'] if result['time'] > 0 else 0
        print(f"{result['workers']:<10} {result['time']:<12.2f} {result['fps']:<12.1f} "
              f"{speedup:<12.2f}x {result['successful']:<10}")

    print("\n" + "="*80)

    if len(results) >= 2:
        best = max(results, key=lambda x: x['fps'])
        print(f"\nBest performance: {best['workers']} workers at {best['fps']:.1f} fps")
        print(f"Speedup vs sequential: {best['fps'] / results[0]['fps']:.2f}x")
        print(f"Time reduction: {(1 - best['time']/results[0]['time'])*100:.1f}%")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nERROR: Trial path required")
        print("\nUsage:")
        print("  python test_parallel_processing.py <trial_path> [camera_id]")
        print("\nExamples:")
        print("  python test_parallel_processing.py sample_raw_data/trial_1 cam1")
        print("  python test_parallel_processing.py sample_raw_data/1")
        sys.exit(1)

    trial_path = sys.argv[1]
    camera_id = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.isdir(trial_path):
        print(f"ERROR: Trial path does not exist: {trial_path}")
        sys.exit(1)

    # Test with 1, 2, 4, and 8 workers
    test_performance(trial_path, camera_id, test_workers=[1, 2, 4, 8])


if __name__ == "__main__":
    main()
