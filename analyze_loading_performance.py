"""
Minimal Performance Analysis for Image Loading

Directly measures the bottlenecks without complex dependencies.
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def benchmark_sequential_frame_processing():
    """
    Simulates what happens in process_trial.py lines 184-209
    CRITICAL BOTTLENECK: This runs on camera change and blocks UI
    """
    print("="*70)
    print("BENCHMARK 1: SEQUENTIAL FRAME PROCESSING (BLOCKS UI)")
    print("="*70)
    print("Location: process_trial.py:184-209")
    print("Triggered by: ui_data_loader.py:303 (on_camera_changed)")
    print()

    # Use actual trial data
    trial_path = Path("/Users/ivy/Documents/GitHub/pointing/trial_input/1/single_camera").resolve().resolve()

    if not trial_path.exists():
        print(f"Trial not found: {trial_path}")
        print(f"Checking if path is accessible...")
        print(f"  Exists: {trial_path.exists()}")
        print(f"  Is dir: {trial_path.is_dir()}")
        return None

    color_dir = trial_path / "color"
    depth_dir = trial_path / "depth"

    # Get frame files
    frame_files = sorted(color_dir.glob("frame_*.png"))[:30]  # Test with 30 frames

    print(f"Processing {len(frame_files)} frames sequentially...")
    print("Simulating: cv2.imread + cv2.imwrite + np.load + np.save")
    print()

    start_time = time.time()

    for i, color_file in enumerate(frame_files):
        frame_num = color_file.stem.split('_')[1]
        depth_file = depth_dir / f"frame_{frame_num}.npy"

        # Load (simulating load from original source)
        color = cv2.imread(str(color_file))
        depth = np.load(str(depth_file))

        # Save (simulating write to trial_input)
        _ = cv2.imencode('.png', color)[1]  # Simulate write
        _ = depth.tobytes()  # Simulate serialize

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{len(frame_files)} ({rate:.1f} fps) - UI BLOCKED")

    total_time = time.time() - start_time
    time_per_frame = total_time / len(frame_files)

    print()
    print(f"Results:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Time per frame: {time_per_frame*1000:.2f} ms")
    print(f"  Throughput: {len(frame_files)/total_time:.2f} fps")
    print(f"  UI BLOCKED FOR: {total_time:.2f} seconds")
    print(f"  Severity: {'CRITICAL' if total_time > 3 else 'HIGH'}")

    return {
        'method': 'sequential_processing',
        'frames': len(frame_files),
        'total_time': total_time,
        'time_per_frame_ms': time_per_frame * 1000,
        'fps': len(frame_files) / total_time,
        'ui_blocked': total_time
    }


def benchmark_single_frame_load():
    """
    Simulates what happens in ui_data_loader.py:487-516 (load_current_frame)
    """
    print("\n" + "="*70)
    print("BENCHMARK 2: SINGLE FRAME LOADING")
    print("="*70)
    print("Location: ui_data_loader.py:505-509 (load_frame)")
    print("Triggered by: User navigates to different frame")
    print()

    trial_path = Path("/Users/ivy/Documents/GitHub/pointing/trial_input/1/single_camera").resolve()
    color_dir = trial_path / "color"
    depth_dir = trial_path / "depth"

    frame_files = sorted(color_dir.glob("frame_*.png"))

    if len(frame_files) == 0:
        print(f"No frames found in {color_dir}")
        return None

    # Sample 20 frames (or fewer if less available)
    num_samples = min(20, len(frame_files))
    sample_indices = np.linspace(0, len(frame_files)-1, num_samples, dtype=int)
    sample_files = [frame_files[i] for i in sample_indices]

    print(f"Testing {len(sample_files)} random frames...")

    load_times = []
    for color_file in sample_files:
        frame_num = color_file.stem.split('_')[1]
        depth_file = depth_dir / f"frame_{frame_num}.npy"

        start = time.time()
        color = cv2.imread(str(color_file))
        depth = np.load(str(depth_file))
        load_times.append((time.time() - start) * 1000)

    avg_time = np.mean(load_times)
    min_time = np.min(load_times)
    max_time = np.max(load_times)

    print()
    print(f"Results:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Range: {min_time:.2f} - {max_time:.2f} ms")
    print(f"  Max navigation speed: {1000/avg_time:.2f} fps")
    print(f"  User experience: {'Smooth' if avg_time < 100 else 'Sluggish'}")

    return {
        'method': 'single_frame',
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'max_fps': 1000 / avg_time
    }


def benchmark_parallel_loading():
    """
    Simulates trial_input_manager.py:128-172 (batch_load_frames)
    """
    print("\n" + "="*70)
    print("BENCHMARK 3: PARALLEL BATCH LOADING")
    print("="*70)
    print("Location: trial_input_manager.py:165 (ThreadPoolExecutor max_workers=8)")
    print("Triggered by: ui_data_loader.py:561 (preload_surrounding_frames)")
    print()

    trial_path = Path("/Users/ivy/Documents/GitHub/pointing/trial_input/1/single_camera").resolve()
    color_dir = trial_path / "color"
    depth_dir = trial_path / "depth"

    frame_files = sorted(color_dir.glob("frame_*.png"))[:20]

    def load_single_frame(color_file):
        frame_num = color_file.stem.split('_')[1]
        depth_file = depth_dir / f"frame_{frame_num}.npy"
        color = cv2.imread(str(color_file))
        depth = np.load(str(depth_file))
        return color, depth

    print(f"Loading {len(frame_files)} frames in parallel (8 workers)...")

    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_single_frame, f) for f in frame_files]
        results = [f.result() for f in as_completed(futures)]
    parallel_time = (time.time() - start) * 1000

    # Compare to sequential
    start = time.time()
    for f in frame_files:
        _ = load_single_frame(f)
    sequential_time = (time.time() - start) * 1000

    speedup = sequential_time / parallel_time

    print()
    print(f"Results:")
    print(f"  Parallel (8 workers): {parallel_time:.2f} ms")
    print(f"  Sequential: {sequential_time:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time per frame: {parallel_time/len(frame_files):.2f} ms")
    print(f"  Recommendation: Increase to 16-32 workers for I/O bound tasks")

    return {
        'method': 'parallel',
        'workers': 8,
        'parallel_time_ms': parallel_time,
        'sequential_time_ms': sequential_time,
        'speedup': speedup
    }


def benchmark_cache_effectiveness():
    """
    Measures cache hit vs miss performance
    """
    print("\n" + "="*70)
    print("BENCHMARK 4: CACHE EFFECTIVENESS")
    print("="*70)
    print("Location: ui_data_loader.py:497-516 (frame_cache)")
    print("Issue: Preload only after first frame (line 536)")
    print()

    trial_path = Path("/Users/ivy/Documents/GitHub/pointing/trial_input/1/single_camera").resolve()
    color_dir = trial_path / "color"
    depth_dir = trial_path / "depth"

    frame_files = sorted(color_dir.glob("frame_*.png"))[:20]

    # Cache miss (disk I/O)
    print("Testing cache MISS (disk I/O)...")
    miss_times = []
    cache = {}

    for color_file in frame_files:
        frame_num = color_file.stem.split('_')[1]
        depth_file = depth_dir / f"frame_{frame_num}.npy"

        start = time.time()
        color = cv2.imread(str(color_file))
        depth = np.load(str(depth_file))
        miss_times.append((time.time() - start) * 1000)

        cache[frame_num] = (color, depth)

    # Cache hit (memory access)
    print("Testing cache HIT (memory access)...")
    hit_times = []
    for frame_num in cache.keys():
        start = time.time()
        _ = cache[frame_num]
        hit_times.append((time.time() - start) * 1000)

    avg_miss = np.mean(miss_times)
    avg_hit = np.mean(hit_times)
    speedup = avg_miss / avg_hit if avg_hit > 0 else 0

    # Memory usage
    sample_color, sample_depth = list(cache.values())[0]
    frame_mb = (sample_color.nbytes + sample_depth.nbytes) / (1024**2)

    print()
    print(f"Results:")
    print(f"  Cache miss (disk): {avg_miss:.2f} ms")
    print(f"  Cache hit (memory): {avg_hit:.4f} ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Memory per frame: {frame_mb:.2f} MB")
    print(f"  Preload window (20 frames): {frame_mb*20:.1f} MB")
    print(f"  Cache is highly effective - continue using")

    return {
        'method': 'cache',
        'cache_miss_ms': avg_miss,
        'cache_hit_ms': avg_hit,
        'speedup': speedup,
        'frame_size_mb': frame_mb,
        'preload_20_frames_mb': frame_mb * 20
    }


def analyze_code_issues():
    """
    Direct code analysis of the bottlenecks
    """
    print("\n" + "="*70)
    print("CODE-LEVEL BOTTLENECK ANALYSIS")
    print("="*70)

    issues = [
        {
            'priority': 'CRITICAL',
            'file': 'ui_data_loader.py',
            'lines': '292-346',
            'function': 'on_camera_changed',
            'issue': 'Synchronous process_trial blocks UI thread',
            'code_snippet': '''
    def on_camera_changed(self, event):
        # ...
        output_path = process_trial(          # Line 303 - BLOCKS HERE
            trial_path=trial_info.trial_path,
            camera_id=camera,
            output_base="trial_input",
            frame_range=None  # Process all frames
        )
        # User cannot interact with UI until this completes!
            ''',
            'impact': 'UI freezes for 2-10 seconds depending on frame count',
            'solutions': [
                'QUICK FIX: Check if trial already processed (skip if exists)',
                'BETTER FIX: Move to background thread with progress dialog',
                'BEST FIX: Lazy processing - only process frames on-demand',
                'Also consider: Parallel processing with multiprocessing.Pool'
            ]
        },
        {
            'priority': 'HIGH',
            'file': 'process_trial.py',
            'lines': '184-209',
            'function': 'process_trial (tqdm loop)',
            'issue': 'Sequential frame processing',
            'code_snippet': '''
    for frame_num in tqdm(frame_numbers, desc="Processing"):  # Line 184
        try:
            color_img = load_color_flexible(...)  # Sequential I/O
            cv2.imwrite(color_out, color_img)
            depth_img = load_depth_flexible(...)
            np.save(depth_out, depth_img)
        # No parallelization despite I/O-bound operations
            ''',
            'impact': 'Processing takes 100-300ms per frame',
            'solutions': [
                'Use multiprocessing.Pool with 4-8 workers',
                'Expected speedup: 4-6x on quad-core',
                'Keep tqdm for progress tracking'
            ]
        },
        {
            'priority': 'HIGH',
            'file': 'ui_data_loader.py',
            'lines': '536-572',
            'function': 'preload_surrounding_frames',
            'issue': 'Preload only triggers AFTER first frame loads',
            'code_snippet': '''
    def load_current_frame(self):
        # Load first frame from disk
        color, depth = self.trial_input_manager.load_frame(...)  # Line 505-509
        # ...
        # Only NOW start preloading (line 536)
        if self.cache_enabled and not cache_hit:
            self.preload_surrounding_frames(frame_idx)
            ''',
            'impact': 'User experiences slow first navigation',
            'solutions': [
                'Trigger aggressive preload immediately after camera change',
                'Preload first 20-40 frames before user interaction',
                'Don\'t wait for first frame to load'
            ]
        },
        {
            'priority': 'MEDIUM',
            'file': 'trial_input_manager.py',
            'lines': '165',
            'function': 'batch_load_frames',
            'issue': 'Only 8 workers for I/O-bound parallel loading',
            'code_snippet': '''
    with ThreadPoolExecutor(max_workers=8) as executor:  # Line 165
        # I/O bound tasks can benefit from more threads
            ''',
            'impact': 'Preloading could be faster',
            'solutions': [
                'Increase max_workers to 16-32',
                'I/O-bound tasks don\'t suffer from GIL',
                'Test optimal count based on disk speed'
            ]
        }
    ]

    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. [{issue['priority']}] {issue['file']}:{issue['lines']}")
        print(f"   Function: {issue['function']}")
        print(f"   Issue: {issue['issue']}")
        print(f"   Impact: {issue['impact']}")
        print(f"\n   Code:")
        for line in issue['code_snippet'].strip().split('\n'):
            print(f"   {line}")
        print(f"\n   Solutions:")
        for sol in issue['solutions']:
            print(f"     - {sol}")


def main():
    """Run all benchmarks and analysis"""
    print("\n" + "="*70)
    print("IMAGE LOADING PERFORMANCE ANALYSIS")
    print("Pointing Gesture Analysis System")
    print("="*70)

    results = {}

    # Run benchmarks
    results['sequential'] = benchmark_sequential_frame_processing()
    results['single_frame'] = benchmark_single_frame_load()
    results['parallel'] = benchmark_parallel_loading()
    results['cache'] = benchmark_cache_effectiveness()

    # Code analysis
    analyze_code_issues()

    # Summary recommendations
    print("\n" + "="*70)
    print("PRIORITY RECOMMENDATIONS")
    print("="*70)

    if results.get('sequential'):
        seq = results['sequential']
        print(f"\n1. [CRITICAL] Fix on_camera_changed blocking")
        print(f"   Current: UI blocks for {seq['ui_blocked']:.2f} seconds")
        print(f"   File: ui_data_loader.py:303")
        print(f"   Quick fix: Add check - skip if trial already processed")
        print(f"   Better fix: Background thread + progress dialog")

    print(f"\n2. [HIGH] Implement aggressive preloading")
    print(f"   Current: Preload starts AFTER first frame")
    print(f"   File: ui_data_loader.py:536")
    print(f"   Fix: Preload immediately after camera change")

    if results.get('parallel'):
        par = results['parallel']
        print(f"\n3. [MEDIUM] Optimize parallel loading")
        print(f"   Current: 8 workers, {par['speedup']:.1f}x speedup")
        print(f"   File: trial_input_manager.py:165")
        print(f"   Fix: Increase to 16-32 workers")

    # Save results
    output_file = "performance_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmarks': results
        }, f, indent=2)

    print(f"\n\nDetailed results saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
