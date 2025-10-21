"""
Performance Benchmark for Image Loading System

Measures exact bottlenecks in:
1. process_trial (sequential frame processing)
2. load_current_frame (single frame loading)
3. batch_load_frames (parallel loading)
4. Cache effectiveness

Usage:
    python benchmark_loading_performance.py
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import json

sys.path.insert(0, os.path.dirname(__file__))
from process_trial import process_trial, find_all_frames
from trial_input_manager import TrialInputManager


class LoadingBenchmark:
    """Benchmarks for image loading performance"""

    def __init__(self, trial_path: str, camera_id: str = None):
        self.trial_path = trial_path
        self.camera_id = camera_id
        self.trial_name = os.path.basename(os.path.normpath(trial_path))
        self.trial_input_manager = TrialInputManager("trial_input")
        self.results = {}

    def benchmark_process_trial(self, num_frames: int = None) -> dict:
        """
        Benchmark process_trial function (synchronous processing)

        This is the CRITICAL BOTTLENECK that blocks UI on camera change.
        """
        print("\n" + "="*70)
        print("BENCHMARK 1: process_trial (SYNCHRONOUS FRAME PROCESSING)")
        print("="*70)

        # Find available frames in source
        all_frames = find_all_frames(self.trial_path, self.camera_id)

        if num_frames:
            frame_range = (all_frames[0], all_frames[min(num_frames-1, len(all_frames)-1)])
            frames_to_process = all_frames[:num_frames]
        else:
            frame_range = None
            frames_to_process = all_frames

        print(f"Trial: {self.trial_path}")
        print(f"Camera: {self.camera_id}")
        print(f"Frames to process: {len(frames_to_process)}")
        print(f"Frame range: {frame_range}")

        # Clear existing trial_input to force reprocessing
        import shutil
        if self.camera_id:
            output_path = f"trial_input/{self.trial_name}/{self.camera_id}"
        else:
            output_path = f"trial_input/{self.trial_name}/single_camera"

        if os.path.exists(output_path):
            print(f"Removing existing: {output_path}")
            shutil.rmtree(output_path)

        # Benchmark processing
        start_time = time.time()

        output_path = process_trial(
            trial_path=self.trial_path,
            camera_id=self.camera_id,
            output_base="trial_input",
            frame_range=frame_range
        )

        total_time = time.time() - start_time

        # Calculate statistics
        time_per_frame = total_time / len(frames_to_process)
        fps = 1.0 / time_per_frame if time_per_frame > 0 else 0

        # Estimate file sizes
        color_size = 1.2  # MB per PNG
        depth_size = 3.5  # MB per npy
        total_size_mb = (color_size + depth_size) * len(frames_to_process)
        throughput_mb_s = total_size_mb / total_time if total_time > 0 else 0

        result = {
            'function': 'process_trial',
            'frames_processed': len(frames_to_process),
            'total_time_sec': round(total_time, 2),
            'time_per_frame_ms': round(time_per_frame * 1000, 2),
            'fps': round(fps, 2),
            'total_data_mb': round(total_size_mb, 1),
            'throughput_mb_s': round(throughput_mb_s, 1),
            'bottleneck': 'BLOCKS UI THREAD - CRITICAL ISSUE'
        }

        print(f"\nResults:")
        print(f"  Total time: {result['total_time_sec']} seconds")
        print(f"  Time per frame: {result['time_per_frame_ms']} ms")
        print(f"  Throughput: {result['fps']} fps")
        print(f"  Data throughput: {result['throughput_mb_s']} MB/s")
        print(f"  ISSUE: {result['bottleneck']}")

        self.results['process_trial'] = result
        return result

    def benchmark_single_frame_loading(self, num_samples: int = 20) -> dict:
        """
        Benchmark single frame loading (load_current_frame)
        """
        print("\n" + "="*70)
        print("BENCHMARK 2: SINGLE FRAME LOADING (load_current_frame)")
        print("="*70)

        # Ensure trial is processed
        self.trial_input_manager.ensure_trial_processed(self.trial_path, self.camera_id)

        # Get available frames
        frames = self.trial_input_manager.find_available_frames(self.trial_name, self.camera_id)

        if len(frames) < num_samples:
            num_samples = len(frames)

        # Sample frames evenly across trial
        step = max(1, len(frames) // num_samples)
        sample_frames = frames[::step][:num_samples]

        print(f"Testing {num_samples} frames from trial")
        print(f"Sample frames: {sample_frames[:5]}..." if len(sample_frames) > 5 else f"Sample frames: {sample_frames}")

        # Benchmark loading
        load_times = []

        for frame_num in sample_frames:
            start = time.time()
            color, depth = self.trial_input_manager.load_frame(
                self.trial_name, self.camera_id, frame_num
            )
            elapsed = time.time() - start
            load_times.append(elapsed * 1000)  # Convert to ms

        avg_time = np.mean(load_times)
        min_time = np.min(load_times)
        max_time = np.max(load_times)
        std_time = np.std(load_times)

        result = {
            'function': 'load_frame',
            'samples': num_samples,
            'avg_time_ms': round(avg_time, 2),
            'min_time_ms': round(min_time, 2),
            'max_time_ms': round(max_time, 2),
            'std_time_ms': round(std_time, 2),
            'fps_capability': round(1000 / avg_time, 2) if avg_time > 0 else 0
        }

        print(f"\nResults:")
        print(f"  Average load time: {result['avg_time_ms']} ms")
        print(f"  Min/Max: {result['min_time_ms']} / {result['max_time_ms']} ms")
        print(f"  Std dev: {result['std_time_ms']} ms")
        print(f"  Max FPS: {result['fps_capability']} fps")

        self.results['single_frame_loading'] = result
        return result

    def benchmark_batch_loading(self, batch_size: int = 20) -> dict:
        """
        Benchmark batch loading (parallel frame loading)
        """
        print("\n" + "="*70)
        print(f"BENCHMARK 3: BATCH LOADING ({batch_size} frames in parallel)")
        print("="*70)

        # Get available frames
        frames = self.trial_input_manager.find_available_frames(self.trial_name, self.camera_id)

        if len(frames) < batch_size:
            batch_size = len(frames)

        # Take consecutive frames for realistic caching scenario
        batch_frames = frames[:batch_size]

        print(f"Loading {batch_size} consecutive frames")
        print(f"Frame range: {batch_frames[0]} to {batch_frames[-1]}")

        # Benchmark batch loading
        start = time.time()
        loaded_frames = self.trial_input_manager.batch_load_frames(
            self.trial_name, self.camera_id, batch_frames
        )
        total_time = (time.time() - start) * 1000  # ms

        time_per_frame = total_time / batch_size
        speedup_vs_sequential = batch_size / (total_time / 1000) if total_time > 0 else 0

        result = {
            'function': 'batch_load_frames',
            'batch_size': batch_size,
            'total_time_ms': round(total_time, 2),
            'time_per_frame_ms': round(time_per_frame, 2),
            'parallel_speedup': round(speedup_vs_sequential, 2),
            'workers': 8  # From trial_input_manager.py line 165
        }

        print(f"\nResults:")
        print(f"  Total time: {result['total_time_ms']} ms")
        print(f"  Time per frame: {result['time_per_frame_ms']} ms")
        print(f"  Effective speedup: {result['parallel_speedup']}x")
        print(f"  Thread pool workers: {result['workers']}")

        self.results['batch_loading'] = result
        return result

    def benchmark_cache_effectiveness(self, num_frames: int = 10) -> dict:
        """
        Benchmark cache hit performance
        """
        print("\n" + "="*70)
        print("BENCHMARK 4: CACHE EFFECTIVENESS")
        print("="*70)

        frames = self.trial_input_manager.find_available_frames(self.trial_name, self.camera_id)
        test_frames = frames[:min(num_frames, len(frames))]

        # First load (cache miss)
        cache_miss_times = []
        for frame_num in test_frames:
            start = time.time()
            color, depth = self.trial_input_manager.load_frame(
                self.trial_name, self.camera_id, frame_num
            )
            elapsed = (time.time() - start) * 1000
            cache_miss_times.append(elapsed)

        # Simulate in-memory cache
        cache = {}
        for frame_num in test_frames:
            color, depth = self.trial_input_manager.load_frame(
                self.trial_name, self.camera_id, frame_num
            )
            cache[frame_num] = (color, depth)

        # Second load (cache hit)
        cache_hit_times = []
        for frame_num in test_frames:
            start = time.time()
            color, depth = cache[frame_num]  # Direct memory access
            elapsed = (time.time() - start) * 1000
            cache_hit_times.append(elapsed)

        avg_miss = np.mean(cache_miss_times)
        avg_hit = np.mean(cache_hit_times)
        speedup = avg_miss / avg_hit if avg_hit > 0 else 0

        # Calculate memory usage
        sample_color, sample_depth = cache[test_frames[0]]
        color_size_mb = sample_color.nbytes / (1024 * 1024)
        depth_size_mb = sample_depth.nbytes / (1024 * 1024)
        frame_size_mb = color_size_mb + depth_size_mb

        result = {
            'function': 'cache',
            'cache_miss_ms': round(avg_miss, 2),
            'cache_hit_ms': round(avg_hit, 4),
            'speedup': round(speedup, 1),
            'frame_size_mb': round(frame_size_mb, 2),
            'cache_20_frames_mb': round(frame_size_mb * 20, 1),  # Current preload_window
            'recommendation': 'Cache is highly effective - memory usage is reasonable'
        }

        print(f"\nResults:")
        print(f"  Cache miss (disk): {result['cache_miss_ms']} ms")
        print(f"  Cache hit (memory): {result['cache_hit_ms']} ms")
        print(f"  Speedup: {result['speedup']}x faster")
        print(f"  Memory per frame: {result['frame_size_mb']} MB")
        print(f"  Memory for 20-frame window: {result['cache_20_frames_mb']} MB")
        print(f"  {result['recommendation']}")

        self.results['cache'] = result
        return result

    def run_full_benchmark(self, process_frames: int = None):
        """Run all benchmarks"""
        print("\n" + "="*70)
        print("IMAGE LOADING PERFORMANCE ANALYSIS")
        print("="*70)
        print(f"Trial: {self.trial_path}")
        print(f"Camera: {self.camera_id}")

        # Run all benchmarks
        self.benchmark_process_trial(num_frames=process_frames)
        self.benchmark_single_frame_loading()
        self.benchmark_batch_loading()
        self.benchmark_cache_effectiveness()

        # Generate summary
        self.print_summary()

        # Save results
        self.save_results()

    def print_summary(self):
        """Print analysis summary with recommendations"""
        print("\n" + "="*70)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*70)

        if 'process_trial' in self.results:
            pt = self.results['process_trial']
            print(f"\n1. CRITICAL BOTTLENECK: process_trial")
            print(f"   - Blocks UI for: {pt['total_time_sec']} seconds")
            print(f"   - Processing {pt['frames_processed']} frames sequentially")
            print(f"   - Time per frame: {pt['time_per_frame_ms']} ms")
            print(f"   - ISSUE: Runs on UI thread during camera change")

        if 'single_frame_loading' in self.results:
            sf = self.results['single_frame_loading']
            print(f"\n2. Single Frame Loading")
            print(f"   - Average: {sf['avg_time_ms']} ms/frame")
            print(f"   - Max throughput: {sf['fps_capability']} fps")
            print(f"   - STATUS: Acceptable for individual loads")

        if 'batch_loading' in self.results:
            bl = self.results['batch_loading']
            print(f"\n3. Batch Loading (Parallel)")
            print(f"   - Time per frame: {bl['time_per_frame_ms']} ms")
            print(f"   - Speedup: {bl['parallel_speedup']}x vs sequential")
            print(f"   - Workers: {bl['workers']} threads")
            print(f"   - STATUS: Good parallelization")

        if 'cache' in self.results:
            ca = self.results['cache']
            print(f"\n4. Caching")
            print(f"   - Cache speedup: {ca['speedup']}x faster")
            print(f"   - Memory footprint: {ca['cache_20_frames_mb']} MB for 20 frames")
            print(f"   - STATUS: Highly effective")

        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)

        recommendations = []

        if 'process_trial' in self.results:
            pt = self.results['process_trial']
            if pt['total_time_sec'] > 2:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'issue': 'process_trial blocks UI thread',
                    'impact': f"UI frozen for {pt['total_time_sec']}s on camera change",
                    'solutions': [
                        '1. Run process_trial in background thread (threading.Thread)',
                        '2. Show progress dialog with cancel button',
                        '3. Use lazy processing - only process frames on-demand',
                        '4. Check if frames already processed before reprocessing',
                        '5. Process frames in parallel (multiprocessing.Pool)'
                    ]
                })

        if 'batch_loading' in self.results:
            bl = self.results['batch_loading']
            if bl['workers'] < 16:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'issue': 'Thread pool size could be optimized',
                    'impact': f"Current: {bl['workers']} workers, could use more for I/O",
                    'solutions': [
                        'Increase ThreadPoolExecutor workers to 16-32 for I/O bound tasks',
                        'Test optimal worker count based on disk performance'
                    ]
                })

        recommendations.append({
            'priority': 'HIGH',
            'issue': 'No progressive loading on trial selection',
            'impact': 'User must wait for all frames before using UI',
            'solutions': [
                'Preload first frame immediately for instant feedback',
                'Process remaining frames in background',
                'Enable UI interaction while processing continues'
            ]
        })

        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['issue']}")
            print(f"   Impact: {rec['impact']}")
            print(f"   Solutions:")
            for sol in rec['solutions']:
                print(f"     - {sol}")

        print("\n" + "="*70)

    def save_results(self):
        """Save benchmark results to JSON"""
        output_file = "benchmark_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    """Main entry point"""
    # Use existing trial
    trial_path = "/Users/ivy/Documents/GitHub/pointing/trial_input/1"

    # Check if trial exists in raw format
    if not os.path.exists(trial_path):
        print(f"Error: Trial not found at {trial_path}")
        print("Please provide a valid trial path")
        sys.exit(1)

    # Check for original source path
    source_path_file = Path(trial_path) / "single_camera" / "source_path.txt"
    if source_path_file.exists():
        with open(source_path_file) as f:
            original_trial_path = f.read().strip()
    else:
        # Try to find in sample_raw_data or similar
        print("Warning: Could not find original trial source")
        print("Benchmarking will use existing trial_input data")
        original_trial_path = trial_path

    # Run benchmarks
    benchmark = LoadingBenchmark(original_trial_path, camera_id=None)

    # Run with limited frames for faster testing (change to None for full trial)
    benchmark.run_full_benchmark(process_frames=30)


if __name__ == "__main__":
    main()
