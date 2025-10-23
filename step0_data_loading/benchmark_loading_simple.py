"""
Simplified Performance Benchmark - Works with Existing trial_input Data

Measures performance bottlenecks in the image loading system without
requiring original source data.

Usage:
    python benchmark_loading_simple.py
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(__file__))
from trial_input_manager import TrialInputManager


class SimpleBenchmark:
    """Simple benchmarks using existing trial_input data"""

    def __init__(self):
        # trial_input is in parent directory
        trial_input_path = Path(__file__).parent.parent / "trial_input"
        self.trial_input_manager = TrialInputManager(str(trial_input_path))
        self.results = {}

        # Find first available trial (check manually for nested structure)
        trial_input_base = trial_input_path
        self.trial_name = None
        self.camera_id = None

        # Look for trials with metadata
        for item in trial_input_base.iterdir():
            if item.is_dir() and item.name != '.DS_Store':
                # Check for single_camera structure
                single_cam = item / "single_camera"
                if single_cam.exists() and (single_cam / "metadata.json").exists():
                    self.trial_name = item.name
                    self.camera_id = None
                    break
                # Check for multi-camera structure
                for cam in item.iterdir():
                    if cam.is_dir() and (cam / "metadata.json").exists():
                        self.trial_name = item.name
                        self.camera_id = cam.name
                        break
                if self.trial_name:
                    break

        if not self.trial_name:
            raise ValueError("No trials found in trial_input/")

        print(f"Using trial: {self.trial_name}")
        if self.camera_id:
            print(f"Using camera: {self.camera_id}")

    def benchmark_sequential_loading(self, num_frames: int = 30) -> dict:
        """
        Simulate process_trial's sequential loading behavior
        This is what happens on camera change (BLOCKS UI)
        """
        print("\n" + "="*70)
        print("BENCHMARK 1: SEQUENTIAL FRAME LOADING (Simulates process_trial)")
        print("="*70)
        print("This simulates what happens when user changes camera")
        print("CRITICAL: Blocks UI thread during entire operation")

        frames = self.trial_input_manager.find_available_frames(self.trial_name, self.camera_id)
        test_frames = frames[:min(num_frames, len(frames))]

        print(f"\nLoading {len(test_frames)} frames sequentially...")

        # Simulate sequential processing (like process_trial)
        start_time = time.time()
        processed_frames = []

        for i, frame_num in enumerate(test_frames):
            # Load frame (cv2.imread + np.load)
            color, depth = self.trial_input_manager.load_frame(
                self.trial_name, self.camera_id, frame_num
            )

            # Simulate processing (like cv2.imwrite + np.save)
            # Even if files already exist, this shows the sequential bottleneck
            _ = cv2.imencode('.png', color)  # Simulate encoding
            _ = depth.tobytes()  # Simulate serialization

            processed_frames.append(frame_num)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i+1}/{len(test_frames)} frames ({rate:.1f} fps)")

        total_time = time.time() - start_time
        time_per_frame = total_time / len(test_frames)
        fps = len(test_frames) / total_time

        result = {
            'method': 'sequential_processing',
            'frames': len(test_frames),
            'total_time_sec': round(total_time, 2),
            'time_per_frame_ms': round(time_per_frame * 1000, 2),
            'fps': round(fps, 2),
            'ui_blocked_for': round(total_time, 2),
            'severity': 'CRITICAL' if total_time > 3 else 'HIGH'
        }

        print(f"\nResults:")
        print(f"  Total time: {result['total_time_sec']} seconds")
        print(f"  Per frame: {result['time_per_frame_ms']} ms")
        print(f"  Throughput: {result['fps']} fps")
        print(f"  UI BLOCKED FOR: {result['ui_blocked_for']} seconds")
        print(f"  Severity: {result['severity']}")

        self.results['sequential_loading'] = result
        return result

    def benchmark_single_frame_loading(self, num_samples: int = 20) -> dict:
        """
        Benchmark single frame loading (current frame display)
        """
        print("\n" + "="*70)
        print("BENCHMARK 2: SINGLE FRAME LOADING")
        print("="*70)
        print("This is what happens when user navigates to a new frame")

        frames = self.trial_input_manager.find_available_frames(self.trial_name, self.camera_id)
        sample_frames = frames[::max(1, len(frames)//num_samples)][:num_samples]

        print(f"\nTesting {len(sample_frames)} random frames...")

        load_times = []
        for frame_num in sample_frames:
            start = time.time()
            color, depth = self.trial_input_manager.load_frame(
                self.trial_name, self.camera_id, frame_num
            )
            elapsed = (time.time() - start) * 1000  # ms
            load_times.append(elapsed)

        avg_time = np.mean(load_times)
        min_time = np.min(load_times)
        max_time = np.max(load_times)

        result = {
            'method': 'single_frame',
            'samples': len(sample_frames),
            'avg_time_ms': round(avg_time, 2),
            'min_time_ms': round(min_time, 2),
            'max_time_ms': round(max_time, 2),
            'max_fps': round(1000 / avg_time, 2),
            'user_experience': 'Acceptable' if avg_time < 100 else 'Sluggish'
        }

        print(f"\nResults:")
        print(f"  Average: {result['avg_time_ms']} ms")
        print(f"  Range: {result['min_time_ms']} - {result['max_time_ms']} ms")
        print(f"  Max navigation speed: {result['max_fps']} fps")
        print(f"  User experience: {result['user_experience']}")

        self.results['single_frame_loading'] = result
        return result

    def benchmark_parallel_loading(self, batch_size: int = 20) -> dict:
        """
        Benchmark parallel batch loading (preloading)
        """
        print("\n" + "="*70)
        print(f"BENCHMARK 3: PARALLEL BATCH LOADING ({batch_size} frames)")
        print("="*70)
        print("This is what happens during frame preloading")

        frames = self.trial_input_manager.find_available_frames(self.trial_name, self.camera_id)
        test_frames = frames[:min(batch_size, len(frames))]

        print(f"\nLoading {len(test_frames)} frames in parallel...")

        start = time.time()
        loaded = self.trial_input_manager.batch_load_frames(
            self.trial_name, self.camera_id, test_frames
        )
        total_time = (time.time() - start) * 1000  # ms

        time_per_frame = total_time / len(test_frames)

        # Compare to sequential
        if 'single_frame_loading' in self.results:
            seq_time = self.results['single_frame_loading']['avg_time_ms'] * len(test_frames)
            speedup = seq_time / total_time
        else:
            speedup = 0

        result = {
            'method': 'parallel_batch',
            'batch_size': len(test_frames),
            'total_time_ms': round(total_time, 2),
            'time_per_frame_ms': round(time_per_frame, 2),
            'workers': 8,
            'speedup_vs_sequential': round(speedup, 2) if speedup > 0 else 'N/A',
            'recommendation': 'Increase workers to 16-32 for better I/O parallelism'
        }

        print(f"\nResults:")
        print(f"  Total: {result['total_time_ms']} ms")
        print(f"  Per frame: {result['time_per_frame_ms']} ms")
        print(f"  Workers: {result['workers']}")
        if speedup > 0:
            print(f"  Speedup: {result['speedup_vs_sequential']}x vs sequential")
        print(f"  {result['recommendation']}")

        self.results['parallel_loading'] = result
        return result

    def benchmark_cache_effectiveness(self, num_frames: int = 20) -> dict:
        """
        Test cache vs disk performance
        """
        print("\n" + "="*70)
        print("BENCHMARK 4: CACHE EFFECTIVENESS")
        print("="*70)

        frames = self.trial_input_manager.find_available_frames(self.trial_name, self.camera_id)
        test_frames = frames[:min(num_frames, len(frames))]

        # First load (cache miss - disk I/O)
        print(f"\nTesting cache MISS (disk I/O) on {len(test_frames)} frames...")
        miss_times = []
        for frame_num in test_frames:
            start = time.time()
            color, depth = self.trial_input_manager.load_frame(
                self.trial_name, self.camera_id, frame_num
            )
            miss_times.append((time.time() - start) * 1000)

        # Build cache
        cache = {}
        for frame_num in test_frames:
            color, depth = self.trial_input_manager.load_frame(
                self.trial_name, self.camera_id, frame_num
            )
            cache[frame_num] = (color, depth)

        # Cache hit (memory access)
        print(f"Testing cache HIT (memory access)...")
        hit_times = []
        for frame_num in test_frames:
            start = time.time()
            _ = cache[frame_num]
            hit_times.append((time.time() - start) * 1000)

        avg_miss = np.mean(miss_times)
        avg_hit = np.mean(hit_times)
        speedup = avg_miss / avg_hit if avg_hit > 0 else 0

        # Calculate memory usage
        sample_color, sample_depth = cache[test_frames[0]]
        frame_mb = (sample_color.nbytes + sample_depth.nbytes) / (1024**2)

        result = {
            'method': 'caching',
            'cache_miss_ms': round(avg_miss, 2),
            'cache_hit_ms': round(avg_hit, 4),
            'speedup': round(speedup, 1),
            'frame_size_mb': round(frame_mb, 2),
            'preload_window_frames': 20,
            'preload_window_mb': round(frame_mb * 20, 1),
            'recommendation': 'Cache is highly effective - current window size is good'
        }

        print(f"\nResults:")
        print(f"  Disk load: {result['cache_miss_ms']} ms")
        print(f"  Cache load: {result['cache_hit_ms']} ms")
        print(f"  Speedup: {result['speedup']}x")
        print(f"  Memory per frame: {result['frame_size_mb']} MB")
        print(f"  Preload window (20 frames): {result['preload_window_mb']} MB")
        print(f"  {result['recommendation']}")

        self.results['cache'] = result
        return result

    def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        print("\n" + "="*70)
        print("IMAGE LOADING PERFORMANCE BENCHMARK")
        print("="*70)

        # Run benchmarks
        self.benchmark_sequential_loading(num_frames=30)
        self.benchmark_single_frame_loading(num_samples=20)
        self.benchmark_parallel_loading(batch_size=20)
        self.benchmark_cache_effectiveness(num_frames=20)

        # Generate analysis
        self.print_analysis()
        self.save_results()

    def print_analysis(self):
        """Print detailed analysis with specific recommendations"""
        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS & RECOMMENDATIONS")
        print("="*70)

        # Analysis 1: Sequential processing bottleneck
        if 'sequential_loading' in self.results:
            seq = self.results['sequential_loading']
            print(f"\n1. CRITICAL: Sequential Processing on Camera Change")
            print(f"   Location: ui_data_loader.py:292-346 (on_camera_changed)")
            print(f"   Problem: process_trial runs synchronously")
            print(f"   Impact: UI frozen for {seq['ui_blocked_for']} seconds")
            print(f"   Frames: {seq['frames']} @ {seq['time_per_frame_ms']} ms each")
            print(f"\n   SOLUTIONS:")
            print(f"   A. Immediate fix - Check if already processed:")
            print(f"      - Add check at line 303: if trial already in trial_input, skip")
            print(f"      - Current behavior: reprocesses on every camera change")
            print(f"   B. Background processing:")
            print(f"      - Move process_trial to threading.Thread")
            print(f"      - Show progress dialog with cancel option")
            print(f"      - Load first frame immediately, process rest in background")
            print(f"   C. Lazy processing:")
            print(f"      - Only process frames when actually accessed")
            print(f"      - Check if frame exists in trial_input before loading")
            print(f"   D. Parallel processing:")
            print(f"      - Use multiprocessing.Pool for frame processing")
            print(f"      - Expected speedup: 4-8x on multi-core systems")

        # Analysis 2: Single frame loading
        if 'single_frame_loading' in self.results:
            single = self.results['single_frame_loading']
            print(f"\n2. Single Frame Loading Performance")
            print(f"   Location: ui_data_loader.py:487-542 (load_current_frame)")
            print(f"   Current: {single['avg_time_ms']} ms average")
            print(f"   Status: {single['user_experience']}")
            if single['avg_time_ms'] < 100:
                print(f"   No action needed - performance is good")
            else:
                print(f"   Consider SSD upgrade or file format optimization")

        # Analysis 3: Parallel loading
        if 'parallel_loading' in self.results:
            par = self.results['parallel_loading']
            print(f"\n3. Parallel Batch Loading (Preloading)")
            print(f"   Location: trial_input_manager.py:128-172 (batch_load_frames)")
            print(f"   Current workers: {par['workers']}")
            print(f"   Current speedup: {par['speedup_vs_sequential']}x")
            print(f"\n   SOLUTIONS:")
            print(f"   - Increase workers from 8 to 16-32 (line 165)")
            print(f"   - I/O bound tasks benefit from more threads")
            print(f"   - Test optimal count based on disk speed")

        # Analysis 4: Cache
        if 'cache' in self.results:
            cache = self.results['cache']
            print(f"\n4. Caching Strategy")
            print(f"   Location: ui_data_loader.py:543-572 (preload_surrounding_frames)")
            print(f"   Speedup: {cache['speedup']}x faster than disk")
            print(f"   Memory usage: {cache['preload_window_mb']} MB for 20 frames")
            print(f"\n   CURRENT ISSUE:")
            print(f"   - Preload only happens AFTER first frame loads (line 536)")
            print(f"   - First frame navigation has no preload benefit")
            print(f"\n   SOLUTIONS:")
            print(f"   - Trigger aggressive preload on trial selection")
            print(f"   - Preload first 20 frames immediately after camera change")
            print(f"   - Consider increasing preload window to 40-50 frames")

        # Overall recommendations
        print("\n" + "="*70)
        print("PRIORITY ACTION ITEMS")
        print("="*70)

        if 'sequential_loading' in self.results:
            seq = self.results['sequential_loading']
            if seq['ui_blocked_for'] > 2:
                print("\n[CRITICAL] Fix camera change blocking:")
                print("  File: ui_data_loader.py")
                print("  Lines: 292-346 (on_camera_changed)")
                print("  Quick fix:")
                print("    1. Check if trial_input already has processed data")
                print("    2. Skip process_trial if already exists")
                print("    3. Add force_reprocess flag for manual refresh")
                print("  Better fix:")
                print("    1. Move process_trial to background thread")
                print("    2. Load first frame immediately")
                print("    3. Show progress bar for background processing")

        print("\n[HIGH] Improve preloading strategy:")
        print("  File: ui_data_loader.py")
        print("  Lines: 543-572 (preload_surrounding_frames)")
        print("  Changes:")
        print("    1. Call preload immediately after camera change")
        print("    2. Preload first 20-40 frames aggressively")
        print("    3. Don't wait for first frame to load")

        print("\n[MEDIUM] Optimize parallel loading:")
        print("  File: trial_input_manager.py")
        print("  Line: 165")
        print("  Change: ThreadPoolExecutor(max_workers=8) -> max_workers=16")

        print("\n" + "="*70)

    def save_results(self):
        """Save results to JSON file"""
        output_file = "benchmark_results.json"

        report = {
            'trial': self.trial_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmarks': self.results,
            'summary': {
                'critical_issues': [],
                'high_priority': [],
                'medium_priority': []
            }
        }

        # Categorize issues
        if 'sequential_loading' in self.results:
            seq = self.results['sequential_loading']
            if seq['ui_blocked_for'] > 2:
                report['summary']['critical_issues'].append({
                    'issue': 'UI blocks during camera change',
                    'file': 'ui_data_loader.py:292-346',
                    'duration': f"{seq['ui_blocked_for']}s",
                    'solution': 'Move process_trial to background thread'
                })

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main entry point"""
    try:
        benchmark = SimpleBenchmark()
        benchmark.run_all_benchmarks()

    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease ensure trial_input/ folder has processed data.")
        print("Run ui_data_loader.py first to process a trial.")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
