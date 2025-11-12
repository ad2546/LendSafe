"""
Monitor fine-tuning progress in real-time
"""
import time
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta

LOG_FILE = Path(__file__).parent.parent / "finetune_log.txt"

def parse_progress(log_content):
    """Extract training progress from log"""

    # Look for progress indicators
    progress_pattern = r'(\d+)%\|.*?\| (\d+)/(\d+)'
    loss_pattern = r"'loss': ([\d.]+)"
    step_pattern = r"'step': (\d+)"

    progress_match = re.findall(progress_pattern, log_content)
    loss_matches = re.findall(loss_pattern, log_content)
    step_matches = re.findall(step_pattern, log_content)

    if progress_match:
        latest = progress_match[-1]
        percent = int(latest[0])
        current = int(latest[1])
        total = int(latest[2])
    else:
        # Check for step numbers
        if step_matches:
            current = int(step_matches[-1])
            total = 255
            percent = int((current / total) * 100)
        else:
            percent = 0
            current = 0
            total = 255

    latest_loss = float(loss_matches[-1]) if loss_matches else None

    return {
        'percent': percent,
        'current_step': current,
        'total_steps': total,
        'latest_loss': latest_loss
    }


def estimate_time_remaining(current_step, total_steps, elapsed_seconds):
    """Estimate time remaining based on current progress"""

    if current_step == 0:
        return "Calculating..."

    steps_remaining = total_steps - current_step
    avg_time_per_step = elapsed_seconds / current_step
    estimated_remaining = steps_remaining * avg_time_per_step

    return str(timedelta(seconds=int(estimated_remaining)))


def format_duration(seconds):
    """Format duration in human-readable format"""
    return str(timedelta(seconds=int(seconds)))


def get_progress_bar(percent, width=40):
    """Create ASCII progress bar"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return bar


def clear_screen():
    """Clear terminal screen"""
    print('\033[2J\033[H', end='')


def monitor_training(update_interval=10):
    """Monitor training progress"""

    print("="*70)
    print("🔍 GRANITE FINE-TUNING MONITOR")
    print("="*70)
    print(f"Log file: {LOG_FILE}")
    print(f"Update interval: {update_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")

    start_time = time.time()
    last_step = 0
    last_update_time = start_time

    try:
        while True:
            if not LOG_FILE.exists():
                print("⏳ Waiting for log file to be created...")
                time.sleep(update_interval)
                continue

            # Read log file
            with open(LOG_FILE, 'r') as f:
                log_content = f.read()

            # Parse progress
            progress = parse_progress(log_content)

            # Calculate timings
            current_time = time.time()
            elapsed = current_time - start_time

            # Estimate time remaining
            if progress['current_step'] > last_step:
                last_step = progress['current_step']
                last_update_time = current_time

            time_remaining = estimate_time_remaining(
                progress['current_step'],
                progress['total_steps'],
                elapsed
            )

            # Display progress
            clear_screen()

            print("="*70)
            print("🚀 GRANITE FINE-TUNING PROGRESS")
            print("="*70)

            # Progress bar
            progress_bar = get_progress_bar(progress['percent'])
            print(f"\n{progress_bar} {progress['percent']}%")

            # Steps
            print(f"\n📊 Steps: {progress['current_step']}/{progress['total_steps']}")

            # Loss
            if progress['latest_loss']:
                print(f"📉 Latest Loss: {progress['latest_loss']:.4f}")

            # Time
            print(f"\n⏱️  Elapsed: {format_duration(elapsed)}")
            print(f"⏰ Estimated Remaining: {time_remaining}")

            # Status
            if progress['current_step'] == 0:
                status = "🔄 Initializing first training step..."
                status_detail = "The first step takes 5-15 minutes (graph compilation, memory allocation)"
            elif progress['current_step'] < 50:
                status = "🌡️  Warming up..."
                status_detail = "Learning rate gradually increasing"
            elif progress['current_step'] < 200:
                status = "🚀 Main training phase"
                status_detail = "Model learning from data"
            elif progress['current_step'] < 255:
                status = "🎯 Converging..."
                status_detail = "Final refinement phase"
            else:
                status = "✅ Training complete!"
                status_detail = "Saving model..."

            print(f"\n{status}")
            print(f"   {status_detail}")

            # Performance indicators
            if progress['current_step'] > 0:
                steps_per_second = progress['current_step'] / elapsed
                print(f"\n⚡ Performance: {steps_per_second:.2f} steps/second")

            # Check if training is stalled
            time_since_update = current_time - last_update_time
            if time_since_update > 300 and progress['current_step'] > 0:  # 5 minutes
                print(f"\n⚠️  Warning: No progress for {int(time_since_update/60)} minutes")

            # Footer
            print("\n" + "="*70)
            print(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
            print("Press Ctrl+C to stop monitoring")
            print("="*70)

            # Check if complete
            if progress['current_step'] >= progress['total_steps']:
                print("\n🎉 Training completed successfully!")
                print("\nNext steps:")
                print("1. Check finetune_log.txt for detailed results")
                print("2. Run: python scripts/evaluate_model.py")
                break

            # Wait before next update
            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped by user")
        print(f"Training is still running in background (PID: check with 'ps aux | grep finetune')")
        print(f"View log: tail -f {LOG_FILE}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Training may still be running. Check manually.")


if __name__ == "__main__":
    # Check for custom update interval
    update_interval = 10
    if len(sys.argv) > 1:
        try:
            update_interval = int(sys.argv[1])
        except ValueError:
            print(f"Invalid interval. Using default: {update_interval} seconds")

    monitor_training(update_interval)
