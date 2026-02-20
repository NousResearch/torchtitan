#!/usr/bin/env python3
"""
Run GRPO training and validate health.

This script:
1. Submits a SLURM job
2. Monitors training health
3. Reports pass/fail
4. Writes results to file

Usage:
    python run_and_validate.py online_multinode_vllm_test.slurm --steps 50
    python run_and_validate.py online_multinode_vllm_test.slurm --auto-kill-on-critical
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from monitor_training import HealthMonitor, tail_log_file


def submit_job(slurm_script: str, env_vars: dict = None) -> tuple[str, Path]:
    cmd = ["sbatch"]

    if env_vars:
        for key, val in env_vars.items():
            cmd.extend(["--export", f"{key}={val}"])

    cmd.append(slurm_script)

    print(f"Submitting job: {slurm_script}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Failed to submit job: {result.stderr}")
        sys.exit(1)

    # Extract job ID
    job_id = result.stdout.strip().split()[-1]
    print(f"Job submitted: {job_id}")

    log_file = find_log_file(job_id)

    return job_id, log_file


def find_log_file(job_id: str, wait_seconds: int = 60) -> Path:
    search_paths = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
        Path.cwd().parent.parent.parent,
    ]

    patterns = [f"{job_id}.out", f"slurm-{job_id}.out"]

    print(f"Looking for log file...")

    for _ in range(wait_seconds):
        for search_path in search_paths:
            if not search_path.exists():
                continue

            logs_dir = search_path / "logs"
            if logs_dir.exists():
                for pattern in patterns:
                    candidates = list(logs_dir.glob(pattern))
                    if candidates:
                        print(f"Found log: {candidates[0]}")
                        return candidates[0]

            for pattern in patterns:
                candidates = list(search_path.rglob(pattern))
                if candidates:
                    print(f"Found log: {candidates[0]}")
                    return candidates[0]

        time.sleep(1)

    print(f"Could not find log file for job {job_id}")
    sys.exit(1)


def monitor_until_complete(
    log_file: Path,
    job_id: str,
    max_steps: int = None,
    auto_kill_on_critical: bool = False,
    check_interval: int = 5,
) -> dict:
    monitor = HealthMonitor(window_size=10, check_interval=check_interval)

    print("\n" + "=" * 80)
    print("Starting health monitoring...")
    print("=" * 80 + "\n")

    last_health_check = None
    job_running = True

    training_started = False

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-500:]:
                metrics = monitor.parse_line(line)
                if metrics:
                    health_result = monitor.update(metrics)
                    if health_result:
                        last_health_check = health_result
                        print_health_summary(health_result, monitor.step)

                        if auto_kill_on_critical and health_result["status"] == "CRITICAL":
                            print(f"\nCRITICAL FAILURE - Killing job {job_id}")
                            subprocess.run(["scancel", job_id])
                            return finalize_results(monitor, last_health_check, "FAILED")

            if monitor.step > 0:
                training_started = True
                print(f"Caught up to step {monitor.step}\n")

            print("Monitoring new log lines...\n")

            process = subprocess.Popen(
                ["tail", "-f", "-n", "0", str(log_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            while job_running:
                line = process.stdout.readline()
                if not line:
                    status = subprocess.run(
                        ["squeue", "-j", job_id, "-h"],
                        capture_output=True,
                        text=True
                    )
                    if not status.stdout.strip():
                        print("\nJob completed")
                        job_running = False
                        break
                    time.sleep(0.1)
                    continue

                metrics = monitor.parse_line(line)
                if metrics:
                    health_result = monitor.update(metrics)
                    if health_result:
                        last_health_check = health_result
                        print_health_summary(health_result, monitor.step)

                        if auto_kill_on_critical and health_result["status"] == "CRITICAL":
                            print(f"\nCRITICAL FAILURE - Killing job {job_id}")
                            subprocess.run(["scancel", job_id])
                            process.kill()
                            return finalize_results(monitor, last_health_check, "FAILED")

                        if max_steps and monitor.step >= max_steps:
                            print(f"\nReached target step {max_steps}")
                            process.kill()
                            return finalize_results(monitor, last_health_check, "PASSED")

            process.kill()

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        return finalize_results(monitor, last_health_check, "INTERRUPTED")

    return finalize_results(monitor, last_health_check, "COMPLETED")


def print_health_summary(health_result: dict, step: int):
    """Print a compact health summary."""
    status = health_result["status"]

    if status == "CRITICAL":
        color = "\033[91m\033[1m" 
    else:
        color = "\033[92m\033[1m"

    reset = "\033[0m"

    print(f"Step {step}: {color}{status}{reset}")

    for msg in health_result.get("criticals", []):
        print(f"  {msg}")
    for msg in health_result.get("warnings", []):
        print(f"  {msg}")
    for msg in health_result.get("checks", []):
        print(f"  {msg}")
    print()


def finalize_results(monitor: HealthMonitor, last_health_check: dict, outcome: str) -> dict:
    return {
        "outcome": outcome,
        "final_step": monitor.step,
        "final_status": last_health_check["status"] if last_health_check else "UNKNOWN",
        "critical_count": monitor.critical_count,
        "warning_count": monitor.warning_count,
        "final_reward": list(monitor.rewards)[-1] if monitor.rewards else None,
        "final_loss": list(monitor.losses)[-1] if monitor.losses else None,
    }


def write_report(results: dict, output_file: Path, job_id: str, slurm_script: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
GRPO Training Validation Report
{'=' * 80}
Timestamp: {timestamp}
Job ID: {job_id}
Script: {slurm_script}

RESULTS
-------
Outcome: {results['outcome']}
Final Status: {results['final_status']}
Final Step: {results['final_step']}

METRICS
-------
Final Loss: {results['final_loss']:.4f if results['final_loss'] is not None else 'N/A'}
Final Reward: {results['final_reward']:.4f if results['final_reward'] is not None else 'N/A'}

ISSUES
------
Critical Issues: {results['critical_count']}
Warnings: {results['warning_count']}

VERDICT
-------
"""

    if results['outcome'] in ['PASSED', 'COMPLETED'] and results['final_status'] == 'HEALTHY':
        report += "PASS - Training completed successfully with healthy metrics\n"
        exit_code = 0
    elif results['outcome'] == 'FAILED' or results['final_status'] == 'CRITICAL':
        report += "FAIL - Training failed or critical issues detected\n"
        exit_code = 1
    else:
        report += "INCOMPLETE - Training interrupted or inconclusive\n"
        exit_code = 2

    report += "=" * 80 + "\n"

    # Write to file
    with open(output_file, 'w') as f:
        f.write(report)

    # Print to console
    print("\n" + report)
    print(f"Report written to: {output_file}")

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Run and validate GRPO training")
    parser.add_argument("slurm_script", help="Path to SLURM script")
    parser.add_argument("--steps", type=int, help="Stop after N steps (for testing)")
    parser.add_argument("--auto-kill-on-critical", action="store_true",
                        help="Automatically kill job on critical failure")
    parser.add_argument("--output", type=Path, default=Path("training_report.txt"),
                        help="Output file for report")
    parser.add_argument("--check-interval", type=int, default=5,
                        help="Health check interval (steps)")

    args = parser.parse_args()

    job_id, log_file = submit_job(args.slurm_script)

    results = monitor_until_complete(
        log_file=log_file,
        job_id=job_id,
        max_steps=args.steps,
        auto_kill_on_critical=args.auto_kill_on_critical,
        check_interval=args.check_interval,
    )

    exit_code = write_report(results, args.output, job_id, args.slurm_script)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
