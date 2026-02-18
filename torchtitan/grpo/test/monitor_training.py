#!/usr/bin/env python3
"""
Real-time GRPO training health monitor.

Usage:
    # Monitor via SLURM job ID
    python monitor_training.py --job-id 12345

    # Monitor via log file
    python monitor_training.py --log-file /path/to/slurm-12345.out

    # Monitor via wandb
    python monitor_training.py --wandb-run username/project/run_id

    # With auto-kill on critical failure
    python monitor_training.py --job-id 12345 --auto-kill

Example:
    # Run in background while training
    python monitor_training.py --job-id $(sbatch train.slurm | awk '{print $4}') &
"""

import argparse
import re
import subprocess
import time
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Dict


class HealthMonitor:
    def __init__(self, window_size=10, check_interval=5):
        self.window_size = window_size
        self.check_interval = check_interval

        self.rewards = deque(maxlen=window_size)
        self.ratios = deque(maxlen=window_size)
        self.pos_logp = deque(maxlen=window_size)
        self.neg_logp = deque(maxlen=window_size)
        self.kl_div = deque(maxlen=window_size)
        self.clip_ratio = deque(maxlen=window_size)

        self.step = 0
        self.warning_count = 0
        self.critical_count = 0
        self.is_healthy = True

        self.patterns = {
            "step": re.compile(r"step[:\s]+(\d+)", re.IGNORECASE),
            "reward": re.compile(r"global_advantages[:\s]+([+-]?\d+\.?\d*)", re.IGNORECASE),
            "ratio": re.compile(r"global_ratio[:\s]+([+-]?\d+\.?\d*)", re.IGNORECASE),
            "pos_logp": re.compile(r"global_pos_logp[:\s]+([+-]?\d+\.?\d*)", re.IGNORECASE),
            "neg_logp": re.compile(r"global_neg_logp[:\s]+([+-]?\d+\.?\d*)", re.IGNORECASE),
            "kl": re.compile(r"kl_div_est[:\s]+([+-]?\d+\.?\d*)", re.IGNORECASE),
            "clip": re.compile(r"global_clip_ratio[:\s]+([+-]?\d+\.?\d*)", re.IGNORECASE),
        }

    def parse_line(self, line: str) -> Dict[str, float]:
        metrics = {}
        for key, pattern in self.patterns.items():
            match = pattern.search(line)
            if match:
                try:
                    metrics[key] = float(match.group(1))
                except (ValueError, IndexError):
                    pass
        return metrics

    def update(self, metrics: Dict[str, float]):
        if "step" in metrics:
            self.step = int(metrics["step"])

        if "reward" in metrics:
            self.rewards.append(metrics["reward"])
        if "ratio" in metrics:
            self.ratios.append(metrics["ratio"])
        if "pos_logp" in metrics:
            self.pos_logp.append(metrics["pos_logp"])
        if "neg_logp" in metrics:
            self.neg_logp.append(metrics["neg_logp"])
        if "kl" in metrics:
            self.kl_div.append(metrics["kl"])
        if "clip" in metrics:
            self.clip_ratio.append(metrics["clip"])

        if len(self.rewards) >= 5 and len(self.rewards) % self.check_interval == 0:
            return self.check_health()
        return None

    def check_health(self) -> Dict:
        warnings = []
        criticals = []
        checks = []

        # 1. Reward checks
        if len(self.rewards) >= 5:
            recent_rewards = list(self.rewards)[-5:]
            mean_reward = sum(recent_rewards) / len(recent_rewards)

            if mean_reward < -1.0:
                criticals.append(f"CRITICAL: Reward collapse! Mean={mean_reward:.4f}")
                self.critical_count += 1

            if len(self.rewards) >= self.window_size:
                first_half = list(self.rewards)[:self.window_size // 2]
                second_half = list(self.rewards)[self.window_size // 2:]
                improvement = (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))

                if improvement < -0.05:
                    warnings.append(f"Rewards declining by {improvement:.4f}")
                    self.warning_count += 1
                elif improvement > 0.02:
                    checks.append(f"Rewards improving (+{improvement:.4f})")

        # 2. Policy ratio checks
        if len(self.ratios) >= 3:
            recent_ratios = list(self.ratios)[-3:]
            max_ratio = max(recent_ratios)
            min_ratio = min(recent_ratios)

            if max_ratio > 3.0 or min_ratio < 0.3:
                criticals.append(f"CRITICAL: Policy unstable! Range=[{min_ratio:.3f}, {max_ratio:.3f}]")
                self.critical_count += 1
            elif max_ratio > 2.0 or min_ratio < 0.5:
                warnings.append(f"Policy ratio outside normal range: [{min_ratio:.3f}, {max_ratio:.3f}]")
                self.warning_count += 1
            elif 0.8 <= min_ratio <= 1.2 and 0.8 <= max_ratio <= 1.2:
                checks.append(f"Policy ratio stable (~1.0)")

        # 3. KL divergence checks
        if len(self.kl_div) >= 3 and any(kl > 0 for kl in self.kl_div):
            recent_kl = [kl for kl in list(self.kl_div)[-3:] if kl > 0]
            if recent_kl:
                mean_kl = sum(recent_kl) / len(recent_kl)

                if mean_kl > 1.0:
                    criticals.append(f"CRITICAL: KL explosion! KL={mean_kl:.4f}")
                    self.critical_count += 1
                elif mean_kl > 0.5:
                    warnings.append(f"High KL divergence: {mean_kl:.4f}")
                elif mean_kl < 0.1:
                    checks.append(f"KL divergence healthy: {mean_kl:.4f}")

        # 4. Logp separation checks
        if len(self.pos_logp) >= 3 and len(self.neg_logp) >= 3:
            recent_pos = list(self.pos_logp)[-3:]
            recent_neg = list(self.neg_logp)[-3:]
            sep = (sum(recent_pos) / len(recent_pos)) - (sum(recent_neg) / len(recent_neg))

            if sep < -0.5:
                criticals.append(f"CRITICAL: Model prefers wrong answers! Separation={sep:.4f}")
                self.critical_count += 1
            elif sep < 0.2:
                warnings.append(f"Weak pos/neg separation: {sep:.4f}")
            elif sep > 0.5:
                checks.append(f"Good answer discrimination: Δ={sep:.4f}")

        # 5. Clipping checks
        if len(self.clip_ratio) >= 3 and any(c > 0 for c in self.clip_ratio):
            recent_clip = [c for c in list(self.clip_ratio)[-3:] if c > 0]
            if recent_clip:
                mean_clip = sum(recent_clip) / len(recent_clip)

                if mean_clip > 0.8:
                    warnings.append(f"Heavy clipping: {mean_clip:.1%}")
                elif mean_clip > 0.5:
                    checks.append(f"Moderate clipping: {mean_clip:.1%}")

        # Determine status
        if criticals:
            status = "CRITICAL"
            self.is_healthy = False
        elif warnings:
            status = "WARNING"
        else:
            status = "HEALTHY"
            self.is_healthy = True

        return {
            "status": status,
            "checks": checks,
            "warnings": warnings,
            "criticals": criticals,
        }

    def print_status(self, health_result: Dict):
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        BOLD = '\033[1m'

        print("\n" + "=" * 80)
        print(f"GRPO Health Check - Step {self.step}")
        print("=" * 80)

        for msg in health_result.get("criticals", []):
            print(msg)
        for msg in health_result.get("warnings", []):
            print(msg)
        for msg in health_result.get("checks", []):
            print(msg)

        # Summary
        if self.rewards:
            print(f"\nLatest metrics:")
            print(f"   Reward: {list(self.rewards)[-1]:.4f}")
            if self.ratios:
                print(f"   Ratio:  {list(self.ratios)[-1]:.4f}")
            if self.pos_logp and self.neg_logp:
                sep = list(self.pos_logp)[-1] - list(self.neg_logp)[-1]
                print(f"   Pos/Neg Separation: {sep:.4f}")

        # Colored status
        status = health_result['status']
        if status == "HEALTHY":
            status_colored = f"{GREEN}{BOLD}{status}{RESET}"
        elif status == "WARNING":
            status_colored = f"{YELLOW}{BOLD}{status}{RESET}"
        elif status == "CRITICAL":
            status_colored = f"{RED}{BOLD}{status}{RESET}"
        else:
            status_colored = status

        print(f"\nStatus: {status_colored}")
        print(f"   Warnings: {self.warning_count} | Critical: {self.critical_count}")
        print("=" * 80 + "\n")


def tail_log_file(log_file: Path, monitor: HealthMonitor, auto_kill_job: Optional[str] = None):
    print(f"Monitoring log file: {log_file}")

    try:
        process = subprocess.Popen(
            ["tail", "-f", "-n", "0", str(log_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        while True:
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue

            metrics = monitor.parse_line(line)
            if metrics:
                health_result = monitor.update(metrics)
                if health_result:
                    monitor.print_status(health_result)

                    if auto_kill_job and health_result["status"] == "CRITICAL":
                        print(f"\nCRITICAL FAILURE DETECTED - Killing job {auto_kill_job}")
                        subprocess.run(["scancel", auto_kill_job])
                        print(f"Job {auto_kill_job} killed")
                        break

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        process.kill()
    except Exception as e:
        print(f"\nError: {e}")
        process.kill()


def monitor_job(job_id: str, monitor: HealthMonitor, auto_kill: bool = False):
    print(f"🔍 Looking for log file for job {job_id}...")

    log_patterns = [
        f"{job_id}.out",
        f"slurm-{job_id}.out",
        f"slurm_{job_id}.out",
    ]

    log_file = None
    for pattern in log_patterns:
        candidates = list(Path.cwd().rglob(pattern))
        if candidates:
            log_file = candidates[0]
            break

    if not log_file:
        print(f"Log file not found. Waiting for job to start...")
        for _ in range(60):
            time.sleep(1)
            for pattern in log_patterns:
                candidates = list(Path.cwd().rglob(pattern))
                if candidates:
                    log_file = candidates[0]
                    break
            if log_file:
                break

    if not log_file:
        print(f"Could not find log file for job {job_id}")
        sys.exit(1)

    print(f"Found log file: {log_file}")
    tail_log_file(log_file, monitor, job_id if auto_kill else None)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time GRPO training health monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--job-id", type=str, help="SLURM job ID to monitor")
    parser.add_argument("--log-file", type=Path, help="Log file path to monitor")
    parser.add_argument("--window-size", type=int, default=10, help="Rolling window size for metrics")
    parser.add_argument("--check-interval", type=int, default=5, help="Check health every N updates")
    parser.add_argument("--auto-kill", action="store_true", help="Automatically kill job on critical failure")

    args = parser.parse_args()

    if not args.job_id and not args.log_file:
        parser.error("Must specify either --job-id or --log-file")

    monitor = HealthMonitor(window_size=args.window_size, check_interval=args.check_interval)

    print("GRPO Training Health Monitor Starting...")
    print("=" * 80)

    if args.log_file:
        tail_log_file(args.log_file, monitor)
    else:
        monitor_job(args.job_id, monitor, auto_kill=args.auto_kill)


if __name__ == "__main__":
    main()
