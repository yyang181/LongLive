#!/usr/bin/env python3
"""
每隔一个小时执行一次 aws s3 sync，将本地 logs 同步到 S3。
"""

import subprocess
import time
import datetime
import sys
import argparse

DEFAULT_LOCAL_PATH = "/local-ssd/code/LongLive/logs"
DEFAULT_S3_PATH = "s3://datamodel-code-us-west-2/yixinyang/code/LongLive/logs"
INTERVAL_SECONDS = 3600  # 1 小时


def log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def sync_once(local_path: str, s3_path: str) -> int:
    cmd = ["aws", "s3", "sync", local_path, s3_path, "--region", "us-west-2"]
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            log("Sync completed successfully.")
        else:
            log(f"Sync failed with return code {result.returncode}.")
        return result.returncode
    except FileNotFoundError:
        log("Error: 'aws' CLI not found. Please install awscli.")
        return 127
    except Exception as e:
        log(f"Sync raised exception: {e}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="每隔一段时间执行 aws s3 sync，将本地目录同步到 S3。")
    parser.add_argument(
        "--local_path",
        type=str,
        default=DEFAULT_LOCAL_PATH,
        help=f"本地源目录路径（默认: {DEFAULT_LOCAL_PATH}）",
    )
    parser.add_argument(
        "--s3_path",
        type=str,
        default=DEFAULT_S3_PATH,
        help=f"S3 目标路径（默认: {DEFAULT_S3_PATH}）",
    )
    args = parser.parse_args()

    log(f"Auto backup started. Interval = {INTERVAL_SECONDS}s")
    log(f"Source: {args.local_path}")
    log(f"Target: {args.s3_path}")
    while True:
        sync_once(args.local_path, args.s3_path)
        log(f"Sleeping {INTERVAL_SECONDS}s until next sync...")
        try:
            time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log("Interrupted by user. Exiting.")
            sys.exit(0)


if __name__ == "__main__":
    main()
