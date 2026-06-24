#!/usr/bin/env python3
"""
每隔一个小时执行一次 aws s3 sync，将本地 logs 同步到 S3。
"""

import subprocess
import time
import datetime
import sys

LOCAL_PATH = "/local-ssd/code/LongLive/logs"
S3_PATH = "s3://datamodel-code-us-west-2/yixinyang/code/LongLive/logs"
INTERVAL_SECONDS = 3600  # 1 小时


def log(msg: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def sync_once() -> int:
    cmd = ["aws", "s3", "sync", LOCAL_PATH, S3_PATH]
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
    log(f"Auto backup started. Interval = {INTERVAL_SECONDS}s")
    log(f"Source: {LOCAL_PATH}")
    log(f"Target: {S3_PATH}")
    while True:
        sync_once()
        log(f"Sleeping {INTERVAL_SECONDS}s until next sync...")
        try:
            time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log("Interrupted by user. Exiting.")
            sys.exit(0)


if __name__ == "__main__":
    main()
