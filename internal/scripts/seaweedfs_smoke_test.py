from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.storage import SeaweedFSS3Config, SeaweedFSObjectStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test SeaweedFS S3 using boto3.")
    parser.add_argument(
        "--prefix",
        default="smoke",
        help="Object prefix inside the bucket.",
    )
    parser.add_argument(
        "--text",
        default="seaweedfs smoke test",
        help="Text payload to upload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = SeaweedFSS3Config.from_env()
    store = SeaweedFSObjectStore(config)
    store.ensure_bucket()

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = f"{args.prefix.rstrip('/')}/{now}"

    txt_key = f"{run_prefix}/hello.txt"
    meta_key = f"{run_prefix}/meta.json"
    download_path = ROOT / "state" / "seaweedfs_smoke_download.txt"

    store.put_bytes(txt_key, args.text.encode("utf-8"), content_type="text/plain")
    store.put_json(
        meta_key,
        {
            "bucket": config.bucket,
            "endpoint_url": config.endpoint_url,
            "prefix": config.prefix,
            "uploaded_key": txt_key,
            "timestamp_utc": now,
        },
    )

    payload = store.get_bytes(txt_key).decode("utf-8")
    store.download_file(txt_key, download_path)
    objects = store.list_objects(run_prefix)

    print(
        json.dumps(
            {
                "bucket": config.bucket,
                "endpoint_url": config.endpoint_url,
                "base_prefix": config.prefix,
                "run_prefix": run_prefix,
                "payload_roundtrip": payload,
                "download_path": str(download_path),
                "object_count": len(objects),
                "objects": objects,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
