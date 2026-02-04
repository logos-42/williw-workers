"""
本地 demo：直接构造 metadata + nodes，请求 Rust Worker 的 /api/process，打印精简输出。

你可以把 Worker 部署到 Cloudflare 后，把 WORKER_URL 换成线上地址；
本地如果没有 Worker 环境，也可以先看请求/响应格式与 demo 输出文件。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import requests


def main() -> int:
    worker_url = os.environ.get("WORKER_URL", "http://127.0.0.1:8787")

    # demo metadata（假设 12 层）
    metadata: Dict[str, Any] = {
        "layers": [
            {"layer_index": i, "name": f"layer.{i}", "compute_required": float((i + 1) * 1000)}
            for i in range(12)
        ]
    }

    nodes = [
        {"node_id": "node_a", "compute_power": 3.0},
        {"node_id": "node_b", "compute_power": 2.0},
        {"node_id": "node_c", "compute_power": 1.0},
    ]

    payload = {"model_id": "demo-model", "metadata": metadata, "nodes": nodes, "strategy": "compute"}

    # 允许离线：如果没起 worker，就直接落盘 payload，方便你对接接口层
    out_dir = Path("node_client")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "demo_process_request.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        r = requests.post(f"{worker_url}/api/process", json=payload, timeout=10)
        r.raise_for_status()
        resp = r.json()
        (out_dir / "demo_process_response.json").write_text(
            json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(resp, ensure_ascii=False, indent=2))
        print(f"写入: {out_dir/'demo_process_response.json'}")
    except Exception as e:
        print(f"未能请求 worker（这是允许的离线 demo）：{e}")
        print(f"已写入请求样例: {out_dir/'demo_process_request.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

