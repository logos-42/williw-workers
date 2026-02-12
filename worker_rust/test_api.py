#!/usr/bin/env python3
"""Test script for Williw Rust Worker API"""

import requests
import json
import sys

BASE_URL = "https://williw-worker-rust.yuanjieliu65.workers.dev"


def test_request_endpoint():
    """Test /api/request endpoint"""
    print("🧪 Testing /api/request endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/request",
            json={"model_id": "test"},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        print(f"✅ Status: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_process_endpoint():
    """Test /api/process endpoint"""
    print("\n🧪 Testing /api/process endpoint...")
    try:
        payload = {
            "model_id": "bert-base-uncased",
            "metadata": {
                "layers": [
                    {
                        "layer_index": 0,
                        "name": "layer_0",
                        "compute_required": 100.0,
                        "num_params": 1000000,
                    },
                    {
                        "layer_index": 1,
                        "name": "layer_1",
                        "compute_required": 150.0,
                        "num_params": 1500000,
                    },
                    {
                        "layer_index": 2,
                        "name": "layer_2",
                        "compute_required": 120.0,
                        "num_params": 1200000,
                    },
                ]
            },
            "nodes": [
                {"node_id": "node_1", "compute_power": 200.0},
                {"node_id": "node_2", "compute_power": 170.0},
            ],
            "strategy": "compute",
        }

        response = requests.post(
            f"{BASE_URL}/api/process",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        print(f"✅ Status: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_equal_strategy():
    """Test equal split strategy"""
    print("\n🧪 Testing /api/process with equal strategy...")
    try:
        payload = {
            "model_id": "test-model",
            "metadata": {
                "layers": [
                    {
                        "layer_index": 0,
                        "name": "layer_0",
                        "compute_required": 100.0,
                        "num_params": 1000000,
                    },
                    {
                        "layer_index": 1,
                        "name": "layer_1",
                        "compute_required": 150.0,
                        "num_params": 1500000,
                    },
                    {
                        "layer_index": 2,
                        "name": "layer_2",
                        "compute_required": 120.0,
                        "num_params": 1200000,
                    },
                    {
                        "layer_index": 3,
                        "name": "layer_3",
                        "compute_required": 130.0,
                        "num_params": 1300000,
                    },
                ]
            },
            "nodes": [
                {"node_id": "node_1", "compute_power": 200.0},
                {"node_id": "node_2", "compute_power": 170.0},
            ],
            "strategy": "equal",
        }

        response = requests.post(
            f"{BASE_URL}/api/process",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        print(f"✅ Status: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Williw Rust Worker API Test Suite")
    print(f"📍 Base URL: {BASE_URL}")
    print("=" * 50)

    results = []
    results.append(("/api/request", test_request_endpoint()))
    results.append(("/api/process (compute)", test_process_endpoint()))
    results.append(("/api/process (equal)", test_equal_strategy()))

    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    for endpoint, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {endpoint}")

    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
