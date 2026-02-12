#!/bin/bash
# Test script for Williw Rust Worker API using curl

BASE_URL="https://williw-worker-rust.yuanjieliu65.workers.dev"

echo "🚀 Williw Rust Worker API Test Suite"
echo "📍 Base URL: $BASE_URL"
echo "======================================"

# Test 1: /api/request
echo ""
echo "🧪 Test 1: /api/request"
response1=$(curl -s -X POST "$BASE_URL/api/request" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "test"}' \
  -w "\nHTTP Status: %{http_code}\n")
echo "$response1"

# Test 2: /api/process (compute strategy)
echo ""
echo "🧪 Test 2: /api/process (compute strategy)"
response2=$(curl -s -X POST "$BASE_URL/api/process" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "bert-base-uncased",
    "metadata": {
      "layers": [
        {"layer_index": 0, "name": "layer_0", "compute_required": 100.0, "num_params": 1000000},
        {"layer_index": 1, "name": "layer_1", "compute_required": 150.0, "num_params": 1500000},
        {"layer_index": 2, "name": "layer_2", "compute_required": 120.0, "num_params": 1200000}
      ]
    },
    "nodes": [
      {"node_id": "node_1", "compute_power": 200.0},
      {"node_id": "node_2", "compute_power": 170.0}
    ],
    "strategy": "compute"
  }' \
  -w "\nHTTP Status: %{http_code}\n")
echo "$response2"

# Test 3: /api/process (equal strategy)
echo ""
echo "🧪 Test 3: /api/process (equal strategy)"
response3=$(curl -s -X POST "$BASE_URL/api/process" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test-model",
    "metadata": {
      "layers": [
        {"layer_index": 0, "name": "layer_0", "compute_required": 100.0, "num_params": 1000000},
        {"layer_index": 1, "name": "layer_1", "compute_required": 150.0, "num_params": 1500000},
        {"layer_index": 2, "name": "layer_2", "compute_required": 120.0, "num_params": 1200000},
        {"layer_index": 3, "name": "layer_3", "compute_required": 130.0, "num_params": 1300000}
      ]
    },
    "nodes": [
      {"node_id": "node_1", "compute_power": 200.0},
      {"node_id": "node_2", "compute_power": 170.0}
    ],
    "strategy": "equal"
  }' \
  -w "\nHTTP Status: %{http_code}\n")
echo "$response3"

echo ""
echo "======================================"
echo "✅ All tests completed!"
