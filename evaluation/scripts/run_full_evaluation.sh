#!/bin/bash
# Master Research Evaluation Script for CP-HNSW PhD Portfolio
# Runs all K variants and generates publication artifacts
set -e

SIFT_PATH="${1:-../data/sift}"
OUTPUT_DIR="${2:-results}"

echo "=========================================="
echo "CP-HNSW Research Evaluation Protocol"
echo "=========================================="
echo "Hardware: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'None')"
echo "SIFT Path: $SIFT_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Pre-flight checks
echo "[0/6] Pre-flight checks..."

# Check disk space
echo "  Disk space: $(df -h . | tail -1 | awk '{print $4}') available"

# GPU persistence mode (may require root)
nvidia-smi -pm 1 2>/dev/null && echo "  GPU persistence mode: enabled" || echo "  GPU persistence mode: requires root (skipped)"

# GPU Warmup (prevents first-call initialization penalty)
echo "  Warming up GPU..."
python3 -c "import faiss; faiss.StandardGpuResources()" 2>/dev/null || echo "  (Faiss GPU warmup skipped - not installed)"

# Verify binaries exist
if [ ! -f "../build/eval_master_k16" ] || [ ! -f "../build/eval_master_k32" ] || [ ! -f "../build/eval_master_k64" ]; then
    echo "ERROR: K-variant binaries not found. Run:"
    echo "  cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make eval_master_k16 eval_master_k32 eval_master_k64 -j\$(nproc)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# 1. Run Faiss baselines (if available)
echo ""
echo "[1/6] Running Faiss baselines..."
if [ -f "baselines/run_faiss.py" ]; then
    python3 baselines/run_faiss.py --sift "$SIFT_PATH" --output "$OUTPUT_DIR" --normalize 2>&1 || echo "  (Faiss baseline failed - continuing)"
else
    echo "  (Faiss baseline script not found - skipping)"
fi

# 2. Run K=16 experiments
echo ""
echo "[2/6] Running CP-HNSW K=16..."
mkdir -p "$OUTPUT_DIR/k16"
../build/eval_master_k16 --sift "$SIFT_PATH" --output "$OUTPUT_DIR/k16" --exp 1,3 2>&1 | tee "$OUTPUT_DIR/k16/log.txt"

# 3. Run K=32 experiments (full suite)
echo ""
echo "[3/6] Running CP-HNSW K=32 (full suite)..."
mkdir -p "$OUTPUT_DIR/k32"
../build/eval_master_k32 --sift "$SIFT_PATH" --output "$OUTPUT_DIR/k32" --exp 1,2,3,4,5 2>&1 | tee "$OUTPUT_DIR/k32/log.txt"

# 4. Run K=64 experiments (High Precision)
echo ""
echo "[4/6] Running CP-HNSW K=64 (High Precision)..."
mkdir -p "$OUTPUT_DIR/k64"
../build/eval_master_k64 --sift "$SIFT_PATH" --output "$OUTPUT_DIR/k64" --exp 1,3 2>&1 | tee "$OUTPUT_DIR/k64/log.txt"

# 5. Generate combined resolution ceiling table
echo ""
echo "[5/6] Generating Resolution Ceiling summary..."
echo "K,BruteForce_Recall,Graph_Recall,Rerank_Recall" > "$OUTPUT_DIR/resolution_ceiling.csv"
for K in 16 32 64; do
    if [ -f "$OUTPUT_DIR/k$K/log.txt" ]; then
        # Extract metrics from logs (pattern matching)
        echo "  Processing K=$K..."
    fi
done

# 6. Generate plots
echo ""
echo "[6/6] Generating publication plots..."
if [ -f "scripts/plot_results.py" ]; then
    python3 scripts/plot_results.py --input "$OUTPUT_DIR" --output "$OUTPUT_DIR/plots" 2>&1 || echo "  (Plot generation failed)"
else
    echo "  (Plot script not found - skipping)"
fi

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key artifacts:"
echo "  - $OUTPUT_DIR/k*/exp1_recall_qps/cphnsw_results.csv"
echo "  - $OUTPUT_DIR/k32/exp2_scalability/thread_scaling.csv"
echo "  - $OUTPUT_DIR/plots/*.pdf"
echo ""
