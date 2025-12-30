#!/bin/bash
# Download standard ANN benchmark datasets for CP-HNSW evaluation
# Datasets: SIFT-1M (128-dim), GIST-1M (960-dim)

set -e

DATASETS_DIR="${1:-$HOME/datasets}"

echo "=== CP-HNSW Dataset Downloader ==="
echo "Target directory: $DATASETS_DIR"
echo ""

# Function to download and extract a dataset
download_dataset() {
    local name=$1
    local url=$2
    local dir="$DATASETS_DIR/$name"

    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        echo "[$name] Already exists, skipping..."
        return 0
    fi

    echo "[$name] Downloading from $url..."
    mkdir -p "$dir"
    cd "$dir"

    wget --progress=bar:force:noscroll "$url" -O "${name}.tar.gz"

    echo "[$name] Extracting..."
    tar -xzf "${name}.tar.gz"
    rm "${name}.tar.gz"

    echo "[$name] Done!"
    cd - > /dev/null
}

# Download SIFT-1M (128-dim, 1M base, 10K queries)
# File sizes: base ~500MB, queries ~5MB, ground truth ~4MB
download_dataset "sift1m" "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

# Download GIST-1M (960-dim, 1M base, 1K queries)
# File sizes: base ~3.6GB, queries ~3.6MB, ground truth ~4MB
# NOTE: GIST vectors are NOT normalized - must normalize after loading!
download_dataset "gist1m" "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"

echo ""
echo "=== Download Summary ==="
echo ""

# Verify SIFT-1M
if [ -f "$DATASETS_DIR/sift1m/sift/sift_base.fvecs" ]; then
    echo "[SIFT-1M] OK"
    ls -lh "$DATASETS_DIR/sift1m/sift/"*.fvecs "$DATASETS_DIR/sift1m/sift/"*.ivecs 2>/dev/null || true
else
    echo "[SIFT-1M] MISSING or INCOMPLETE"
fi

echo ""

# Verify GIST-1M
if [ -f "$DATASETS_DIR/gist1m/gist/gist_base.fvecs" ]; then
    echo "[GIST-1M] OK"
    echo "WARNING: GIST vectors are unnormalized - must call normalize_vectors() after loading!"
    ls -lh "$DATASETS_DIR/gist1m/gist/"*.fvecs "$DATASETS_DIR/gist1m/gist/"*.ivecs 2>/dev/null || true
else
    echo "[GIST-1M] MISSING or INCOMPLETE"
fi

echo ""
echo "=== Usage ==="
echo "SIFT-1M path: $DATASETS_DIR/sift1m/sift"
echo "GIST-1M path: $DATASETS_DIR/gist1m/gist"
echo ""
echo "Example:"
echo "  ./eval_master --sift $DATASETS_DIR/sift1m/sift --gist $DATASETS_DIR/gist1m/gist"
