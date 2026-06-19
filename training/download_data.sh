#!/bin/bash
# Download pretokenized TinyStories data for ANE training
# Format: flat uint16 token IDs (Llama2 BPE, 32K vocab)
# Source: enio/TinyStories on HuggingFace (pretokenized with karpathy/llama2.c)
#
# The tar.gz contains data00.bin..data49.bin (50 shards). We extract two:
#   data00 -> tinystories_data00.bin   (TRAIN shard)
#   data01 -> tinystories_data01.bin   (VAL shard — held out, no train/val leakage)
# This gives the R2/R3 efficiency rungs a proper held-out validation set
# (PRD User Story 3).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_TRAIN="$SCRIPT_DIR/tinystories_data00.bin"
OUT_VAL="$SCRIPT_DIR/tinystories_data01.bin"

report() {
    local f="$1" label="$2"
    local size tokens
    size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
    tokens=$((size / 2))
    echo "  $label: $f ($tokens tokens, $(echo "scale=1; $size/1000000" | bc) MB)"
}

if [ -f "$OUT_TRAIN" ] && [ -f "$OUT_VAL" ]; then
    echo "Both shards already present:"
    report "$OUT_TRAIN" "train"
    report "$OUT_VAL" "val"
    exit 0
fi

TAR_URL="https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok32000/TinyStories_tok32000.tar.gz?download=true"
TAR_FILE="$SCRIPT_DIR/TinyStories_tok32000.tar.gz"

echo "=== TinyStories Data Download ==="
echo "Downloading pretokenized TinyStories (32K vocab, ~993 MB)..."
echo "  Source: enio/TinyStories on HuggingFace"
echo "  Extracting train (data00) + val (data01) shards."
echo ""

# Download the tar.gz
if [ ! -f "$TAR_FILE" ]; then
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$TAR_FILE" "$TAR_URL"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$TAR_FILE" "$TAR_URL"
    else
        echo "Error: need curl or wget"
        exit 1
    fi
else
    echo "Tar file already downloaded, skipping..."
fi

# Verify it's actually a gzip file (not an error page)
if ! file "$TAR_FILE" | grep -q "gzip"; then
    echo "Error: Downloaded file is not a valid gzip archive."
    echo "Content: $(head -c 100 "$TAR_FILE")"
    rm -f "$TAR_FILE"
    exit 1
fi

echo ""
echo "Extracting data00.bin (train) + data01.bin (val) from archive..."

# Extract one shard from the tar into its expected flat location.
extract_shard() {
    local shard="$1" out="$2"
    [ -f "$out" ] && { echo "  $out already exists, skipping."; return; }
    local path
    path=$(tar tzf "$TAR_FILE" 2>/dev/null | grep "$shard"'\.bin' | head -1)
    if [ -z "$path" ]; then
        echo "Error: $shard.bin not found in archive. Contents:"
        tar tzf "$TAR_FILE" | head -20
        exit 1
    fi
    echo "  Found: $path"
    tar xzf "$TAR_FILE" -C "$SCRIPT_DIR" "$path"
    local extracted="$SCRIPT_DIR/$path"
    if [ "$extracted" != "$out" ]; then
        mv "$extracted" "$out"
        rmdir "$(dirname "$extracted")" 2>/dev/null || true
    fi
}

extract_shard "data00" "$OUT_TRAIN"
extract_shard "data01" "$OUT_VAL"

# Clean up tar.gz to save disk space
echo "Cleaning up archive..."
rm -f "$TAR_FILE"

echo ""
echo "Done:"
report "$OUT_TRAIN" "train"
report "$OUT_VAL" "val"

# Sanity check
python3 -c "
import struct
for f in ['$OUT_TRAIN', '$OUT_VAL']:
    with open(f, 'rb') as fh:
        tokens = struct.unpack('<10H', fh.read(20))
        print(f'{f}: first 10 tokens = {tokens}')
" 2>/dev/null || true
