#!/usr/bin/env bash
# Download and extract GIST1M (TEXMEX corpus, Jegou et al.) into data/gist1m/.
# Resumable: re-running continues a partial download (curl -C -).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DL_DIR="$ROOT/data/gist1m_download"
OUT_DIR="$ROOT/data/gist1m"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
EXPECTED_BYTES=2740172684

mkdir -p "$DL_DIR" "$OUT_DIR"
cd "$DL_DIR"

echo "[download] $URL -> $DL_DIR/gist.tar.gz"
curl -C - --retry 5 --retry-delay 10 -o gist.tar.gz "$URL"

actual=$(stat -f%z gist.tar.gz 2>/dev/null || stat -c%s gist.tar.gz)
if [ "$actual" -ne "$EXPECTED_BYTES" ]; then
  echo "[download] size mismatch: got $actual, expected $EXPECTED_BYTES" >&2
  exit 1
fi

echo "[extract] archive contents:"
tar -tzf gist.tar.gz
# Extract only regular .fvecs/.ivecs files, flattening paths, into OUT_DIR.
for f in $(tar -tzf gist.tar.gz | grep -E '\.(fvecs|ivecs)$'); do
  case "$f" in /*|*..*) echo "refusing suspicious path: $f" >&2; exit 1;; esac
  tar -xzOf gist.tar.gz "$f" > "$OUT_DIR/$(basename "$f")"
done
ls -l "$OUT_DIR"
echo "[done]"
