#!/usr/bin/env bash
set -euo pipefail

# Simple end-to-end smoke run using Ollama + quick configs.
# - Installs Ollama if missing
# - Uses custom Ollama data dir at /local/yada/ollama
# - Starts the Ollama server
# - Pulls a small model (default: llama3.2:1b)
# - Runs a minimal RATTPO search (1 round, 1 sample) on a custom prompt
# - Prints where artifacts are saved

HERE_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$(cd "$HERE_DIR/.." && pwd)"

# Tunables (override via env)
: "${OLLAMA_HOME:=/local/yada/ollama}"
: "${OLLAMA_HOST:=127.0.0.1}"
: "${OLLAMA_PORT:=11434}"
export OLLAMA_URL="${OLLAMA_HOST}:${OLLAMA_PORT}"

: "${MODEL:=llama3.2:1b}"
: "${PROMPT:=A cozy cabin in the snow at night}"
: "${EXP_NAME:=ollama_quick_$(date +%Y%m%d_%H%M%S)}"
: "${CONDA_ENV:=rattpo}"
: "${KEEP_OLLAMA:=0}"

echo "[run_ollama_quick] Using OLLAMA_HOME=$OLLAMA_HOME"
mkdir -p "$OLLAMA_HOME"

# Ensure conda env exists
if ! conda env list | awk '{print $1}' | grep -q "^${CONDA_ENV}$"; then
  echo "[run_ollama_quick] Conda env '${CONDA_ENV}' not found. Aborting." >&2
  exit 1
fi

# Install Ollama if not present
if ! command -v ollama >/dev/null 2>&1; then
  echo "[run_ollama_quick] Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama server
echo "[run_ollama_quick] Starting Ollama server on ${OLLAMA_URL}..."
export OLLAMA_HOME
export OLLAMA_HOST
nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
OLLAMA_PID=$!

# Wait for readiness
echo -n "[run_ollama_quick] Waiting for Ollama to become ready"
for i in {1..60}; do
  if curl -sf "http://${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    echo " - ready"
    break
  fi
  echo -n "."
  sleep 1
  if ! kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
    echo "\n[run_ollama_quick] Ollama server died unexpectedly; see /tmp/ollama_serve.log" >&2
    exit 1
  fi
  if [[ $i -eq 60 ]]; then
    echo "\n[run_ollama_quick] Timed out waiting for Ollama" >&2
    exit 1
  fi
done

# Pull a small model for local LLM
echo "[run_ollama_quick] Pulling model: ${MODEL} (this may take a while)"
OLLAMA_HOME="$OLLAMA_HOME" ollama pull "$MODEL"

# Run a tiny RATTPO job (1 round, 1 sample); use quick configs
echo "[run_ollama_quick] Running quick RATTPO job..."
pushd "$REPO_ROOT" >/dev/null
conda run -n "$CONDA_ENV" python main.py \
  --config \
    configs/diffusion_configs/sd1.4_quick.json \
    configs/search_configs/ollama_quick.json \
    configs/verifier_configs/aesthetic_quick.json \
  --dataset custom \
  --prompt "$PROMPT" \
  --seed 42 \
  --exp_name "$EXP_NAME"
popd >/dev/null

# Compute and print output path
HOST_NAME=$(hostname)
OUTPUT_DIR="$REPO_ROOT/output/${HOST_NAME}/custom/sd-v1.4/aesthetic/${EXP_NAME}"
echo "[run_ollama_quick] Artifacts should be in: $OUTPUT_DIR"
if [[ -d "$OUTPUT_DIR" ]]; then
  echo "[run_ollama_quick] Listing round_1 outputs:"
  ls -la "$OUTPUT_DIR"/round_1 || true
fi

# Stop Ollama unless asked to keep running
if [[ "$KEEP_OLLAMA" != "1" ]]; then
  echo "[run_ollama_quick] Stopping Ollama (PID $OLLAMA_PID)"
  kill "$OLLAMA_PID" >/dev/null 2>&1 || true
fi

echo "[run_ollama_quick] Done."
