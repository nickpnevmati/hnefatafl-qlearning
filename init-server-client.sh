#!/usr/bin/env bash

# Runs server & GUI client

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/hnefatafl"

cargo run --release --bin hnefatafl-server-full &
server_pid=$!

cleanup() {
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Give the server a moment to start listening before launching the client.
sleep 2
cargo run --release --bin hnefatafl-client -- --host localhost
