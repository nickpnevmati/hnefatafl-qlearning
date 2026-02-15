#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../hnefatafl"

cargo run --bin hnefatafl-ai -- \
    --host localhost \
    --username attacker \
    --role attacker \
    --ai basic \
    --join-game 0