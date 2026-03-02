# Copenhagen Hnefatafl Q-Learning

This repository bundles three things:

- the upstream Copenhagen Hnefatafl server/client/engine sources (`hnefatafl/`, written in Rust),
- a Python implementation of the linear function-approximation Q-learning agents plus tooling (`qlearning/`), and
- the documentation artifacts for the project (report & presentation).

The goal of the code is to train attacker/defender agents that can connect to a running Hnefatafl server over its text protocol, learn from self-play, and later be evaluated or played against.

## Cloning the Repository

`git clone https://github.com/nickpnevmati/hnefatafl-qlearning --recurse-submodules`

## Prerequisites

1. **Rust toolchain** (stable) with `cargo` to build/run the server and optional GUI client. The provided `shell.nix` drops you into a suitable environment on Nix systems.
2. **Python 3.10+** (the code uses type annotations from 3.10) plus `pip`. Only the metrics plotting script depends on external packages:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install numpy matplotlib
   ```
   The training/evaluation scripts only rely on the standard library.

## Running the server and GUI client

1. Build the Rust binaries once:
   ```bash
   cd hnefatafl
   cargo build --release --bin hnefatafl-server-full --bin hnefatafl-client
   ```
2. Start a local text-protocol server:
   ```bash
   cargo run --release --bin hnefatafl-server-full
   ```
   The default port (`49152`) and `VERSION_ID` (`ad746a65`) match what the Python players expect. Leave this process running.
3. (Optional) launch the GUI client in another terminal to observe or play:
   ```bash
   cargo run --release --bin hnefatafl-client -- --host localhost
   ```
   You can also run `./init-server-client.sh` from the repo root to spawn both components; the script will stop the server when you exit the GUI.

## Setup

Before running an agent, you must create an account for it, this can be achieved like so:

`python qlearning/random_player.py --username USERNAME --create-account --account-only`

## Training with the orchestrator

The orchestrator (`qlearning/orchestrator.py`) automates matches between a learning agent and various opponents. It streams learner output, spawns an opponent for every `new game id`, saves models, and records metrics to JSONL files.

1. Ensure the server is running (see previous section).
2. Activate your Python environment.
3. Train one role at a time. Example: defender vs random attackers for 40k games:
   ```bash
   python qlearning/orchestrator.py \
     --role <defender|attacker> \
     --games N \
     --host HOST \
     --port PORT \
     --learner-username q-def \
     --opponent-username q-att \
     --checkpoint-every N \
     --eval-interval N \
     --opponent-policy <random|qlearning|pool>
     --auto-accept
   ```

Key behaviors & files:

- `--role` selects whether the learner plays as attacker or defender. Run the orchestrator twice (once per role) to train both models.
- By default the learner model is stored at `qlearning/qtable_<role>.json`, learning metrics at `qlearning/learning_metrics_<role>.jsonl`, and checkpoints under `qlearning/checkpoints/`.
- `--checkpoint-every` and `--eval-interval` control when checkpoints are persisted and when evaluation batches against random/checkpoint opponents are injected.
- For pool play, provide checkpoints with `--opponent-checkpoint-glob 'qlearning/checkpoints/attacker_g*.json'` and optionally `--opponent-pool-order round-robin`.
- Resume training using `--start-game <N>`; the script automatically decays alpha/epsilon based on that starting index.

The orchestrator logs win/loss counts per chunk to stdout. Each spawned subprocess logs into `qlearning/learning_metrics_<role>.jsonl` so you can resume or analyze learning curves later.

## Playing or evaluating without more training

To run the trained weights without further updates (for example to play against the GUI client), call `qlearning_player.py` directly with `--no-train`:

```bash
python qlearning/qlearning_player.py \
  --host localhost --port 49152 \
  --username ai-def --password '' \
  --role defender --games 1 --auto-accept \
  --qtable-path qlearning/qtable_defender.json \
  --learning-log-path /tmp/ai_def_eval.jsonl \
  --no-train
```

Tips:

- Launch the GUI client (or another account) and challenge `ai-def`. Because the agent uses `--auto-accept`, it will accept the pending challenge it created.
- Swap `--role attacker` and point `--qtable-path` at the attacker weights to test that side.
- For a quick baseline opponent, run `python qlearning/random_player.py --host localhost --username rand-att --role attacker --games 1 --auto-accept` in another terminal; it will move randomly but obey the protocol.

## Plotting logged metrics

Once training finishes you can convert the JSONL logs into PNGs:

```bash
python qlearning/metrics.py \
  --log-file qlearning/learning_metrics_defender.jsonl \
  --out-dir qlearning/metrics/images \
  --prefix defender \
  --rolling-window 200
```

The script (requires `numpy` + `matplotlib`) produces combined plots plus focused TD-error, weight-norm, and win-rate charts for the given log. Run it again with the attacker log to inspect that role.

## Troubleshooting & tips

- Make sure only one learner attempts to log in with a given username at a time. Use `--learner-username` / `--opponent-username` to keep roles distinct.
- If you see `VERSION_ID mismatch` errors, pass `--version-id` to the orchestrator/player scripts so they match the Rust server that is running.
- Delete or move `qlearning/checkpoints/` when starting a fresh experiment; the orchestrator will recreate the folder.
- Use `--verbose-player-output` on the orchestrator to stream every line from the learner, which helps debug protocol issues or illegal moves.
