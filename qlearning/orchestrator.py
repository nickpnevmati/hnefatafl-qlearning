#!/usr/bin/env python3
"""Train/evaluate the linear Q-learning player with configurable opponents."""

from __future__ import annotations

import argparse
import atexit
import glob
import json
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


GAME_ID_RE = re.compile(r"new game id:\s*(\d+)")
ACTIVE_PROCESSES: list[subprocess.Popen[str]] = []


def opposite(role: str) -> str:
    return "defender" if role == "attacker" else "attacker"


def track_process(proc: subprocess.Popen[str]) -> subprocess.Popen[str]:
    ACTIVE_PROCESSES.append(proc)
    return proc


def cleanup() -> None:
    """Terminate tracked subprocesses."""
    for proc in ACTIVE_PROCESSES[:]:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        ACTIVE_PROCESSES.remove(proc)


def signal_handler(signum: int, _frame: object) -> None:
    print(f"[orchestrator] received signal {signum}; cleaning up", file=sys.stderr, flush=True)
    cleanup()
    raise SystemExit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup)


@dataclass
class BatchSummary:
    label: str
    requested_games: int
    recorded_games: int
    wins: int
    losses: int
    draws: int
    other: int


@dataclass
class OpponentSelector:
    policy: str
    fixed_qtable: Optional[Path]
    pool: list[Path]
    pool_order: str
    rng: random.Random
    _pool_index: int = 0

    def choose_qtable(self) -> Path:
        if self.policy == "qlearning":
            if self.fixed_qtable is None:
                raise ValueError("opponent policy qlearning requires --opponent-qtable-path")
            return self.fixed_qtable
        if self.policy == "pool":
            if not self.pool:
                raise ValueError("opponent policy pool requires checkpoint paths")
            if self.pool_order == "round-robin":
                picked = self.pool[self._pool_index % len(self.pool)]
                self._pool_index += 1
                return picked
            return self.rng.choice(self.pool)
        raise ValueError(f"unsupported opponent policy: {self.policy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="server hostname or IP")
    parser.add_argument("--port", type=int, default=49152, help="server port")
    parser.add_argument("--version-id", default="ad746a65", help="VERSION_ID expected by server")
    parser.add_argument("--time-control", default="rated fischer 900000 10 11", help="new_game time control string")
    parser.add_argument("--password", default="", help="password for both learner/opponent accounts")
    parser.add_argument("--python", default=sys.executable, help="python executable to use for subprocesses")

    parser.add_argument("--role", choices=("attacker", "defender"), default="defender", help="learner role")
    parser.add_argument("--games", type=int, default=10000, help="number of games to run in this invocation")
    parser.add_argument(
        "--start-game",
        type=int,
        default=0,
        help="starting global game index for resumed training (affects decay/checkpoint numbering)",
    )
    parser.add_argument("--eval-interval", type=int, default=1000, help="run evaluation every N training games (0 disables)")
    parser.add_argument("--eval-games", type=int, default=200, help="evaluation games per matchup")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="save learner checkpoint every N training games (0 disables)")
    parser.add_argument("--checkpoint-dir", default="qlearning/checkpoints", help="checkpoint output directory")
    parser.add_argument("--join-delay-seconds", type=float, default=0.1, help="delay before spawning joiner after new game id")
    parser.add_argument("--seed", type=int, default=7, help="random seed for pool sampling")
    parser.add_argument("--verbose-player-output", action="store_true", help="stream all learner protocol output")

    parser.add_argument("--learner-username", default=None, help="learner account username")
    parser.add_argument("--learner-qtable-path", default=None, help="learner model path")
    parser.add_argument("--learner-learning-log-path", default=None, help="learner training metrics jsonl path")
    parser.add_argument("--eval-learning-log-path", default=None, help="evaluation metrics jsonl path")

    parser.add_argument("--alpha", type=float, default=0.02, help="initial alpha")
    parser.add_argument("--alpha-min", type=float, default=0.002, help="minimum alpha")
    parser.add_argument("--alpha-decay", type=float, default=0.9997, help="alpha decay per game")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="initial epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="epsilon decay per game")

    parser.add_argument(
        "--opponent-policy",
        choices=("random", "qlearning", "pool"),
        default="random",
        help="training opponent type",
    )
    parser.add_argument("--opponent-username", default=None, help="opponent account username")
    parser.add_argument("--opponent-qtable-path", default=None, help="fixed qlearning opponent weights path")
    parser.add_argument(
        "--opponent-checkpoint-glob",
        action="append",
        default=[],
        help="glob pattern for opponent pool checkpoints (repeatable)",
    )
    parser.add_argument(
        "--opponent-pool-order",
        choices=("random", "round-robin"),
        default="random",
        help="how to pick checkpoints from opponent pool",
    )
    parser.add_argument(
        "--opponent-learning-log-path",
        default="/tmp/hnefatafl_opponent_metrics.jsonl",
        help="metrics path for no-train qlearning opponents",
    )

    parser.add_argument(
        "--eval-opponent-checkpoint",
        action="append",
        default=[],
        help="extra fixed checkpoint(s) to evaluate against (repeatable)",
    )
    parser.add_argument(
        "--eval-opponent-checkpoint-glob",
        action="append",
        default=[],
        help="glob pattern(s) for additional eval checkpoints",
    )
    return parser.parse_args()


def resolve_paths(paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        expanded = [Path(p) for p in sorted(glob.glob(raw))]
        if not expanded:
            candidate = Path(raw)
            if candidate.exists():
                resolved.append(candidate)
            continue
        resolved.extend(expanded)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        abs_path = path.resolve()
        if abs_path in seen:
            continue
        deduped.append(abs_path)
        seen.add(abs_path)
    return deduped


def metric_offset(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size


def read_metrics_since(path: Path, offset: int) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
    return entries


def decayed_value(initial: float, minimum: float, decay: float, games_elapsed: int) -> float:
    value = initial * (decay ** games_elapsed)
    return max(minimum, value)


def extract_game_id(line: str) -> Optional[int]:
    match = GAME_ID_RE.search(line)
    if not match:
        return None
    return int(match.group(1))


def checkpoint_model(src: Path, checkpoint_dir: Path, role: str, games_done: int) -> Optional[Path]:
    if not src.exists():
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dst = checkpoint_dir / f"{role}_g{games_done:06d}.json"
    if src.resolve() == dst.resolve():
        return dst
    shutil.copy2(src, dst)
    return dst


def summarize_metrics(entries: list[dict], role: str) -> tuple[int, int, int, int]:
    wins = 0
    losses = 0
    draws = 0
    other = 0
    opp = opposite(role)
    for entry in entries:
        winner = entry.get("winner")
        if winner == role:
            wins += 1
        elif winner == opp:
            losses += 1
        elif winner == "draw" or winner is None:
            draws += 1
        else:
            other += 1
    return wins, losses, draws, other


def print_summary(summary: BatchSummary) -> None:
    total = summary.recorded_games
    if total == 0:
        print(
            f"[{summary.label}] games=0 requested={summary.requested_games} "
            "wins=0 losses=0 draws=0",
            flush=True,
        )
        return
    win_rate = summary.wins / float(total)
    print(
        f"[{summary.label}] games={total}/{summary.requested_games} wins={summary.wins} "
        f"losses={summary.losses} draws={summary.draws} other={summary.other} "
        f"win_rate={win_rate:.3f}",
        flush=True,
    )


def make_common_player_args(args: argparse.Namespace, username: str, role: str) -> list[str]:
    return [
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--username",
        username,
        "--password",
        args.password,
        "--role",
        role,
        "--version-id",
        args.version_id,
        "--time-control",
        args.time_control,
    ]


def spawn_opponent(
    args: argparse.Namespace,
    selector: OpponentSelector,
    opponent_role: str,
    opponent_username: str,
    game_id: int,
) -> subprocess.Popen[str]:
    base = [args.python]
    common = make_common_player_args(args, opponent_username, opponent_role)
    join_args = ["--join-game", str(game_id), "--games", "1"]

    if selector.policy == "random":
        cmd = base + ["qlearning/random_player.py"] + common + join_args
    else:
        model_path = selector.choose_qtable()
        cmd = (
            base
            + ["qlearning/qlearning_player.py"]
            + common
            + join_args
            + [
                "--qtable-path",
                str(model_path),
                "--learning-log-path",
                args.opponent_learning_log_path,
                "--no-train",
            ]
        )

    proc = track_process(
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    )
    return proc


def run_batch(
    args: argparse.Namespace,
    *,
    label: str,
    role: str,
    games: int,
    train: bool,
    qtable_path: Path,
    learning_log_path: Path,
    alpha: float,
    alpha_min: float,
    alpha_decay: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    opponent_selector: OpponentSelector,
    learner_username: str,
    opponent_username: str,
) -> BatchSummary:
    log_offset = metric_offset(learning_log_path)
    opponent_role = opposite(role)

    learner_cmd = (
        [args.python, "qlearning/qlearning_player.py"]
        + make_common_player_args(args, learner_username, role)
        + [
            "--auto-accept",
            "--games",
            str(games),
            "--qtable-path",
            str(qtable_path),
            "--learning-log-path",
            str(learning_log_path),
            "--alpha",
            str(alpha),
            "--alpha-min",
            str(alpha_min),
            "--alpha-decay",
            str(alpha_decay),
            "--gamma",
            str(gamma),
            "--epsilon",
            str(epsilon),
            "--epsilon-min",
            str(epsilon_min),
            "--epsilon-decay",
            str(epsilon_decay),
        ]
    )
    if not train:
        learner_cmd.append("--no-train")

    learner = track_process(
        subprocess.Popen(
            learner_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    )

    spawned_opponents: list[subprocess.Popen[str]] = []
    assert learner.stdout is not None
    for raw_line in learner.stdout:
        line = raw_line.rstrip()
        if args.verbose_player_output:
            print(f"[{label}] {line}", flush=True)

        game_id = extract_game_id(line)
        if game_id is None:
            continue

        if not args.verbose_player_output:
            print(f"[{label}] new game id {game_id}", flush=True)

        if args.join_delay_seconds > 0:
            time.sleep(args.join_delay_seconds)

        proc = spawn_opponent(
            args=args,
            selector=opponent_selector,
            opponent_role=opponent_role,
            opponent_username=opponent_username,
            game_id=game_id,
        )
        spawned_opponents.append(proc)

    return_code = learner.wait()
    ACTIVE_PROCESSES.remove(learner)

    for proc in spawned_opponents:
        if proc.poll() is None:
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=2)
        if proc in ACTIVE_PROCESSES:
            ACTIVE_PROCESSES.remove(proc)

    if return_code != 0:
        raise RuntimeError(f"{label} learner exited with code {return_code}")

    entries = read_metrics_since(learning_log_path, log_offset)
    wins, losses, draws, other = summarize_metrics(entries, role=role)
    return BatchSummary(
        label=label,
        requested_games=games,
        recorded_games=len(entries),
        wins=wins,
        losses=losses,
        draws=draws,
        other=other,
    )


def normalize_defaults(args: argparse.Namespace) -> None:
    role = args.role
    if args.learner_username is None:
        args.learner_username = f"q-{role}"
    if args.opponent_username is None:
        args.opponent_username = f"q-{opposite(role)}"
    if args.learner_qtable_path is None:
        args.learner_qtable_path = f"qlearning/qtable_{role}.json"
    if args.learner_learning_log_path is None:
        args.learner_learning_log_path = f"qlearning/learning_metrics_{role}.jsonl"
    if args.eval_learning_log_path is None:
        args.eval_learning_log_path = f"qlearning/eval_metrics_{role}.jsonl"


def make_training_selector(args: argparse.Namespace, rng: random.Random) -> OpponentSelector:
    pool = resolve_paths(args.opponent_checkpoint_glob)
    fixed = Path(args.opponent_qtable_path).resolve() if args.opponent_qtable_path else None

    if args.opponent_policy == "qlearning" and fixed is None:
        raise ValueError("--opponent-policy qlearning requires --opponent-qtable-path")
    if args.opponent_policy == "pool" and not pool:
        raise ValueError("--opponent-policy pool requires --opponent-checkpoint-glob")

    return OpponentSelector(
        policy=args.opponent_policy,
        fixed_qtable=fixed,
        pool=pool,
        pool_order=args.opponent_pool_order,
        rng=rng,
    )


def make_random_eval_selector(rng: random.Random) -> OpponentSelector:
    return OpponentSelector(
        policy="random",
        fixed_qtable=None,
        pool=[],
        pool_order="random",
        rng=rng,
    )


def build_eval_checkpoint_list(args: argparse.Namespace) -> list[Path]:
    explicit = [Path(p).resolve() for p in args.eval_opponent_checkpoint]
    from_glob = resolve_paths(args.eval_opponent_checkpoint_glob)
    merged = explicit + from_glob
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in merged:
        if path in seen:
            continue
        deduped.append(path)
        seen.add(path)
    return deduped


def main() -> int:
    args = parse_args()
    normalize_defaults(args)

    role = args.role
    qtable_path = Path(args.learner_qtable_path).resolve()
    learning_log_path = Path(args.learner_learning_log_path).resolve()
    eval_log_path = Path(args.eval_learning_log_path).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()

    qtable_path.parent.mkdir(parents=True, exist_ok=True)
    learning_log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    training_selector = make_training_selector(args, rng)
    eval_checkpoints = build_eval_checkpoint_list(args)

    print(
        f"[orchestrator] role={role} games={args.games} start_game={args.start_game} "
        f"eval_interval={args.eval_interval} "
        f"eval_games={args.eval_games} opponent_policy={args.opponent_policy}",
        flush=True,
    )
    print(
        f"[orchestrator] learner_qtable={qtable_path} learner_log={learning_log_path}",
        flush=True,
    )

    if args.start_game < 0:
        raise ValueError("--start-game must be >= 0")

    games_done = args.start_game
    target_games = args.start_game + args.games

    checkpoint_next: Optional[int] = None
    if args.checkpoint_every > 0:
        checkpoint_next = ((games_done // args.checkpoint_every) + 1) * args.checkpoint_every

    eval_next: Optional[int] = None
    if args.eval_interval > 0:
        eval_next = ((games_done // args.eval_interval) + 1) * args.eval_interval

    while games_done < target_games:
        remaining = target_games - games_done
        chunk = remaining
        if eval_next is not None:
            chunk = min(chunk, eval_next - games_done)
        if checkpoint_next is not None:
            chunk = min(chunk, checkpoint_next - games_done)
        if chunk <= 0:
            chunk = 1

        alpha_now = decayed_value(args.alpha, args.alpha_min, args.alpha_decay, games_done)
        epsilon_now = decayed_value(args.epsilon, args.epsilon_min, args.epsilon_decay, games_done)
        print(
            f"[train] starting chunk games={chunk} at_game={games_done} "
            f"alpha={alpha_now:.6f} epsilon={epsilon_now:.6f}",
            flush=True,
        )

        summary = run_batch(
            args,
            label=f"train-{games_done + 1}-{games_done + chunk}",
            role=role,
            games=chunk,
            train=True,
            qtable_path=qtable_path,
            learning_log_path=learning_log_path,
            alpha=alpha_now,
            alpha_min=args.alpha_min,
            alpha_decay=args.alpha_decay,
            gamma=args.gamma,
            epsilon=epsilon_now,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            opponent_selector=training_selector,
            learner_username=args.learner_username,
            opponent_username=args.opponent_username,
        )
        print_summary(summary)
        games_done += chunk

        if checkpoint_next is not None and games_done >= checkpoint_next:
            saved = checkpoint_model(qtable_path, checkpoint_dir, role, games_done)
            if saved is not None:
                print(f"[checkpoint] saved {saved}", flush=True)
            checkpoint_next += args.checkpoint_every

        if eval_next is not None and games_done >= eval_next and args.eval_games > 0:
            print(f"[eval] running random-opponent eval at game={games_done}", flush=True)
            random_eval = run_batch(
                args,
                label=f"eval-random-{games_done}",
                role=role,
                games=args.eval_games,
                train=False,
                qtable_path=qtable_path,
                learning_log_path=eval_log_path,
                alpha=args.alpha,
                alpha_min=args.alpha_min,
                alpha_decay=args.alpha_decay,
                gamma=args.gamma,
                epsilon=args.epsilon,
                epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay,
                opponent_selector=make_random_eval_selector(rng),
                learner_username=args.learner_username,
                opponent_username=args.opponent_username,
            )
            print_summary(random_eval)

            for index, checkpoint in enumerate(eval_checkpoints, start=1):
                if not checkpoint.exists():
                    print(f"[eval] skipping missing checkpoint {checkpoint}", flush=True)
                    continue
                print(
                    f"[eval] running checkpoint eval {index}/{len(eval_checkpoints)} "
                    f"checkpoint={checkpoint}",
                    flush=True,
                )
                checkpoint_eval = run_batch(
                    args,
                    label=f"eval-ckpt{index}-{games_done}",
                    role=role,
                    games=args.eval_games,
                    train=False,
                    qtable_path=qtable_path,
                    learning_log_path=eval_log_path,
                    alpha=args.alpha,
                    alpha_min=args.alpha_min,
                    alpha_decay=args.alpha_decay,
                    gamma=args.gamma,
                    epsilon=args.epsilon,
                    epsilon_min=args.epsilon_min,
                    epsilon_decay=args.epsilon_decay,
                    opponent_selector=OpponentSelector(
                        policy="qlearning",
                        fixed_qtable=checkpoint,
                        pool=[],
                        pool_order="random",
                        rng=rng,
                    ),
                    learner_username=args.learner_username,
                    opponent_username=args.opponent_username,
                )
                print_summary(checkpoint_eval)

            eval_next += args.eval_interval

    if args.checkpoint_every > 0:
        saved = checkpoint_model(qtable_path, checkpoint_dir, role, games_done)
        if saved is not None:
            print(f"[checkpoint] saved final {saved}", flush=True)

    print("[orchestrator] finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
