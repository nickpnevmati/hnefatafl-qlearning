#!/usr/bin/env python3
"""Approximate Q-learning Hnefatafl player built on top of the shared player core."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    from qlearning.player_core import (
        BOARD_SIZE,
        BasePlayer,
        Board,
        PlayerConfig,
        build_player_config,
        opposite,
        parse_common_args,
    )
except ModuleNotFoundError:
    from player_core import (
        BOARD_SIZE,
        BasePlayer,
        Board,
        PlayerConfig,
        build_player_config,
        opposite,
        parse_common_args,
    )

Action = Tuple[str, str]

ATTACKER_START_COUNT = 24
DEFENDER_START_COUNT = 13  # defenders + king
MOBILITY_NORM = 120.0
MAX_MANHATTAN = float((BOARD_SIZE - 1) * 2)
FEATURE_COUNT = 11
CORNERS = (
    (0, 0),
    (0, BOARD_SIZE - 1),
    (BOARD_SIZE - 1, 0),
    (BOARD_SIZE - 1, BOARD_SIZE - 1),
)
SHAPING_BETA = 0.25
SHAPING_CLIP = 0.05
POTENTIAL_MATERIAL_WEIGHT = 0.45
POTENTIAL_MOBILITY_WEIGHT = 0.35
POTENTIAL_KING_WEIGHT = 0.20


@dataclass
class PieceSnapshot:
    own_pieces: int
    opp_pieces: int
    own_mobility: int
    opp_mobility: int
    king_progress: float


class QLearningPlayer(BasePlayer):
    def __init__(
        self,
        config: PlayerConfig,
        qtable_path: Path,
        learning_log_path: Path,
        alpha: float,
        alpha_min: float,
        alpha_decay: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        train: bool,
    ) -> None:
        super().__init__(config)
        self.qtable_path = qtable_path
        self.learning_log_path = learning_log_path
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.train = train

        self.max_own_pieces = (
            ATTACKER_START_COUNT if self.config.role == "attacker" else DEFENDER_START_COUNT
        )
        self.max_opp_pieces = (
            DEFENDER_START_COUNT if self.config.role == "attacker" else ATTACKER_START_COUNT
        )

        self.weights: List[float] = self._load_weights()

        self._pending_action: Optional[Action] = None
        self._pending_features: Optional[List[float]] = None
        self._pending_snapshot: Optional[PieceSnapshot] = None
        self._updates_in_game = 0
        self._sum_abs_td_error = 0.0
        self._sum_abs_delta_w = 0.0
        self._sum_shaped_reward = 0.0
        self._sum_terminal_reward = 0.0

    def _load_weights(self) -> List[float]:
        if not self.qtable_path.exists():
            return [0.0] * FEATURE_COUNT

        try:
            raw = json.loads(self.qtable_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] failed to load q-model from {self.qtable_path}: {exc}", flush=True)
            return [0.0] * FEATURE_COUNT

        if isinstance(raw, dict):
            raw_weights = raw.get("weights", [])
        elif isinstance(raw, list):
            raw_weights = raw
        else:
            raw_weights = []

        if not isinstance(raw_weights, list):
            raw_weights = []

        weights = [0.0] * FEATURE_COUNT
        for idx in range(min(FEATURE_COUNT, len(raw_weights))):
            try:
                weights[idx] = float(raw_weights[idx])
            except (TypeError, ValueError):
                weights[idx] = 0.0

        return weights

    def save_q_table(self) -> None:
        payload = {
            "format": "linear_q_v1",
            "feature_count": FEATURE_COUNT,
            "role": self.config.role,
            "weights": self.weights,
        }

        try:
            self.qtable_path.parent.mkdir(parents=True, exist_ok=True)
            self.qtable_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[warn] failed to save q-model to {self.qtable_path}: {exc}", flush=True)

    @staticmethod
    def _copy_board(board: Board) -> Board:
        copied = Board()
        copied.grid = [row[:] for row in board.grid]
        copied.seen_positions = set(board.seen_positions)
        return copied

    def _pieces_for_role(self, counts: dict[str, int], role: str) -> int:
        if role == "attacker":
            return counts.get("A", 0)
        return counts.get("D", 0) + counts.get("K", 0)

    @staticmethod
    def _king_position(board: Board) -> Optional[Tuple[int, int]]:
        for y, row in enumerate(board.grid):
            for x, piece in enumerate(row):
                if piece == "K":
                    return x, y
        return None

    def _king_progress(self, board: Board) -> float:
        king_pos = self._king_position(board)
        if king_pos is None:
            return 1.0 if self.config.role == "attacker" else 0.0

        min_distance = min(
            abs(king_pos[0] - corner[0]) + abs(king_pos[1] - corner[1]) for corner in CORNERS
        )
        defender_progress = 1.0 - (float(min_distance) / MAX_MANHATTAN)
        defender_progress = max(0.0, min(1.0, defender_progress))

        if self.config.role == "defender":
            return defender_progress
        return 1.0 - defender_progress

    def _snapshot(self, board: Board) -> PieceSnapshot:
        counts = board.piece_counts()
        return PieceSnapshot(
            own_pieces=self._pieces_for_role(counts, self.config.role),
            opp_pieces=self._pieces_for_role(counts, opposite(self.config.role)),
            own_mobility=len(board.legal_moves(self.config.role)),
            opp_mobility=len(board.legal_moves(opposite(self.config.role))),
            king_progress=self._king_progress(board),
        )

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _potential(self, snapshot: PieceSnapshot) -> float:
        material_term = (
            float(snapshot.own_pieces) / float(self.max_own_pieces)
            - float(snapshot.opp_pieces) / float(self.max_opp_pieces)
        )
        mobility_term = float(snapshot.own_mobility - snapshot.opp_mobility) / MOBILITY_NORM
        king_term = (2.0 * snapshot.king_progress) - 1.0

        potential = (
            POTENTIAL_MATERIAL_WEIGHT * material_term
            + POTENTIAL_MOBILITY_WEIGHT * mobility_term
            + POTENTIAL_KING_WEIGHT * king_term
        )
        return self._clamp(potential, -1.0, 1.0)

    def _shaped_reward(self, before: PieceSnapshot, after: PieceSnapshot) -> float:
        # Potential-based shaping preserves optimal policies while densifying feedback.
        delta = (self.gamma * self._potential(after)) - self._potential(before)
        shaping = SHAPING_BETA * delta
        return self._clamp(shaping, -SHAPING_CLIP, SHAPING_CLIP)

    def _terminal_reward(self, winner: Optional[str]) -> float:
        if winner == self.config.role:
            return 1.0
        if winner == opposite(self.config.role):
            return -1.0
        return 0.0

    @staticmethod
    def _move_distance(move: Action) -> float:
        x1, y1 = Board.coord_to_xy(move[0])
        x2, y2 = Board.coord_to_xy(move[1])
        return float(abs(x1 - x2) + abs(y1 - y2))

    def _features_for_action(self, board_before: Board, action: Action) -> List[float]:
        before_counts = board_before.piece_counts()
        before_own = self._pieces_for_role(before_counts, self.config.role)
        before_opp = self._pieces_for_role(before_counts, opposite(self.config.role))

        board_after = self._copy_board(board_before)
        moved = board_after.apply_move(self.config.role, action[0], action[1])
        if not moved:
            return [0.0] * FEATURE_COUNT

        board_after.record_position()

        after_counts = board_after.piece_counts()
        after_own = self._pieces_for_role(after_counts, self.config.role)
        after_opp = self._pieces_for_role(after_counts, opposite(self.config.role))

        own_mobility = len(board_after.legal_moves(self.config.role))
        opp_mobility = len(board_after.legal_moves(opposite(self.config.role)))

        winner = board_after.winner()
        move_distance = self._move_distance(action)

        return [
            1.0,
            float(after_own) / float(self.max_own_pieces),
            float(after_opp) / float(self.max_opp_pieces),
            float(own_mobility) / MOBILITY_NORM,
            float(opp_mobility) / MOBILITY_NORM,
            float(before_opp - after_opp) / float(self.max_opp_pieces),
            float(before_own - after_own) / float(self.max_own_pieces),
            self._king_progress(board_after),
            1.0 if winner == self.config.role else 0.0,
            1.0 if winner == opposite(self.config.role) else 0.0,
            move_distance / float(BOARD_SIZE - 1),
        ]

    @staticmethod
    def _dot(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def _q_value(self, board: Board, action: Action) -> float:
        features = self._features_for_action(board, action)
        return self._dot(self.weights, features)

    def _max_future_q(self, board: Board) -> float:
        legal_moves = board.legal_moves(self.config.role)
        if not legal_moves:
            return 0.0
        return max(self._q_value(board, move) for move in legal_moves)

    def _clear_pending(self) -> None:
        self._pending_action = None
        self._pending_features = None
        self._pending_snapshot = None

    def _reset_game_training_stats(self) -> None:
        self._updates_in_game = 0
        self._sum_abs_td_error = 0.0
        self._sum_abs_delta_w = 0.0
        self._sum_shaped_reward = 0.0
        self._sum_terminal_reward = 0.0

    def _append_learning_metrics(
        self,
        game_id: int,
        winner: Optional[str],
        weight_norm: float,
        avg_abs_td_error: float,
        avg_abs_delta_w: float,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "game_id": game_id,
            "role": self.config.role,
            "winner": winner,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "updates": self._updates_in_game,
            "avg_abs_td_error": avg_abs_td_error,
            "avg_abs_delta_w": avg_abs_delta_w,
            "sum_shaped_reward": self._sum_shaped_reward,
            "sum_terminal_reward": self._sum_terminal_reward,
            "weight_l1_norm": weight_norm,
        }

        try:
            self.learning_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.learning_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        except OSError as exc:
            print(
                f"[warn] failed to append learning metrics to {self.learning_log_path}: {exc}",
                flush=True,
            )

    def _update_pending(self, board: Board, winner: Optional[str]) -> None:
        if not self.train:
            return
        if self._pending_action is None or self._pending_features is None:
            return
        if self._pending_snapshot is None:
            return

        next_snapshot = self._snapshot(board)
        shaped_reward = self._shaped_reward(self._pending_snapshot, next_snapshot)
        terminal_reward = self._terminal_reward(winner)
        reward = shaped_reward + terminal_reward
        self._sum_shaped_reward += shaped_reward
        self._sum_terminal_reward += terminal_reward

        future_q = 0.0 if winner is not None else self._max_future_q(board)

        current_q = self._dot(self.weights, self._pending_features)
        td_target = reward + self.gamma * future_q
        td_error = td_target - current_q

        update_scale = self.alpha * td_error
        delta_w_l1 = 0.0
        for idx in range(FEATURE_COUNT):
            delta = update_scale * self._pending_features[idx]
            self.weights[idx] += delta
            delta_w_l1 += abs(delta)

        self._updates_in_game += 1
        self._sum_abs_td_error += abs(td_error)
        self._sum_abs_delta_w += delta_w_l1

        if winner is not None:
            self._clear_pending()

    def _select_action(self, board: Board, legal_moves: List[Action]) -> Tuple[Action, List[float]]:
        scored: List[Tuple[Action, List[float], float]] = []
        for move in legal_moves:
            features = self._features_for_action(board, move)
            score = self._dot(self.weights, features)
            scored.append((move, features, score))

        if self.train and random.random() < self.epsilon:
            move, features, _ = random.choice(scored)
            return move, features

        best_score = max(score for _, _, score in scored)
        best = [(move, features) for move, features, score in scored if score == best_score]
        return random.choice(best)

    def choose_move(self, board: Board, role: str, game_id: int) -> Optional[Action]:
        del game_id

        if self.train:
            self._update_pending(board, winner=None)

        legal_moves = board.legal_moves(role)
        if not legal_moves:
            self._clear_pending()
            return None

        move, features = self._select_action(board, legal_moves)

        if self.train:
            self._pending_action = move
            self._pending_features = features
            self._pending_snapshot = self._snapshot(board)

        return move

    def on_game_started(self, game_id: int, board: Board) -> None:
        del game_id, board
        self._clear_pending()
        self._reset_game_training_stats()

    def on_game_finished(
        self,
        game_id: int,
        board: Board,
        winner: Optional[str],
        game_over_tokens: list[str],
    ) -> None:
        del game_over_tokens

        if self.train:
            self._update_pending(board, winner=winner)
            self._clear_pending()
            self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.save_q_table()
        weight_norm = sum(abs(weight) for weight in self.weights)
        if self._updates_in_game > 0:
            avg_abs_td_error = self._sum_abs_td_error / float(self._updates_in_game)
            avg_abs_delta_w = self._sum_abs_delta_w / float(self._updates_in_game)
        else:
            avg_abs_td_error = 0.0
            avg_abs_delta_w = 0.0
        self._append_learning_metrics(
            game_id=game_id,
            winner=winner,
            weight_norm=weight_norm,
            avg_abs_td_error=avg_abs_td_error,
            avg_abs_delta_w=avg_abs_delta_w,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parse_common_args(parser)
    parser.add_argument(
        "--qtable-path",
        default="qlearning/qtable.json",
        help="path to JSON file used for loading/saving linear Q weights",
    )
    parser.add_argument(
        "--weights-path",
        dest="qtable_path",
        help="alias for --qtable-path",
    )
    parser.add_argument(
        "--learning-log-path",
        default="qlearning/learning_metrics.jsonl",
        help="path to append per-game learning metrics as JSON lines",
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.02, 
        help="Q-learning alpha"
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.002,
        help="lower bound for alpha during decay",
    )
    parser.add_argument(
        "--alpha-decay",
        type=float,
        default=0.9997,
        help="multiplicative alpha decay applied at each game end",
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.95, 
        help="discount factor"
    )
    parser.add_argument(
        "--epsilon", 
        type=float, 
        default=0.2, 
        help="epsilon-greedy exploration"
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
        help="lower bound for epsilon during decay",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="multiplicative epsilon decay applied at each game end",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="disable Q updates (inference only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    player = QLearningPlayer(
        config=build_player_config(args),
        qtable_path=Path(args.qtable_path),
        learning_log_path=Path(args.learning_log_path),
        alpha=args.alpha,
        alpha_min=args.alpha_min,
        alpha_decay=args.alpha_decay,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        train=not args.no_train,
    )

    try:
        return player.run()
    finally:
        player.save_q_table()


if __name__ == "__main__":
    sys.exit(main())
