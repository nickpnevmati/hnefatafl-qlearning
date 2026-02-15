#!/usr/bin/env python3
"""LSPI-based Hnefatafl player built on top of the shared player core."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

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
FrozenBoard = Tuple[str, ...]

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


@dataclass
class Snapshot:
    own_pieces: int
    opp_pieces: int
    king_progress: float


@dataclass
class Transition:
    state: FrozenBoard
    action: Action
    reward: float
    next_state: Optional[FrozenBoard]
    done: bool


class LSPIPlayer(BasePlayer):
    def __init__(
        self,
        config: PlayerConfig,
        weights_path: Path,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        regularization: float,
        lspi_iterations: int,
        min_samples: int,
        buffer_size: int,
        train: bool,
    ) -> None:
        super().__init__(config)
        self.weights_path = weights_path
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.regularization = regularization
        self.lspi_iterations = lspi_iterations
        self.min_samples = min_samples
        self.train = train

        self.max_own_pieces = (
            ATTACKER_START_COUNT if self.config.role == "attacker" else DEFENDER_START_COUNT
        )
        self.max_opp_pieces = (
            DEFENDER_START_COUNT if self.config.role == "attacker" else ATTACKER_START_COUNT
        )

        self.weights = self._load_weights()
        self.replay: Deque[Transition] = deque(maxlen=buffer_size)

        self._pending_state: Optional[FrozenBoard] = None
        self._pending_action: Optional[Action] = None
        self._pending_snapshot: Optional[Snapshot] = None

    def _load_weights(self) -> List[float]:
        if not self.weights_path.exists():
            return [0.0] * FEATURE_COUNT

        try:
            data = json.loads(self.weights_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] failed to load LSPI weights from {self.weights_path}: {exc}", flush=True)
            return [0.0] * FEATURE_COUNT

        if isinstance(data, dict):
            raw_weights = data.get("weights", [])
        elif isinstance(data, list):
            raw_weights = data
        else:
            raw_weights = []

        if not isinstance(raw_weights, list):
            raw_weights = []

        weights: List[float] = [0.0] * FEATURE_COUNT
        for idx in range(min(FEATURE_COUNT, len(raw_weights))):
            try:
                weights[idx] = float(raw_weights[idx])
            except (TypeError, ValueError):
                weights[idx] = 0.0
        return weights

    def save_weights(self) -> None:
        payload = {
            "weights": self.weights,
            "feature_count": FEATURE_COUNT,
            "role": self.config.role,
        }
        try:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            self.weights_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[warn] failed to save LSPI weights to {self.weights_path}: {exc}", flush=True)

    @staticmethod
    def _freeze_board(board: Board) -> FrozenBoard:
        return tuple("".join(row) for row in board.grid)

    @staticmethod
    def _thaw_board(state: FrozenBoard) -> Board:
        board = Board()
        board.grid = [list(row) for row in state]
        return board

    @staticmethod
    def _copy_board(board: Board) -> Board:
        copied = Board()
        copied.grid = [row[:] for row in board.grid]
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

    def _snapshot(self, board: Board) -> Snapshot:
        counts = board.piece_counts()
        return Snapshot(
            own_pieces=self._pieces_for_role(counts, self.config.role),
            opp_pieces=self._pieces_for_role(counts, opposite(self.config.role)),
            king_progress=self._king_progress(board),
        )

    def _shaped_reward(self, before: Snapshot, after: Snapshot) -> float:
        captures = before.opp_pieces - after.opp_pieces
        losses = before.own_pieces - after.own_pieces
        progress_gain = after.king_progress - before.king_progress
        return 0.2 * float(captures - losses) + 0.1 * progress_gain

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

    def _features(self, state: FrozenBoard, action: Action) -> List[float]:
        board = self._thaw_board(state)
        return self._features_for_action(board, action)

    @staticmethod
    def _dot(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def _q_value_with_weights(
        self,
        state: FrozenBoard,
        action: Action,
        weights: Sequence[float],
    ) -> float:
        return self._dot(weights, self._features(state, action))

    def _greedy_action_from_state(
        self,
        state: FrozenBoard,
        weights: Sequence[float],
    ) -> Optional[Action]:
        board = self._thaw_board(state)
        legal_moves = board.legal_moves(self.config.role)
        if not legal_moves:
            return None

        best_score: Optional[float] = None
        best_moves: List[Action] = []
        for action in legal_moves:
            score = self._q_value_with_weights(state, action, weights)
            if best_score is None or score > best_score:
                best_score = score
                best_moves = [action]
            elif score == best_score:
                best_moves.append(action)

        return random.choice(best_moves)

    def _solve_linear_system(self, matrix: List[List[float]], target: List[float]) -> List[float]:
        n = len(target)
        augmented = [matrix[i][:] + [target[i]] for i in range(n)]

        for col in range(n):
            pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
            if abs(augmented[pivot][col]) < 1e-12:
                continue

            if pivot != col:
                augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

            pivot_value = augmented[col][col]
            for j in range(col, n + 1):
                augmented[col][j] /= pivot_value

            for row in range(n):
                if row == col:
                    continue
                factor = augmented[row][col]
                if abs(factor) < 1e-12:
                    continue
                for j in range(col, n + 1):
                    augmented[row][j] -= factor * augmented[col][j]

        solution = [0.0] * n
        for row in range(n):
            diag = augmented[row][row]
            if abs(diag) < 1e-8:
                continue
            solution[row] = augmented[row][n] / diag
        return solution

    def _train_lspi(self) -> None:
        if not self.train or len(self.replay) < self.min_samples:
            return

        weights = self.weights[:]

        for _ in range(self.lspi_iterations):
            matrix = [[0.0 for _ in range(FEATURE_COUNT)] for _ in range(FEATURE_COUNT)]
            target = [0.0 for _ in range(FEATURE_COUNT)]

            for transition in self.replay:
                phi_sa = self._features(transition.state, transition.action)

                if transition.done or transition.next_state is None:
                    phi_next = [0.0] * FEATURE_COUNT
                else:
                    next_action = self._greedy_action_from_state(transition.next_state, weights)
                    if next_action is None:
                        phi_next = [0.0] * FEATURE_COUNT
                    else:
                        phi_next = self._features(transition.next_state, next_action)

                for i in range(FEATURE_COUNT):
                    target[i] += phi_sa[i] * transition.reward
                    for j in range(FEATURE_COUNT):
                        matrix[i][j] += phi_sa[i] * (phi_sa[j] - self.gamma * phi_next[j])

            for i in range(FEATURE_COUNT):
                matrix[i][i] += self.regularization

            new_weights = self._solve_linear_system(matrix, target)
            delta = max(abs(new_weights[i] - weights[i]) for i in range(FEATURE_COUNT))
            weights = new_weights
            if delta < 1e-5:
                break

        self.weights = weights

    def _clear_pending(self) -> None:
        self._pending_state = None
        self._pending_action = None
        self._pending_snapshot = None

    def _finalize_pending(self, board: Board, winner: Optional[str], done: bool) -> None:
        if not self.train:
            return
        if self._pending_state is None or self._pending_action is None:
            return
        if self._pending_snapshot is None:
            return

        after = self._snapshot(board)
        reward = self._shaped_reward(self._pending_snapshot, after)
        reward += self._terminal_reward(winner)

        self.replay.append(
            Transition(
                state=self._pending_state,
                action=self._pending_action,
                reward=reward,
                next_state=None if done else self._freeze_board(board),
                done=done,
            )
        )

        self._clear_pending()

    def choose_move(self, board: Board, role: str, game_id: int) -> Optional[Action]:
        del game_id

        if self.train:
            self._finalize_pending(board, winner=None, done=False)

        legal_moves = board.legal_moves(role)
        if not legal_moves:
            self._clear_pending()
            return None

        state = self._freeze_board(board)

        if self.train and random.random() < self.epsilon:
            action = random.choice(legal_moves)
        else:
            best_score: Optional[float] = None
            best_actions: List[Action] = []
            for candidate in legal_moves:
                score = self._q_value_with_weights(state, candidate, self.weights)
                if best_score is None or score > best_score:
                    best_score = score
                    best_actions = [candidate]
                elif score == best_score:
                    best_actions.append(candidate)
            action = random.choice(best_actions)

        if self.train:
            self._pending_state = state
            self._pending_action = action
            self._pending_snapshot = self._snapshot(board)

        return action

    def on_game_started(self, game_id: int, board: Board) -> None:
        del game_id, board
        self._clear_pending()

    def on_game_finished(
        self,
        game_id: int,
        board: Board,
        winner: Optional[str],
        game_over_tokens: list[str],
    ) -> None:
        del game_id, game_over_tokens

        if self.train:
            self._finalize_pending(board, winner=winner, done=True)
            self._train_lspi()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.save_weights()
        print(
            (
                f"[info] LSPI samples={len(self.replay)} "
                f"epsilon={self.epsilon:.5f} "
                f"w0={self.weights[0]:.5f}"
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parse_common_args(parser)
    parser.add_argument(
        "--weights-path",
        default="qlearning/lspi_weights.json",
        help="path to JSON file used for loading/saving LSPI weights",
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="epsilon-greedy exploration")
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.02,
        help="lower bound for epsilon during decay",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="multiplicative epsilon decay applied after each game",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-3,
        help="ridge regularization added to LSPI normal matrix",
    )
    parser.add_argument(
        "--lspi-iterations",
        type=int,
        default=5,
        help="policy-iteration passes per training update",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="minimum replay transitions before running LSPI",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=5000,
        help="maximum replay transitions retained in memory",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="disable LSPI updates (inference only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    player = LSPIPlayer(
        config=build_player_config(args),
        weights_path=Path(args.weights_path),
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        regularization=args.regularization,
        lspi_iterations=args.lspi_iterations,
        min_samples=args.min_samples,
        buffer_size=args.buffer_size,
        train=not args.no_train,
    )

    try:
        return player.run()
    finally:
        player.save_weights()


if __name__ == "__main__":
    sys.exit(main())
