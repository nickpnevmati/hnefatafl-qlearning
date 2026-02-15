#!/usr/bin/env python3
"""Tabular Q-learning Hnefatafl player built on top of the shared player core."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from qlearning.player_core import (
        BasePlayer,
        Board,
        PlayerConfig,
        build_player_config,
        opposite,
        parse_common_args,
    )
except ModuleNotFoundError:
    from player_core import (
        BasePlayer,
        Board,
        PlayerConfig,
        build_player_config,
        opposite,
        parse_common_args,
    )

Action = Tuple[str, str]
StateKey = str
ActionKey = str
QTable = Dict[StateKey, Dict[ActionKey, float]]


@dataclass
class PieceSnapshot:
    attackers: int
    defenders: int
    king_alive: bool


class QLearningPlayer(BasePlayer):
    def __init__(
        self,
        config: PlayerConfig,
        qtable_path: Path,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        train: bool,
    ) -> None:
        super().__init__(config)
        self.qtable_path = qtable_path
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.train = train
        self.q_table: QTable = self._load_q_table()

        self._pending_state: Optional[StateKey] = None
        self._pending_action: Optional[ActionKey] = None
        self._pending_snapshot: Optional[PieceSnapshot] = None

    def _load_q_table(self) -> QTable:
        if not self.qtable_path.exists():
            return {}

        try:
            raw = json.loads(self.qtable_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] failed to load q-table from {self.qtable_path}: {exc}", flush=True)
            return {}

        if not isinstance(raw, dict):
            return {}

        cleaned: QTable = {}
        for state_key, action_map in raw.items():
            if not isinstance(state_key, str) or not isinstance(action_map, dict):
                continue
            cleaned[state_key] = {}
            for action_key, value in action_map.items():
                if not isinstance(action_key, str):
                    continue
                try:
                    cleaned[state_key][action_key] = float(value)
                except (TypeError, ValueError):
                    continue
        return cleaned

    def save_q_table(self) -> None:
        try:
            self.qtable_path.parent.mkdir(parents=True, exist_ok=True)
            self.qtable_path.write_text(
                json.dumps(self.q_table, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[warn] failed to save q-table to {self.qtable_path}: {exc}", flush=True)

    def _state_key(self, board: Board) -> StateKey:
        board_cells = "".join("".join(row) for row in board.grid)
        return f"{self.config.role}:{board_cells}"

    @staticmethod
    def _action_key(move: Action) -> ActionKey:
        return f"{move[0]}->{move[1]}"

    def _snapshot(self, board: Board) -> PieceSnapshot:
        counts = board.piece_counts()
        return PieceSnapshot(
            attackers=counts["A"],
            defenders=counts["D"],
            king_alive=counts["K"] > 0,
        )

    @staticmethod
    def _pieces_for_role(snapshot: PieceSnapshot, role: str) -> int:
        if role == "attacker":
            return snapshot.attackers
        return snapshot.defenders + (1 if snapshot.king_alive else 0)

    def _shaped_reward(self, before: PieceSnapshot, after: PieceSnapshot) -> float:
        own_before = self._pieces_for_role(before, self.config.role)
        own_after = self._pieces_for_role(after, self.config.role)
        opp_role = opposite(self.config.role)
        opp_before = self._pieces_for_role(before, opp_role)
        opp_after = self._pieces_for_role(after, opp_role)

        captures = opp_before - opp_after
        losses = own_before - own_after
        return 0.2 * float(captures - losses)

    def _terminal_reward(self, winner: Optional[str]) -> float:
        if winner == self.config.role:
            return 1.0
        if winner == opposite(self.config.role):
            return -1.0
        return 0.0

    def _q_value(self, state_key: StateKey, action_key: ActionKey) -> float:
        return self.q_table.get(state_key, {}).get(action_key, 0.0)

    def _max_future_q(self, state_key: StateKey, legal_moves: List[Action]) -> float:
        if not legal_moves:
            return 0.0
        return max(self._q_value(state_key, self._action_key(move)) for move in legal_moves)

    def _clear_pending(self) -> None:
        self._pending_state = None
        self._pending_action = None
        self._pending_snapshot = None

    def _update_pending(self, board: Board, winner: Optional[str]) -> None:
        if not self.train:
            return
        if self._pending_state is None or self._pending_action is None:
            return
        if self._pending_snapshot is None:
            return

        next_snapshot = self._snapshot(board)
        reward = self._shaped_reward(self._pending_snapshot, next_snapshot)
        reward += self._terminal_reward(winner)

        next_state = self._state_key(board)
        if winner is None:
            future_q = self._max_future_q(next_state, board.legal_moves(self.config.role))
        else:
            future_q = 0.0

        old_q = self._q_value(self._pending_state, self._pending_action)
        target = reward + self.gamma * future_q
        updated = old_q + self.alpha * (target - old_q)
        self.q_table.setdefault(self._pending_state, {})[self._pending_action] = updated

        if winner is not None:
            self._clear_pending()

    def _select_action(self, state_key: StateKey, legal_moves: List[Action]) -> Action:
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        scored_moves: List[Tuple[float, Action]] = []
        for move in legal_moves:
            action_key = self._action_key(move)
            scored_moves.append((self._q_value(state_key, action_key), move))

        best_score = max(score for score, _ in scored_moves)
        best_moves = [move for score, move in scored_moves if score == best_score]
        return random.choice(best_moves)

    def choose_move(self, board: Board, role: str, game_id: int) -> Optional[Action]:
        del game_id # TODO remove

        if self.train:
            self._update_pending(board, winner=None)

        legal_moves = board.legal_moves(role)
        if not legal_moves:
            self._clear_pending()
            return None

        state_key = self._state_key(board)
        move = self._select_action(state_key, legal_moves)

        if self.train:
            self._pending_state = state_key
            self._pending_action = self._action_key(move)
            self._pending_snapshot = self._snapshot(board)

        return move

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
            self._update_pending(board, winner=winner)
            self._clear_pending()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.save_q_table()
        print(
            f"[info] q-table states={len(self.q_table)} epsilon={self.epsilon:.5f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parse_common_args(parser)
    parser.add_argument(
        "--qtable-path",
        default="qlearning/qtable.json",
        help="path to a JSON file used for loading/saving Q-values",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="Q-learning alpha")
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
        help="multiplicative epsilon decay applied at each game end",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="disable Q-table updates (inference only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    player = QLearningPlayer(
        config=build_player_config(args),
        qtable_path=Path(args.qtable_path),
        alpha=args.alpha,
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
