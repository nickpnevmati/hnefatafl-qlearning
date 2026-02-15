#!/usr/bin/env python3
"""Random-policy Hnefatafl text-protocol client."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

try:
    from qlearning.player_core import (
        BasePlayer,
        Board,
        build_player_config,
        parse_common_args,
    )
except ModuleNotFoundError:
    from player_core import BasePlayer, Board, build_player_config, parse_common_args


class RandomPlayer(BasePlayer):
    def choose_move(
        self,
        board: Board,
        role: str,
        game_id: int,
    ) -> Optional[Tuple[str, str]]:
        del game_id
        return board.random_legal_move(role)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parse_common_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    player = RandomPlayer(build_player_config(args))
    return player.run()


if __name__ == "__main__":
    sys.exit(main())
