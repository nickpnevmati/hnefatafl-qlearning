#!/usr/bin/env python3
"""Shared Hnefatafl text-protocol player core and board logic."""

from __future__ import annotations

import argparse
import random
import socket
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional, Set, TextIO, Tuple

DEFAULT_VERSION_ID = "ad746a65"
DEFAULT_PORT = 49152
DEFAULT_TIME = "rated fischer 900000 10 11"
BOARD_SIZE = 11
BOARD_LETTERS = "abcdefghijk"
THRONE = (5, 5)
RESTRICTED = {(0, 0), (10, 0), (0, 10), (10, 10), (5, 5)}  # corners + throne
CORNERS = {(0, 0), (10, 0), (0, 10), (10, 10)}
STARTING_POSITION_11X11 = [
    "...XXXXX...",
    ".....X.....",
    "...........",
    "X....O....X",
    "X...OOO...X",
    "XX.OOKOO.XX",
    "X...OOO...X",
    "X....O....X",
    "...........",
    ".....X.....",
    "...XXXXX...",
]


@dataclass
class PlayerConfig:
    host: str
    port: int
    username: str
    password: str
    role: str
    games: int
    create_account: bool
    auto_accept: bool
    auto_resign: bool
    stdin: bool
    version_id: str
    account_only: bool
    time_control: str
    join_game: Optional[int]


@dataclass
class Board:
    grid: List[List[str]]
    seen_positions: Set[str]

    def __init__(self) -> None:
        self.grid = [
            list(row.replace("X", "A").replace("O", "D").replace("K", "K"))
            for row in STARTING_POSITION_11X11
        ]
        self.seen_positions = set()
        self.begin_game_history()

    @staticmethod
    def coord_to_xy(coord: str) -> Tuple[int, int]:
        col = coord[0].lower()
        row = int(coord[1:])
        x = BOARD_LETTERS.index(col)
        y = BOARD_SIZE - row
        return x, y

    @staticmethod
    def xy_to_coord(x: int, y: int) -> str:
        return f"{BOARD_LETTERS[x]}{BOARD_SIZE - y}"

    def piece_at(self, x: int, y: int) -> str:
        return self.grid[y][x]

    @staticmethod
    def role_of(piece: str) -> str:
        if piece == "A":
            return "attacker"
        if piece in ("D", "K"):
            return "defender"
        return "roleless"

    def path_clear(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        if x1 == x2:
            step = 1 if y2 > y1 else -1
            for y in range(y1 + step, y2 + step, step):
                if self.grid[y][x1] != ".":
                    return False
        elif y1 == y2:
            step = 1 if x2 > x1 else -1
            for x in range(x1 + step, x2 + step, step):
                if self.grid[y1][x] != ".":
                    return False
        else:
            return False
        return True

    def legal_destination(self, piece: str, x: int, y: int) -> bool:
        if piece != "K" and (x, y) in RESTRICTED:
            return False
        return self.grid[y][x] == "."

    def apply_move(self, role: str, from_coord: str, to_coord: str) -> bool:
        fx, fy = self.coord_to_xy(from_coord)
        tx, ty = self.coord_to_xy(to_coord)

        piece = self.piece_at(fx, fy)
        if self.role_of(piece) != role:
            return False
        if not self.path_clear(fx, fy, tx, ty):
            return False
        if not self.legal_destination(piece, tx, ty):
            return False

        self.grid[fy][fx] = "."
        self.grid[ty][tx] = piece

        self._capture_adjacent(tx, ty, role)
        self._maybe_capture_king(role)
        return True

    def _capture_adjacent(self, tx: int, ty: int, role_from: str) -> None:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = tx + dx, ty + dy
            bx, by = tx + 2 * dx, ty + 2 * dy
            if not (0 <= ax < BOARD_SIZE and 0 <= ay < BOARD_SIZE):
                continue

            opp_piece = self.piece_at(ax, ay)
            if self.role_of(opp_piece) != opposite(role_from) or opp_piece == "K":
                continue

            if 0 <= bx < BOARD_SIZE and 0 <= by < BOARD_SIZE:
                support_piece = self.piece_at(bx, by)
                support_role = self.role_of(support_piece)
                throne_blocked_by_king = (bx, by) == THRONE and support_piece == "K"
                restricted_hostile = (bx, by) in RESTRICTED and not throne_blocked_by_king
                if support_role == role_from or restricted_hostile:
                    self.grid[ay][ax] = "."

    def _maybe_capture_king(self, role_from: str) -> None:
        if role_from != "attacker":
            return

        king_pos = self._find_piece("K")
        if not king_pos:
            return

        kx, ky = king_pos
        neighbors = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = kx + dx, ky + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                neighbors.append(self.piece_at(nx, ny) == "A" or (nx, ny) in RESTRICTED)
            else:
                neighbors.append(False)

        if all(neighbors):
            self.grid[ky][kx] = "."

    def _find_piece(self, target: str) -> Optional[Tuple[int, int]]:
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.grid[y][x] == target:
                    return x, y
        return None

    def _moves_from(self, x: int, y: int, piece: str) -> List[Tuple[str, str]]:
        moves: List[Tuple[str, str]] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if self.grid[ny][nx] != ".":
                    break
                if self.legal_destination(piece, nx, ny):
                    moves.append((self.xy_to_coord(x, y), self.xy_to_coord(nx, ny)))
                nx += dx
                ny += dy
        return moves

    def legal_moves(self, role: str) -> List[Tuple[str, str]]:
        moves: List[Tuple[str, str]] = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                piece = self.grid[y][x]
                if self.role_of(piece) != role:
                    continue
                moves.extend(self._moves_from(x, y, piece))
        if role == "defender":
            moves = [
                move
                for move in moves
                if not self._would_repeat_position(role, move[0], move[1])
            ]
        return moves

    def random_legal_move(self, role: str) -> Optional[Tuple[str, str]]:
        moves = self.legal_moves(role)
        if not moves:
            return None
        return random.choice(moves)

    def piece_counts(self) -> dict[str, int]:
        counts = {"A": 0, "D": 0, "K": 0}
        for row in self.grid:
            for cell in row:
                if cell in counts:
                    counts[cell] += 1
        return counts

    def winner(self) -> Optional[str]:
        king_pos = self._find_piece("K")
        if king_pos is None:
            return "attacker"
        if king_pos in CORNERS:
            return "defender"
        return None

    def position_key(self) -> str:
        return "".join("".join(row) for row in self.grid)

    def begin_game_history(self) -> None:
        self.seen_positions = {self.position_key()}

    def record_position(self) -> None:
        self._print_board()
        self.seen_positions.add(self.position_key())

    def _would_repeat_position(self, role: str, from_coord: str, to_coord: str) -> bool:
        if role != "defender":
            return False
        if not self.seen_positions:
            return False

        sim_board = Board()
        sim_board.grid = [row[:] for row in self.grid]
        sim_board.seen_positions = set()

        if not sim_board.apply_move(role, from_coord, to_coord):
            return True

        return sim_board.position_key() in self.seen_positions

    def _print_board(self):
        for row in self.grid:
            print(
                "".join("X" if c == "A" else "O" if c == "D" else c for c in row),
                flush=True,
            )

def opposite(role: str) -> str:
    return "defender" if role == "attacker" else "attacker"


def send_line(sock: socket.socket, message: str) -> None:
    data = f"{message.rstrip()}\n".encode("utf-8")
    sock.sendall(data)
    print(f">> {message}", flush=True)


def read_line(reader: TextIO) -> str:
    line = reader.readline()
    if not line:
        raise RuntimeError("server closed the connection")
    print(f"<< {line.rstrip()}", flush=True)
    return line.rstrip("\n")


def stdin_passthrough(sock: socket.socket) -> None:
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if line:
                send_line(sock, line)
    except Exception as exc:  # pragma: no cover - interactive convenience
        print(f"[stdin] {exc}")
    finally:
        print("[stdin] stopped")


def extract_game_id(tokens: list[str]) -> int | None:
    if len(tokens) >= 4 and tokens[1] == "new_game":
        try:
            return int(tokens[3])
        except ValueError:
            return None
    return None


def is_game_over(tokens: list[str], game_id: int | None) -> bool:
    if "game_over" in tokens:
        return game_id is None or str(game_id) in tokens
    return False


def infer_winner_from_tokens(tokens: list[str]) -> Optional[str]:
    if "game_over" not in tokens:
        return None

    if "attacker_wins" in tokens:
        return "attacker"
    if "defender_wins" in tokens:
        return "defender"
    if "draw" in tokens:
        return "draw"

    roles = [role for role in ("attacker", "defender") if role in tokens]
    if len(roles) == 1:
        return roles[0]

    return None


def parse_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--host", default="localhost", help="server hostname or IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="server port")
    parser.add_argument("--username", required=True, help="account username")
    parser.add_argument("--password", default="", help="account password")
    parser.add_argument(
        "--role",
        choices=("attacker", "defender"),
        default="attacker",
        help="role when creating games",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="number of games to play back-to-back (0 = infinite)",
    )
    parser.add_argument(
        "--create-account",
        action="store_true",
        help="send a create_account command before logging in",
    )
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help="auto-accept challenges for the game you created",
    )
    parser.add_argument(
        "--auto-resign",
        action="store_true",
        help="auto-resign on your turn",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="keep a background stdin passthrough for manual commands",
    )
    parser.add_argument(
        "--version-id",
        default=DEFAULT_VERSION_ID,
        help="VERSION_ID expected by the server",
    )
    parser.add_argument(
        "--account-only",
        action="store_true",
        help="exit immediately after logging in",
    )
    parser.add_argument(
        "--time-control",
        default=DEFAULT_TIME,
        help="time control string for new_game",
    )
    parser.add_argument(
        "--join-game",
        type=int,
        default=None,
        help="join a pending game id instead of creating a new game",
    )
    return parser


def build_player_config(args: argparse.Namespace) -> PlayerConfig:
    return PlayerConfig(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        role=args.role,
        games=args.games,
        create_account=args.create_account,
        auto_accept=args.auto_accept,
        auto_resign=args.auto_resign,
        stdin=args.stdin,
        version_id=args.version_id,
        account_only=args.account_only,
        time_control=args.time_control,
        join_game=args.join_game,
    )


class BasePlayer:
    """Protocol loop with overridable move selection hooks."""

    def __init__(self, config: PlayerConfig) -> None:
        self.config = config

    def choose_move(
        self,
        board: Board,
        role: str,
        game_id: int,
    ) -> Optional[Tuple[str, str]]:
        raise NotImplementedError

    def on_game_started(self, game_id: int, board: Board) -> None:
        del game_id, board

    def on_opponent_move(
        self,
        board: Board,
        role: str,
        from_coord: str,
        to_coord: str,
    ) -> None:
        del board, role, from_coord, to_coord

    def on_game_finished(
        self,
        game_id: int,
        board: Board,
        winner: Optional[str],
        game_over_tokens: list[str],
    ) -> None:
        del game_id, board, winner, game_over_tokens

    def _handle_generate_move(self, sock: socket.socket, game_id: int, board: Board) -> None:
        if self.config.auto_resign:
            send_line(sock, f"game {game_id} play {self.config.role} resigns _")
            return

        move = self.choose_move(board, self.config.role, game_id)
        if move is None:
            send_line(sock, f"game {game_id} play {self.config.role} resigns _")
            return

        from_coord, to_coord = move
        send_line(sock, f"game {game_id} play {self.config.role} {from_coord} {to_coord}")
        moved = board.apply_move(self.config.role, from_coord, to_coord)
        if moved:
            board.record_position()

    def _play_active_game(self, sock: socket.socket, reader: TextIO, game_id: int, board: Board) -> None:
        while True:
            line = read_line(reader)
            tokens = line.split()

            if len(tokens) >= 3 and tokens[0] == "game" and tokens[1] == str(game_id):
                if (
                    len(tokens) >= 4
                    and tokens[2] == "generate_move"
                    and tokens[3] == self.config.role
                ):
                    self._handle_generate_move(sock, game_id, board)
                    continue

                if tokens[2] == "play" and len(tokens) >= 6:
                    role_play = tokens[3]
                    if tokens[4] != "resigns" and role_play != self.config.role:
                        moved = board.apply_move(role_play, tokens[4], tokens[5])
                        if moved:
                            board.record_position()
                            self.on_opponent_move(
                                board,
                                role_play,
                                tokens[4],
                                tokens[5],
                            )
                    continue

                if tokens[2] == "game_over":
                    winner = infer_winner_from_tokens(tokens) or board.winner()
                    print(
                        f"[info] game {game_id} over; starting next if scheduled",
                        flush=True,
                    )
                    self.on_game_finished(game_id, board, winner, tokens)
                    break

            if is_game_over(tokens, game_id):
                winner = infer_winner_from_tokens(tokens) or board.winner()
                print(
                    f"[info] game {game_id} over (detected late); "
                    "starting next if scheduled",
                    flush=True,
                )
                self.on_game_finished(game_id, board, winner, tokens)
                break

    def _run_join_game(self, sock: socket.socket, reader: TextIO, game_id: int) -> int:
        send_line(sock, f"join_game_pending {game_id}")

        while True:
            line = read_line(reader)
            tokens = line.split()
            if len(tokens) >= 2 and tokens[0] == "=" and tokens[1] == "join_game_pending":
                break
            if len(tokens) >= 2 and tokens[0] == "?" and tokens[1] == "join_game_pending":
                print(f"[error] failed to join pending game {game_id}", flush=True)
                return 1

        board = Board()
        self.on_game_started(game_id, board)
        self._play_active_game(sock, reader, game_id, board)
        return 0

    def _login(self, sock: socket.socket, reader: TextIO) -> bool:
        if self.config.create_account:
            send_line(
                sock,
                f"{self.config.version_id} create_account "
                f"{self.config.username} {self.config.password}",
            )
            if read_line(reader) == "= login":
                return True

        send_line(
            sock,
            f"{self.config.version_id} login {self.config.username} {self.config.password}",
        )
        return read_line(reader) == "= login"

    def run(self) -> int:
        games_to_play = self.config.games

        with socket.create_connection((self.config.host, self.config.port)) as sock:
            reader = sock.makefile("r", encoding="utf-8", newline="\n")

            if not self._login(sock, reader):
                print("login failed; exiting", flush=True)
                return 1

            if self.config.account_only:
                print("[info] logged in; exiting due to --account-only", flush=True)
                return 0

            if self.config.stdin:
                threading.Thread(target=stdin_passthrough, args=(sock,), daemon=True).start()

            if self.config.join_game is not None:
                return self._run_join_game(sock, reader, self.config.join_game)

            played = 0
            while games_to_play == 0 or played < games_to_play:
                send_line(sock, f"new_game {self.config.role} {self.config.time_control}")

                game_id = None
                while game_id is None:
                    line = read_line(reader)
                    game_id = extract_game_id(line.split())

                print(f"[info] new game id: {game_id}", flush=True)
                board = Board()
                self.on_game_started(game_id, board)

                while True:
                    line = read_line(reader)
                    tokens = line.split()
                    if "challenge_requested" not in tokens:
                        continue

                    print("[info] challenge requested; accepting", flush=True)
                    if self.config.auto_accept:
                        send_line(sock, f"join_game {game_id}")
                    else:
                        print("[warn] auto-accept disabled; skipping join", flush=True)
                    break

                self._play_active_game(sock, reader, game_id, board)

                played += 1

        return 0


__all__ = [
    "BOARD_SIZE",
    "Board",
    "BasePlayer",
    "DEFAULT_PORT",
    "DEFAULT_TIME",
    "DEFAULT_VERSION_ID",
    "PlayerConfig",
    "build_player_config",
    "opposite",
    "parse_common_args",
]
