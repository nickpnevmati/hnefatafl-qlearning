#!/usr/bin/env python3
"""Minimal interactive Hnefatafl text-protocol client."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
from typing import TextIO

DEFAULT_VERSION_ID = "ad746a65"
DEFAULT_PORT = 49152


def send_line(sock: socket.socket, message: str) -> None:
    """Send a single protocol line."""
    data = f"{message.rstrip()}\n".encode("utf-8")
    sock.sendall(data)
    print(f">> {message}")


def read_line(reader: TextIO) -> str:
    """Read a single line or raise if the connection closes."""
    line = reader.readline()
    if not line:
        raise RuntimeError("server closed the connection")
    print(f"<< {line.rstrip()}")
    return line.rstrip("\n")


def stream_responses(reader: TextIO) -> None:
    """Continuously print server responses until the socket closes."""
    try:
        for line in reader:
            print(f"<< {line.rstrip()}")
    except Exception as exc:  # pragma: no cover - interactive convenience
        print(f"[reader] {exc}")
    finally:
        print("[reader] connection closed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="server hostname or IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="server port")
    parser.add_argument("--username", required=True, help="account username")
    parser.add_argument("--password", default="", help="account password")
    parser.add_argument(
        "--create-account",
        action="store_true",
        help="send a create_account command before logging in",
    )
    parser.add_argument(
        "--version-id",
        default=DEFAULT_VERSION_ID,
        help="VERSION_ID expected by the server",
    )
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port)) as sock:
        reader = sock.makefile("r", encoding="utf-8", newline="\n")

        if args.create_account:
            send_line(
                sock,
                f"{args.version_id} create_account {args.username} {args.password}",
            )
            read_line(reader)

        send_line(sock, f"{args.version_id} login {args.username} {args.password}")
        response = read_line(reader)
        if response != "= login":
            print("login failed; exiting")
            return 1

        thread = threading.Thread(target=stream_responses, args=(reader,), daemon=True)
        thread.start()

        print("Connected. Type protocol commands, Ctrl-D to quit.")
        try:
            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                send_line(sock, line)
        except (KeyboardInterrupt, BrokenPipeError):
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
