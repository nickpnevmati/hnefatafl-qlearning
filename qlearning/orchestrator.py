import subprocess
import signal
import sys
import atexit
import time

# List to hold all running subprocesses
subprocesses = []

def cleanup():
    """Terminate all tracked subprocesses."""
    print("Cleaning up subprocesses...", file=sys.stderr)
    for proc in subprocesses[:]:  # iterate over a copy
        if proc.poll() is None:   # still running
            proc.terminate()       # send SIGTERM
            # Optionally wait a bit for graceful shutdown
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()        # force kill if necessary
                proc.wait()
    subprocesses.clear()

def signal_handler(signum, frame):
    """Handle termination signals by calling cleanup and exiting."""
    print(f"Received signal {signum}", file=sys.stderr)
    cleanup()
    sys.exit(0)

# Register signal handlers for SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register cleanup for normal interpreter exit (e.g., sys.exit())
atexit.register(cleanup)

def spawn_attacker(game_id: str):
    """Start an attacker in the background and track it."""
    proc = subprocess.Popen(
        [
            'python',
            'qlearning/random_player.py',
            '--username',
            'q-attacker',
            '--role',
            'attacker',
            '--join-game',
            game_id
        ]
    )
    subprocesses.append(proc)
    return proc

def main():
    # Start the defender process
    q_process = subprocess.Popen(
        [
            'python', 
            'qlearning/qlearning_player.py',
            '--username',
            'q-defender',
            '--role',
            'defender',
            '--auto-accept',
            '--games',
            '1000',
        ],
        stdout=subprocess.PIPE, 
        text=True,
        bufsize=1,
    )
    subprocesses.append(q_process)

    # Read defender output line by line
    for line in q_process.stdout:
        if 'new game id' not in line:
            continue

        time.sleep(1)

        # Extract game ID (assuming it's the last token)
        game_id_str = line.strip().split(' ')[-1]
        if not game_id_str.isdigit():
            print(f"Invalid game ID from line: {line!r}", file=sys.stderr)
            continue

        spawn_attacker(game_id_str)

    # After the loop ends (defender finished), wait for attackers to finish?
    # You may want to wait for all attackers here if desired.
    # For now, we'll just let cleanup handle them on exit.

if __name__ == "__main__":
    main()