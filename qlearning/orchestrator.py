import subprocess

def spawn_attacker(game_id: int):
    attacker_proc = subprocess.run(
        [
            'python',
            'qlearning/random_player.py',
            '--username',
            ' q-attacker',
            '--role',
            'attacker',
            '--join-game',
            game_id
        ]
    )

q_process = subprocess.Popen(
    [
        'python', 
        'qlearning/random_player.py',
        '--username',
        'q-defender',
        '--auto-accept',
    ],
    stdout=subprocess.PIPE, 
    text=True,
    bufsize=1,
)

for line in q_process.stdout:
    if 'new game id' not in line:
        continue

    game_id: str = line.split(' ')[-1]
    spawn_attacker(game_id)
