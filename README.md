## Running Two AI Agents Locally

These steps show how to launch a local Copenhagen Hnefatafl server, start the Python sample player, and challenge it with the built-in Rust AI so the two agents play each other.

1. **Build the server and reference AI once.**
   ```sh
   cd hnefatafl
   cargo build --release --bin hnefatafl-server-full --bin hnefatafl-ai
   ```

2. **Start the local server in an in-memory mode.** (Keep this terminal open.)
   ```sh
   cargo run --release --bin hnefatafl-server-full -- \
     --skip-the-data-file --skip-advertising-updates --skip-message
   ```
   The server listens on `localhost:49152` and speaks the text protocol version `ad746a65`.

3. **Run the Python agent (attacker) in another terminal.**
   ```sh
   python qlearning/sample_player.py \
     --host localhost \
     --username sample \
     --password secret \
     --create-account
   ```
   After `= login`, send `change_password secret` (optional) and
   `new_game attacker rated fischer 900000 10 11`. Capture the numeric `game <id>`
   returned by the server; the Python agent should store this automatically once the
   stdin loop is replaced with logic that reacts to server messages.

4. **Launch the Rust AI as the challenger/defender.**
   ```sh
   cargo run --release --bin hnefatafl-ai -- \
     --host localhost \
     --username defender \
     --password secret \
     --ai basic \
     --join_game <id>
   ```
   This binary issues `join_game_pending <id>` for you and waits until the game
   creator accepts.

5. **Accept the challenge from the Python side.**
   When the Python client reads `= challenge_requested <id>`, respond with
   `join_game <id>` (your future automation should send this immediately). From then on:
   - The server alternates `game <id> generate_move attacker|defender` and `game <id> play …`
     messages.
   - When you receive `generate_move` for your role, compute a move and answer with
     `game <id> play <role> <FROM> <TO>` (or `… resigns _`). The string must follow the
     `Plae` format defined by the engine.
   - When you receive a `play` message for the opponent, update your local board state
     before handling the next request.

Following the command/response style implemented in `hnefatafl/src/bin/hnefatafl-ai.rs`
lets the Python agent act autonomously without relying on manual stdin input.
