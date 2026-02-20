import os
import json

def reset_game_ids_in_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for idx, line in enumerate(lines):
        data = json.loads(line)
        data['game_id'] = idx
        new_lines.append(json.dumps(data) + '\n')
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

def main():
    for filename in os.listdir('.'):
        if filename.endswith('.jsonl'):
            print(f"Processing {filename}...")
            reset_game_ids_in_jsonl(filename)
    print("Done.")

if __name__ == "__main__":
    main()