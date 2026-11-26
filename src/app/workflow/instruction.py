from pathlib import Path


def get_instructions(path: Path | str = r"src\app\workflow\agent.md"):
    with open(path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
        return system_prompt
