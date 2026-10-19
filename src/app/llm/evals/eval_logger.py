import json
import os


class EvalLogger:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def add_item(self, data_dict: dict) -> None:
        dir_name = os.path.dirname(self.file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        current_data = []

        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    current_data = json.load(f)

                    if not isinstance(current_data, list):
                        current_data = [current_data]
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: Could not parse existing JSON ({e}). Overwriting with a fresh array.")
                current_data = []

        current_data.append(data_dict)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=4, ensure_ascii=False)


# Initialize the logger
logger = EvalLogger("./data/evaluations.json")
