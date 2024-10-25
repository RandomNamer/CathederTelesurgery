import os

def recursive_mkdir(target_path: str):
    return os.makedirs(target_path, exist_ok=True)
    if os.path.exists(os.path.dirname(target_path)):
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        return
    else:
        recursive_mkdir(os.path.dirname(target_path))