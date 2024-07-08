import yaml
import sys


def get_args(path) -> dict:
    default_path = path

    try:
        model_info = yaml.load(open(f'{default_path}', 'r'), Loader=yaml.Loader)
    except FileNotFoundError:
        model_info = None
        print(f'[ERROR] Config "{default_path}" not found \n[INFO] Aborting')
        sys.exit()
    return model_info
