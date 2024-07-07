import yaml
import sys


def get_args() -> dict:

    default_name = 'default.yml'
    default_path = './cfgs/'
    default_path = '/home/mateusz/Desktop/Demand-Forecast/cfgs/default.yml'

    try:
        model_info = yaml.load(open(f'{default_path}', 'r'), Loader=yaml.Loader)
    except FileNotFoundError:
        model_info = None
        print(f'[ERROR] Config "{default_name}" not found \n[INFO] Aborting')
        sys.exit()
    return model_info
