import os
from uci_utils import runner

DATASET = 'concrete'
BASIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{BASIS_DIR}{os.sep}exp{os.sep}{DATASET}"
print(OUT_DIR)


if __name__ == '__main__':
    debug = False
    runner(DATASET, debug, OUT_DIR, BASIS_DIR)
