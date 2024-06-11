import os

DATASET = 'kin8nm'
BASIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{BASIS_DIR}{os.sep}exp{os.sep}{DATASET}"
print(OUT_DIR)

from uci_utils import runner




if __name__ == '__main__':
    debug = False
    runner(DATASET, debug, OUT_DIR, BASIS_DIR)
