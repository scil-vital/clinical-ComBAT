
def get_root():
    import os
    return os.path.realpath(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

COMBAT_ROOT = get_root()