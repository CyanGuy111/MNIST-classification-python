import pathlib

abs_path = pathlib.Path(__file__)

abs_path = abs_path.parent.parent

def getDir(s):
    return abs_path.joinpath(s)
