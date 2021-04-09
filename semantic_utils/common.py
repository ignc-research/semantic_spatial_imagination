import pathlib
def safe_mkdir(path):
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    except:
        pass