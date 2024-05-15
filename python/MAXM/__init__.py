import os

__all__ = ['base_dir']

base_dir = f"{os.getenv('MAXMROOT')}"

def create_path(path):
    if not os.path.exists(path):
        print(f"Creating dir {path}")
        is_global = str(path).startswith("/")
        list_subdirs = path.split("/")
        tmp_path = "" if not is_global else "/"
        for subdir in list_subdirs:
            tmp_path += f"{subdir}/"
            if not subdir or subdir in [".", ".."]:
                continue
            elif not os.path.exists(tmp_path):
                os.system(f"mkdir {tmp_path}")
