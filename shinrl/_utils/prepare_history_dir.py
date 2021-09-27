import os
import shutil

from .log import add_logfile_handler


def prepare_history_dir(dir_path: str, delete_existing: bool = False) -> None:
    if os.path.exists(dir_path) and delete_existing:
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    add_logfile_handler(os.path.join(dir_path, "log.txt"))
