from . import mysql_storage
from utils import load_mysql_config

def get_playlist_history(playlist_ids_array):
    db_config = load_mysql_config("mysql_config.json")
    return mysql_storage.get_playlist_history(db_config, playlist_ids_array)

def STORAGE_HANDLERS(handler_name: str | None = None, **kwargs):
    if handler_name in (None, "mysql"):
        return mysql_storage
    raise ValueError(f"Unknown storage handler: {handler_name}")
