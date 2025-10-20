import errno
import json
import os

DEFAULT_MYSQL_CONFIG = {
    "connection": {
        "host": "localhost",
        "database": "easy_price_monitor",
        "user": "easy-price-monitor",
        "password": "",
        "port": 3306,
    }
}


def load_mysql_config(MYSQL_CONFIG):
    """Load MySQL config from JSON file, if not create default file"""
    if not os.path.exists(MYSQL_CONFIG):
        with open(MYSQL_CONFIG, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_MYSQL_CONFIG, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Created default mySQL config file: {MYSQL_CONFIG}")
        return DEFAULT_MYSQL_CONFIG["connection"]

    with open(MYSQL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["connection"]
