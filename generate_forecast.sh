#!/bin/bash

# ===============================
# Settings
# ===============================

PROJECT_DIR="/home/sebastian/projects/easy-playlist-forecast" #Replace with yours project path
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/debug.log"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python3"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
SCRIPT_PATH="$PROJECT_DIR/easy_playlist_forecast.py"
if [ "$#" -gt 0 ]; then
    PLAYLIST_IDS_TO_FORECAST=("$@") # read playli id's from CLI
else
    PLAYLIST_IDS_TO_FORECAST=(2 3) # replace with your playlist id's
fi
# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# ===============================
# Helper functions
# ===============================

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

init_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    $PYTHON_BIN -m pip install --upgrade pip
    $PYTHON_BIN -m pip install -r "$REQUIREMENTS_FILE"
}

# ===============================
# Main logic
# ===============================

cd "$PROJECT_DIR" || exit 1
init_venv

log "Starting script"

ARGS=""

for id in "${PLAYLIST_IDS_TO_FORECAST[@]}"; do
    ARGS+=" $id"
done

if [ -f "$SCRIPT_PATH" ]; then
    $PYTHON_BIN "$SCRIPT_PATH" --playlist_ids $ARGS
else
    log "ERROR: Script file '$SCRIPT_PATH' not found!"
fi

deactivate
log "Script finished"
