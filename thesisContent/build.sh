# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

# Active venv if present

if [ -f "$HOME/venv/bin/activate" ]; then
  source "$HOME/venv/bin/activate"
fi

run_script() {
  local python_bin="$1"
  shift

  (
    cd "$THESIS_PATH" || exit 1
    "$python_bin" "$SCRIPT_DIR/$@"
  )
}

# Change to all script sources
cd $SCRIPT_DIR
# Generate the disribtuion of metrics
run_script python3 metricCount.py
THESIS_PATH=$THESIS_PATH python3 english.py
