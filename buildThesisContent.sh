# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$SCRIPT_DIR/.env"
set +a

# Active venv if present

if [ -f "$HOME/venv/bin/activate" ]; then
  source "$HOME/venv/bin/activate"
fi

# Change to all script sources
cd $SCRIPT_DIR
# Generate the disribtuion of metrics
python3 -m thesisContent.metricCount
python3 -m thesisContent.retrievalDomains
python3 -m thesisContent.doneDomains
