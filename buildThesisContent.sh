# Helper script to
# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$SCRIPT_DIR/.env"
set +a

# Active venv if present

if [ -f "$HOME/venv/bin/activate" ]; then
  source "$HOME/venv/bin/activate"
fi

# Copy some scripts to show example of code
# cp evaluation/eval.py $THESIS_PATH/eval.py
# cp evaluation/download.py $THESIS_PATH/download.py

# Change to all script sources
cd $SCRIPT_DIR
# export THESIS_PATH='.'
# Generate the disribtuion of metrics
# Cache load and rebug output
python3 -m common.utils
python3 -m common.loadCache
python3 -m analysis.metrics
python3 -m analysis.domains.chosen
python3 -m analysis.domains.done
python3 -m analysis.domains.anova
python3 -m analysis.domains.paramCorrelation
python3 -m analysis.domains.modelRankingTable
python3 -m analysis.domains.taskSpecializationGain
python3 -m analysis.domains.tauGrid
python3 -m analysis.domains.ndcgLatexTable
python3 -m analysis.languages.ndcgLatexTable
python3 -m analysis.languages.anova
python3 -m analysis.languages.used
python3 -m analysis.languages.done
python3 -m analysis.languages.tauGrid
python3 -m analysis.code.jinaCompare
python3 -m analysis.lv.compare
python3 -m analysis.lv.ailabCompare
python3 -m analysis.lv.ndcgLatexTable
#
# Unused
# python3 -m analysis.domains.tauMDS
# python3 -m analysis.languages.tauMDS
