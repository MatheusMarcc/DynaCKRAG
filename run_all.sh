#!/usr/bin/env bash
# ragbench — versão Linux/WSL/macOS do run_all.ps1.
#   ./run_all.sh [perfil] [all|ragcritic|dynakrag]               roda tudo pela linha de comando
#   ./run_all.sh painel [perfil]                                 abre o painel no navegador
set -euo pipefail
cd "$(dirname "$0")"
MODE="bench"
if [ "${1:-}" = "painel" ]; then MODE="painel"; shift; fi
PROFILE="${1:-quick}"
PAPER="${2:-all}"
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
mkdir -p ragbench_results/logs
[ -x .venv/bin/python ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install -q --upgrade pip
$PY -m pip install -r requirements.txt
if [ "$PROFILE" != "smoke" ]; then
  curl -sf http://localhost:11434/api/version >/dev/null || { echo "Ollama não está rodando (ollama serve)"; exit 1; }
  ollama pull qwen2.5:7b
  ollama pull hf.co/mradermacher/RAG-Critic-3B-GGUF:Q4_K_M
  ollama pull bge-large || echo "sem o bge-large no Ollama: os índices serão montados na CPU (mais devagar, mesmo resultado)"
fi
if [ "$MODE" = "painel" ]; then
  if [ $# -ge 1 ]; then exec $PY -m ragbench ui --profile "$PROFILE"; else exec $PY -m ragbench ui; fi
fi
$PY -m ragbench doctor --profile "$PROFILE" || echo "doctor apontou problemas; continuando..."
$PY -m ragbench prepare --profile "$PROFILE"
$PY -m ragbench bench --profile "$PROFILE" --paper "$PAPER"
$PY -m ragbench report --profile "$PROFILE"
echo "Pronto: ragbench_results/$PROFILE/REPORT.md e FALHAS.md. Para abrir o painel: ./run_all.sh painel"
