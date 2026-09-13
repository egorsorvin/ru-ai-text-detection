#!/usr/bin/env bash
# Full pipeline for a single big GPU (80 GB). Every stage is idempotent: re-running the script
# after a crash continues from the first unfinished stage. Logs go to outputs/logs/.
#
#   nohup bash run_all.sh > outputs/run_all.log 2>&1 &
#   tail -f outputs/run_all.log
#
# Stages: setup -> data -> generate -> features -> zero-shot eval -> mixed training
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
mkdir -p outputs/logs data/gen
PY=python
CAP=${CAP:-16000}            # balanced rows per corpus for training mixes and their features
ENCODER=${ENCODER:-ai-forever/ruRoberta-large}
GEN_N=${GEN_N:-1000}         # sources per task per generator -> 3*GEN_N machine texts
GENERATORS=${GENERATORS:-"t-tech/T-pro-it-2.0:tpro2_32b Qwen/Qwen3.8-27B:qwen38_27b"}
PAIRS=${PAIRS:-"Qwen/Qwen3-14B-Base,Qwen/Qwen3-14B:qwen3_14b yandex/YandexGPT-5-Lite-8B-pretrain,yandex/YandexGPT-5-Lite-8B-instruct:yagpt5_8b Qwen/Qwen3-4B-Base,Qwen/Qwen3-4B:qwen3_4b"}
MAIN_PAIR=${MAIN_PAIR:-qwen3_14b}

stage() { echo; echo "===== [$(date +%H:%M:%S)] $1"; }

stage "setup"
if [ ! -f outputs/.setup_done ]; then
  pip install -q -U pip
  pip install -q "vllm>=0.10" hf_transfer
  pip install -q transformers datasets scikit-learn pandas pyarrow accelerate joblib
  touch outputs/.setup_done
fi
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

stage "data"
$PY -c "from src.corpora import load_corpus; [print(c, len(load_corpus(c))) for c in ['coat','llmtrace','ainl']]" 2>&1 | tail -3

stage "generate"
for spec in $GENERATORS; do
  model=${spec%%:*}; tag=${spec##*:}
  if [ -f "data/gen/${tag}_testset.parquet" ]; then echo "$tag: done"; continue; fi
  $PY scripts/generate_set.py --model "$model" --tag "$tag" --n "$GEN_N" --backend vllm --bs 64 2>&1 | tee "outputs/logs/gen_${tag}.log" | grep -E "jobs|generated|test set|^\[|Error|Traceback" || true
done

stage "features"
for spec in $PAIRS; do
  models=${spec%%:*}; tag=${spec##*:}
  obs=${models%%,*}; perf=${models##*,}
  common="--observer $obs --performer $perf --tag $tag --quant bf16 --bs 32 --max_len 256"
  for c in coat llmtrace ainl; do
    $PY scripts/score_binoculars.py --corpus $c --split test $common 2>&1 | tee -a "outputs/logs/feat_${tag}.log" | grep -E "requested|scored|Error|Traceback" || true
  done
  for g in data/gen/*_testset.parquet; do
    [ -e "$g" ] || continue
    gtag=$(basename "$g" _testset.parquet)
    $PY scripts/score_binoculars.py --corpus "gen_${gtag}" --split test $common 2>&1 | tee -a "outputs/logs/feat_${tag}.log" | grep -E "requested|scored|Error|Traceback" || true
  done
  $PY scripts/score_binoculars.py --corpus coat --split dev $common 2>&1 | tee -a "outputs/logs/feat_${tag}.log" | grep -E "requested|scored|Error|Traceback" || true
  if [ "$tag" = "$MAIN_PAIR" ]; then
    for c in coat llmtrace ainl; do
      $PY scripts/score_binoculars.py --corpus $c --split train --cap $CAP $common 2>&1 | tee -a "outputs/logs/feat_${tag}.log" | grep -E "requested|scored|Error|Traceback" || true
      $PY scripts/score_binoculars.py --corpus $c --split dev $common 2>&1 | tee -a "outputs/logs/feat_${tag}.log" | grep -E "requested|scored|Error|Traceback" || true
    done
  fi
  $PY scripts/eval_binoculars_cache.py --tag $tag --quant bf16 2>&1 | grep -E "threshold|accuracy=" || true
done

stage "mixed training"
train_run() {  # train_run <corpora> <run-name> [extra args]
  local corpora=$1 run=$2; shift 2
  if [ -f "outputs/results/${run}__coat_test.json" ]; then echo "$run: done"; return; fi
  $PY scripts/train_mix.py --train "$corpora" --run "$run" --model "$ENCODER" --cap "$CAP" --bs 16 --lr 1e-5 --epochs 2 --eval_every 300 "$@" 2>&1 | tee "outputs/logs/${run}.log" | grep -E "^train |== step|accuracy=|done in|Error|Traceback" || true
}
train_run coat,llmtrace     mix_coat+llmtrace
train_run coat,ainl         mix_coat+ainl
train_run llmtrace,ainl     mix_llmtrace+ainl
train_run coat,llmtrace     mix_coat+llmtrace_bino     --bino --bino_tag $MAIN_PAIR --bino_quant bf16
train_run coat,ainl         mix_coat+ainl_bino         --bino --bino_tag $MAIN_PAIR --bino_quant bf16
train_run llmtrace,ainl     mix_llmtrace+ainl_bino     --bino --bino_tag $MAIN_PAIR --bino_quant bf16
train_run coat,llmtrace,ainl mix_all_bino              --bino --bino_tag $MAIN_PAIR --bino_quant bf16
train_run coat,llmtrace,ainl mix_all

stage "summary"
$PY scripts/show_results.py
echo "ALL DONE"
