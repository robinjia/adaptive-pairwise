#!/bin/bash
set -eu -o pipefail
if [ "$#" -eq 0 ]
then
  echo "Usage: $0 <qqp|wikiqa> [name] [flags]" 2>&1
  exit 1
fi
world=$1
name=$2
shift
shift
flags="$@"

desc="CosineBERT on ${world}, ${name}"
if [ ! -z "$flags" ]
then
  desc="${desc}, ${flags}"
fi
cl run :train_baselines.py :active_learning.py :get_close_pairs.py :sentence_embedder.py :model_utils.py :evaluation.py :worlds.py :reader.py :util.py cache:init-cache/cache :data "export PYTORCH_TRANSFORMERS_CACHE=cache; mkdir src; cd src; cp ../*.py .; cd ..; python3 src/train_baselines.py --world ${world} --output_dir out --sampling_technique certainty --loss_type sgdbn ${flags}; cp out/labels*/results_pr.json ." --request-docker-image robinjia/spal:0.1.4 --request-gpus 1 --request-cpus 4 --request-memory 32G --request-queue nlp -n "baseline-${world}" -d "${desc}"
