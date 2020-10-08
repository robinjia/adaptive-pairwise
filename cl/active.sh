#!/bin/bash
set -eu -o pipefail
if [ "$#" -eq 0 ]
then
  echo "Usage: $0 <qqp|wikiqa> <uncertainty|certainty> [flags]" 2>&1
  exit 1
fi
world=$1
method=$2
shift
shift
flags="$@"

desc="${method} on ${world}"
if [ ! -z "$flags" ]
then
  desc="${desc}, ${flags}"
fi
if [ "${world}" == "wikiqa" ]
then
  flags="${flags} --al_batches 4"
fi

cl run :active_learning.py :get_close_pairs.py :sentence_embedder.py :model_utils.py :evaluation.py :worlds.py :reader.py :util.py cache:init-cache/cache :data "export PYTORCH_TRANSFORMERS_CACHE=cache; mkdir src; cd src; cp ../*.py .; cd ..; python3 src/active_learning.py --world ${world} --output_dir out --sampling_technique ${method} --loss_type sgdbn --save_last_model ${flags}" --request-docker-image robinjia/spal:0.1.4 --request-gpus 1 --request-cpus 4 --request-memory 32G --request-queue nlp -n "${method}-${world}" -d "${desc}"
