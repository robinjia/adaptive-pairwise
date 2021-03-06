#!/bin/bash
set -eu -o pipefail
if [ "$#" -eq 0 ]
then
  echo "Usage: $0 <qqp|wikiqa> [train dir] [flags]" 2>&1
  exit 1
fi
world=$1
train_dir=$2
shift
shift
flags="$@"

desc="Test CosineBERT on ${world}, train=${train_dir}"
if [ ! -z "$flags" ]
then
  desc="${desc}, ${flags}"
fi

cl run :run_on_test.py :active_learning.py :get_close_pairs.py :sentence_embedder.py :model_utils.py :evaluation.py :worlds.py :reader.py :util.py cache:init-cache/cache :data model:"$train_dir" "export PYTORCH_TRANSFORMERS_CACHE=cache; mkdir src; cd src; cp ../*.py .; cd ..; python3 src/run_on_test.py --world ${world} --output_dir out --sampling_technique certainty --loss_type sgdbn --load_dir model --test ${flags}" --request-docker-image robinjia/spal:0.1.4 --request-gpus 1 --request-cpus 4 --request-memory 32G --request-queue nlp -n "test-${world}" -d "${desc}"
