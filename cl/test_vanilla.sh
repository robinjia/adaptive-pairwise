#!/bin/bash
set -eu -o pipefail
if [ "$#" -eq 0 ]
then
  echo "Usage: $0 <qqp|wikiqa> <bert|xlnet|roberta|albert> [train bundle] [flags]" 2>&1
  exit 1
fi
task=$1
model=$2
train_bundle=$3
shift
shift
shift
flags="$@"

desc="Test ${model} on ${task}, train=${train_bundle}"
if [ ! -z "$flags" ]
then
  desc="${desc}, ${flags}"
fi

docker_image="robinjia/spal:0.1.4"
if [ "$task" == "wikiqa" ]
then
  flags="$flags --wikiqa"
fi

if [ "$model" == "bert" ]
then
  details="base-uncased"
  case="uncased"
elif [ "$model" == "xlnet" ]
then
  details="base-cased"
  case="cased"
elif [ "$model" == "roberta" ]
then
  details="base"
  case="cased"
elif [ "$model" == "albert" ]
then
  details="base-v2"
  case="cased"
fi
if [ $case == "uncased" ]
then
  flags="$flags --do_lower_case"
fi
cl run :run_glue_transformers.py :evaluation.py :worlds.py :reader.py :util.py cache:init-cache/cache model:${train_bundle}/out :data "export PYTORCH_TRANSFORMERS_CACHE=cache; mkdir src; cd src; cp ../*.py .; cd ..; mkdir out; cd out; ln -s ../model/config.json; ln -s ../model/pytorch_model.bin; ln -s ../model/special_tokens_map.json; ln -s ../model/spiece.model; ln -s ../model/tokenizer_config.json; ln -s ../model/training_args.bin; ln -s ../model/merges.txt; ln -s ../model/vocab.txt; ln -s ../model/vocab.json; cd ..; python3 src/run_glue_transformers.py --model_type $model --model_name_or_path "${model}-${details}" --task_name QQP --do_eval --custom_train_file data/stated_${task}/train.tsv --custom_dev_file data/stated_${task}/dev.tsv --max_seq_length 128 --per_gpu_eval_batch_size=32 --output_dir out --test --run_pr_curve $flags" --request-docker-image "${docker_image}" --request-gpus 1 --request-cpus 4 --request-memory 32G --request-queue nlp -n test-${model}-${task} -d "${desc}"

