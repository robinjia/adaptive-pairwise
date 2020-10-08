#!/bin/bash
set -eu -o pipefail
if [ "$#" -eq 0 ]
then
  echo "Usage: $0 <qqp|wikiqa> <bert|xlnet|roberta|albert> train_file dev_file [flags]" 2>&1
  exit 1
fi
task=$1
model=$2
train_file=$3
dev_file=$4
shift
shift
shift
shift
flags="$@"

desc="Train ${model} on ${task}, train=${train_file}, dev=${dev_file}"
if [ ! -z "$flags" ]
then
  desc="${desc}, ${flags}"
fi

if [ "$task" == "wikiqa" ]
then
  flags="$flags --wikiqa"
  epochs=3
else
  epochs=2
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
cl run :run_glue_transformers.py :evaluation.py :worlds.py :reader.py :util.py cache:init-cache/cache train.tsv:"$train_file" dev.tsv:"$dev_file" :data "export PYTORCH_TRANSFORMERS_CACHE=cache; mkdir src; cd src; cp ../*.py .; cd ..; python3 src/run_glue_transformers.py --model_type $model --model_name_or_path "${model}-${details}" --task_name QQP --do_train --do_eval --custom_train_file train.tsv --custom_dev_file dev.tsv --max_seq_length 128 --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=32 --learning_rate 2e-5 --num_train_epochs $epochs --save_steps 0 --output_dir out $flags" --request-docker-image robinjia/spal:0.1.4 --request-gpus 1 --request-cpus 4 --request-memory 32G --request-queue nlp -n train-${model}-${task} -d "${desc}"
