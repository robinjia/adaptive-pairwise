# On the Importance of Adaptive Data Collection for Extremely Imbalanced Pairwise Tasks

This is the official GitHub repository for the following paper:

> **[On the Importance of Adaptive Data Collection for Extremely Imbalanced Pairwise Tasks.]()**  
> Stephen Mussmann,\* Robin Jia,\* and Percy Liang.  
> _Findings of EMNLP_, 2020.  

For more details on how to reproduce all the experiments in the paper, please see the associated [CodaLab Worksheet](https://worksheets.codalab.org/worksheets/0x39ba5559790b4099a7ff75f916ce19a4).

## Setup
1. Install packages:

```
pip install -r requirements.txt
```

For `faiss` on Ubuntu, you may also have to install `libopenblas-dev` and `libomp-dev`:

```
sudo apt-get install libopenblas-dev libomp-dev
```

If you're using Docker, you can also refer to the Dockerfile in this repository,
or just use [this Docker image](https://hub.docker.com/r/robinjia/spal).

2. Download data:

```
bash pull-deps.sh
```

## Running experiments
For full details on how to run all of our experiments, see the commands used in the aforementioned [CodaLab Worksheet](https://worksheets.codalab.org/worksheets/0x39ba5559790b4099a7ff75f916ce19a4).

An example command to run uncertainty sampling on QQP is:
```
python3 src/active_learning.py --world qqp --output_dir my_model_directory --sampling_technique uncertainty --loss_type sgdbn 
```
By default this will save checkpoints from every round of active learning, which takes up a lot of disk space. You can only save the last checkpoint by passing the flag `--save_last_model`.

After active learning finishes, you can evaluate the model on the test set by running:
```
python3 src/run_on_test.py --world qqp --load_dir my_model_directory/labels232100 --output_dir my_results_directory --sampling_technique uncertainty --loss_type sgdbn --test
```
Without the `--test` flag, this will run on dev.

## Codalab scripts
The `cl` directory contains scripts that launch jobs on CodaLab that reproduce our experiments.
You may find this useful if you also use CodaLab, or merely as documentation of what commands we ran.

## Code to run vision experiments
An earlier version of this paper was submitted to ICML and also included experiments on two computer vision datasets, iNaturalist and CelebA.
The code is retained in `src` in case it becomes useful for future work---see `image_embedder.py`, `celeba_reader.py`, and `inat_reader.py`.

## Making your own splits and evaluation sets
The `data` directory created by `pull-deps.sh` contains all the files needed to run on the same exact dataset splits and evaluations sets as we use in our paper.
These files should be used when reproducing or comparing to results from this paper.

We have also included scripts that can be used to generate new splits and evaluation sets.
These may be useful for processing new datasets, or for measuring variance across different dataset splits.

1. For QQP, prepare the initial partition of questions by running:

```
python3 src/setup_qqp.py <path_to_GLUE_QQP_dir> <out_dir>
```

2. For either QQP or WikiQA, generate the set of evaluation pairs by running:

```
python3 src/make_dprcp.py --world [qqp|wikiqa] --random_pairs 10000000 # Pass --test for test set
```
This will use 10 million random pairs, in addition to positives and "nearby" negatives defined by faiss.

3. Generate stated datasets (all positives plus stated negatives) using the current train/dev/test split by running:
```
python3 src/create_data_files.py <out_dir> [qqp|wikiqa]
```
