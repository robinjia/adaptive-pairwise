#!/bin/bash
set -eu -o pipefail
wget http://nlp.stanford.edu/data/robinjia/mussmann2020_pairwise_data.zip
unzip mussmann2020_pairwise_data.zip
rm mussmann2020_pairwise_data.zip
