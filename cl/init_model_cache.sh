#!/bin/bash
cl run :init_model_cache.py 'PYTORCH_TRANSFORMERS_CACHE=cache python3 init_model_cache.py' --request-docker-image robinjia/spal:0.1.4 --request-network -n 'init-cache'
