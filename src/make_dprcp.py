import argparse
import os
import sys

import active_learning
import worlds

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def main(args):
    world = worlds.get_world(args.world)
    agent = active_learning.ActiveLearningAgent(world, "", faiss_nprobe=10)
    agent.WriteDPRCP(args.random_pairs, test=args.test)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', required=True, choices=['qqp', 'wikiqa', 'inat', 'celeba'])
    parser.add_argument('--random_pairs', type=int) 
    parser.add_argument('--test', action='store_true')
    return parser

def parse_args():
    parser = get_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
