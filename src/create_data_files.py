"""Helper script to create data files with stated examples."""
import argparse
import random
import os
import sys

import worlds


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def write_examples(examples, out_file):
    with open(out_file, 'w') as f:
        for (qid1, qid2), label in examples:
            print('{}\t{}\t{}'.format(qid1, qid2, label), file=f)

def write_glue_qqp(out_dir, split):
    examples = []
    with open(os.path.join(DATA_DIR, 'QQP', '{}.tsv'.format(split))) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            toks = line.strip().split('\t')
            if len(toks) != 6:
                print(i, line)
                continue
            qid1 = int(toks[1])
            qid2 = int(toks[2])
            label = int(toks[5])
            examples.append(((qid1, qid2), label))
    out_file = os.path.join(out_dir, '{}.tsv'.format(split))
    write_examples(examples, out_file)

def write_file(out_dir, name, world, primary, secondary):
    examples = []
    for pair in world.positive_pairs:
        if pair[0] in primary and pair[1] in secondary:
            examples.append((pair, 1))
    for pair in world.stated_negative_pairs:
        if pair[0] in primary and pair[1] in secondary:
            examples.append((pair, 0))
    print('Found {} positives, {} negatives, {} total'.format(
            sum(1 for x in examples if x[1] == 1),
            sum(1 for x in examples if x[1] == 0),
            len(examples)))
    random.shuffle(examples)
    out_file = os.path.join(out_dir, '{}.tsv'.format(name))
    write_examples(examples, out_file)


def main(args):
    random.seed(0)
    world = worlds.get_world(args.world)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.glue_qqp:
        write_glue_qqp(args.out_dir, 'train')
        write_glue_qqp(args.out_dir, 'dev')
    else:
        write_file(args.out_dir, 'train', world, world.GetTrainPrimary(), 
                   world.GetTrainPrimary() if world.symmetric else world.GetTrainSecondary())
        write_file(args.out_dir, 'dev', world, world.GetDevPrimary(), 
                   world.GetDevPrimary() if world.symmetric else world.GetDevSecondary())
        write_file(args.out_dir, 'test', world, world.GetTestPrimary(), 
                   world.GetTestPrimary() if world.symmetric else world.GetTestSecondary())
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('world', choices=['qqp', 'wikiqa'])
    parser.add_argument('--glue_qqp', action='store_true', help='Write the GLUE QQP train/dev split')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    main(args)
