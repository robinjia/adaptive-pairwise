import argparse
import collections
import json
import os
import random


CELEBA_DIR = '/u/scr/nlp/dro/celebA/data/img_align_celeba'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
IDENT_FILE = '/u/scr/nlp/data/celebA/identity_CelebA.txt'
POS_PAIRS_FILE = os.path.join(DATA_DIR, 'celeba_pos_pairs.tsv')

def SaveNodes(nodes, filename):
    with open(filename, 'w') as f:
        for node in nodes:
            f.write('{}\n'.format(node))

def ReadNodes(filename):
    with open(filename) as f:
        return set([line.strip() for line in f])

def SavePairs(pairs, filename):
    with open(filename,'w') as f:
        for pair in pairs:
            f.write("{}\t{}\n".format(pair[0],pair[1]))

def ReadPairs(filename):
    pairs = set()
    with open(filename,'r') as f:
        for line in f:
            pairs.add(tuple(sorted([pair_part for pair_part in line[:-1].split('\t')])) )
    return pairs


def MakePartition():
    """Make train/dev/test split."""
    all_images = []
    label_to_images = collections.defaultdict(list)
    with open(IDENT_FILE) as f:
        for line in f:
            image_file, label = line.strip().split(' ')
            label_to_images[label].append(image_file)
            all_images.append(image_file)

    # Some statistics
    freq_labels = sorted(label_to_images.items(), key=lambda x: len(x[1]), reverse=True)
    for label, ids in freq_labels[:50]:
        print('{}: {} instances'.format(label, len(ids)))

    # Split the data
    random.seed(0)
    train_nodes = []
    dev_nodes = []
    test_nodes = []
    for image in all_images:
        if random.random()<0.5:
            train_nodes.append(image)
        else:
            if random.random()<0.5:
                dev_nodes.append(image)
            else:
                test_nodes.append(image)
    print('Splits: {} train, {} dev, {} test'.format(len(train_nodes), len(dev_nodes), len(test_nodes)))
    SaveNodes(train_nodes, os.path.join(DATA_DIR, 'celeba_train.tsv'))
    SaveNodes(dev_nodes, os.path.join(DATA_DIR, 'celeba_dev.tsv'))
    SaveNodes(test_nodes, os.path.join(DATA_DIR, 'celeba_test.tsv'))

    # Save positive pairs
    pos_pairs = []
    train_set = set(train_nodes)
    dev_set = set(dev_nodes)
    test_set = set(test_nodes)
    for label in label_to_images:
        for i in range(len(label_to_images[label])):
            for j in range(i):
                img_j = label_to_images[label][j]
                img_i = label_to_images[label][i]
                if any(img_j in x and img_i in x for x in (train_set, dev_set, test_set)):
                    pos_pairs.append((img_j, img_i))
    print('Found positive {} pairs'.format(len(pos_pairs)))
    SavePairs(pos_pairs, POS_PAIRS_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="")

    args = parser.parse_args()
    run_mode = args.mode

    if run_mode == 'make_celeba_partition':
        MakePartition()
