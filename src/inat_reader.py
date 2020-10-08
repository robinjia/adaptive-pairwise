import argparse
import collections
import json
import os
import random

INAT_DIR = '/u/scr/nlp/data/iNaturalist/2018'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
FILENAMES_FILE = os.path.join(DATA_DIR, 'inat_filenames.json')
POS_PAIRS_FILE = os.path.join(DATA_DIR, 'inat_pos_pairs.tsv')

def ReadAnnotations():
    with open(os.path.join(INAT_DIR, 'train2018.json')) as f:
        train = json.load(f)
    with open(os.path.join(INAT_DIR, 'val2018.json')) as f:
        val = json.load(f)
    all_annotations = train['annotations'] + val['annotations']
    print('Read {} annotations'.format(len(all_annotations)))
    id_to_category = {}
    for a in all_annotations:
        if a['image_id'] != a['id']:
            print('id mismatch', a)
        if a['image_id'] in id_to_category:
            print('id exists', a)
        id_to_category[a['image_id']] = a['category_id']
    
    all_images = train['images'] + val['images']
    print('Read {} images'.format(len(all_images)))
    id_to_filename = {x['id']: x['file_name'] for x in all_images}
    return id_to_category, id_to_filename

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
    id_to_category, id_to_filename = ReadAnnotations()
    with open(FILENAMES_FILE, 'w') as f:
        json.dump(id_to_filename, f)

    category_to_ids = collections.defaultdict(list)
    for img_id, cat in id_to_category.items():
        category_to_ids[cat].append(img_id)

    # Some statistics
    freq_cats = sorted(category_to_ids.items(), key=lambda x: len(x[1]), reverse=True)
    for cat, ids in freq_cats[:50]:
        print('{}: {} instances'.format(cat, len(ids)))

    # Split the data
    random.seed(0)
    train_nodes = []
    dev_nodes = []
    test_nodes = []
    for cat in category_to_ids:
        cur_ids = category_to_ids[cat]
        if len(cur_ids) > 30:  # Downsample labels to have at most 30 examples
            cur_ids = random.sample(cur_ids, 30)
            category_to_ids[cat] = cur_ids
        for image in cur_ids:
            if random.random()<0.5:
                train_nodes.append(image)
            else:
                if random.random()<0.5:
                    dev_nodes.append(image)
                else:
                    test_nodes.append(image)
    print('Splits: {} train, {} dev, {} test'.format(len(train_nodes), len(dev_nodes), len(test_nodes)))
    SaveNodes(train_nodes, os.path.join(DATA_DIR, 'inat_train.tsv'))
    SaveNodes(dev_nodes, os.path.join(DATA_DIR, 'inat_dev.tsv'))
    SaveNodes(test_nodes, os.path.join(DATA_DIR, 'inat_test.tsv'))

    # Save positive pairs
    pos_pairs = []
    train_set = set(train_nodes)
    dev_set = set(dev_nodes)
    test_set = set(test_nodes)
    for cat in category_to_ids:
        for i in range(len(category_to_ids[cat])):
            for j in range(i):
                id_j = category_to_ids[cat][j]
                id_i = category_to_ids[cat][i]
                if any(id_j in x and id_i in x for x in (train_set, dev_set, test_set)):
                    pos_pairs.append((id_j, id_i))
    print('Found positive {} pairs'.format(len(pos_pairs)))
    SavePairs(pos_pairs, POS_PAIRS_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="")

    args = parser.parse_args()
    run_mode = args.mode

    if run_mode == 'make_inat_partition':
        MakePartition()
