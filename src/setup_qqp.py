"""Do initial conversion and split of QQP."""
import argparse
from collections import defaultdict
import numpy as np
import os
import random
import sys

import reader

def ReadQQPtsv(filename):
    qids = set()
    qid_to_questions = {}
    positive_pairs = set()
    stated_negative_pairs = set()

    with open(filename,'r') as f:
        i = 0
        for line in f.readlines():
            if i>0:
                pieces = line.split('\t')
                if len(pieces)==6:
                    pair_id = int(pieces[0].strip())
                    qid1 = int(pieces[1].strip())
                    qid2 = int(pieces[2].strip())
                    assert(qid1 != qid2)

                    question1 = pieces[3].strip()
                    question2 = pieces[4].strip()
        
                    label = int(pieces[5].strip())
                    assert(label==0 or label==1)

                    qids.add(qid1)
                    qids.add(qid2)
                    
                    if qid1 not in qid_to_questions:
                        qid_to_questions[qid1] = question1
                    if qid2 not in qid_to_questions:
                        qid_to_questions[qid2] = question2

                    if label == 1:
                        positive_pairs.add( tuple(sorted([qid1,qid2])) )
                    if label == 0:
                        stated_negative_pairs.add( tuple(sorted([qid1,qid2])) )
            i+=1


    print("finished reading {}".format(filename))
    print("\tnum questions = {}".format(len(qids)))
    print("\tnum positive pairs = {}".format(len(positive_pairs)))
    print("\tnum stated negative pairs = {}".format(len(stated_negative_pairs)))
    print("")

    return qids, qid_to_questions, positive_pairs, stated_negative_pairs

def GetConnectedComponents(nodes, pairs):
    edges = defaultdict(lambda: set())
    for pair in pairs:
        edges[pair[0]].add(pair[1])
        edges[pair[1]].add(pair[0])
    
    connected_components = set()
    discovered_nodes = set()
    for inspect_node in nodes:
        if inspect_node not in discovered_nodes:
            cc = set([inspect_node])
            queue = set([inspect_node])
            while len(queue)>0:
                next_node = queue.pop() 
                cc.add(next_node)
                queue.update(  edges[next_node].difference(cc)  )

            connected_components.add(tuple(sorted(list(cc))))
            discovered_nodes.update(cc)

    counts = defaultdict(lambda: 0)
    for cc in connected_components:
        counts[len(cc)]+=1

    return connected_components

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Do initial conversion
    qids, qtq, pos_pairs, stated_neg_pairs = ReadQQPtsv("{}/original/quora_duplicate_questions.tsv".format(args.raw_dir))
    reader.SaveQuestions(qids, qtq, "{}/qqp_questions.tsv".format(args.out_dir))
    reader.SavePairs(pos_pairs, "{}/qqp_pos_pairs.tsv".format(args.out_dir))
    reader.SavePairs(stated_neg_pairs, "{}/qqp_stated_neg_pairs.tsv".format(args.out_dir))

    # Compute transitive closure
    connected_components = GetConnectedComponents(qtq.keys(), pos_pairs)
    closure_pairs = set()
    for cc in connected_components:
        for i in range(len(cc)):
            for j in range(i+1,len(cc)):
                closure_pairs.add((cc[i],cc[j]))
    reader.SavePairs(closure_pairs,"{}/qqp_transitive_closure_pos_pairs.tsv".format(args.out_dir))

    # Partition the questions
    train_nodes = set()
    dev_nodes = set()
    test_nodes = set()
    random.seed(args.rng_seed)
    for cc in connected_components:
        if random.random()<0.5:
            for node in cc:
                train_nodes.add(node)
        else:
            if random.random()<0.5:
                for node in cc:
                    dev_nodes.add(node)
            else:
                for node in cc:
                    test_nodes.add(node)
    reader.SaveNodes(train_nodes,"{}/qqp_train.tsv".format(args.out_dir))
    reader.SaveNodes(dev_nodes,"{}/qqp_dev.tsv".format(args.out_dir))
    reader.SaveNodes(test_nodes,"{}/qqp_test.tsv".format(args.out_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir', help='Directory where raw QQP data is stored')
    parser.add_argument('out_dir', help='Directory to write processed data')
    parser.add_argument('--rng_seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
