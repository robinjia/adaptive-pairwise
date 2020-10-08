"""Utilities for reading/writing our data."""
def SavePairs(pairs, filename):
    with open(filename,'w') as f:
        for pair in pairs:
            f.write("{}\t{}\n".format(pair[0],pair[1]))
    return


def ReadPairs(filename):
    pairs = set()
    with open(filename,'r') as f:
        for line in f.readlines():
            pairs.add( tuple(sorted([pair_part for pair_part in line[:-1].split('\t')])) )
    return pairs


def SaveQuestions(qids, qtq, filename):
    with open(filename,'w') as f:
        for qid in qids:
            f.write("{}\t{}\n".format(qid,qtq[qid]))
    return

def ReadQuestions(filename):
    qtq = {}
    with open(filename,"r") as f:
        for line in f.readlines():
            qid, question_text = line[:-1].split('\t')
            qtq[qid] = question_text
    return qtq

def SaveNodes(nodes, filename):
    with open(filename,'w') as f:
        for node in nodes:
            f.write("{}\n".format(node))
    return

def ReadNodes(filename):
    nodes = set()
    with open(filename,'r') as f:
        for line in f.readlines():
            nodes.add( line[:-1] )
    return nodes
