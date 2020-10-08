"""General utilities."""
import os
from typing import Union, List

from sentence_transformers.readers import InputExample

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PAWS_DIR = os.path.join(DATA_DIR, 'paws_qqp_data')

class PairExample(InputExample):
    """An InputExample that also stores sentence ID's."""
    def __init__(self, guid: str, sent_ids: List[str], texts: List[str], label: Union[int, float]):
        self.guid = guid
        self.sent_ids = sent_ids
        self.texts = texts
        self.label = label

def load_custom_data(world, filename):
    """Read files output by create_data_files.py"""
    examples = []
    with open(filename) as f:
        for i, line in enumerate(f):
            qid1, qid2, label = line.strip().split('\t')
            q1 = world.id_to_text[qid1]
            q2 = world.id_to_text[qid2]
            examples.append(PairExample(
                    '{}-load-{}-{}'.format(i, qid1, qid2), 
                    [qid1, qid2], [q1, q2], int(label)))
    return examples

def load_paws_qqp_data(split=None):
    """Load PAWS QQP data.
    
    Generated from the instructions in https://github.com/google-research-datasets/paws
    Default: Read in all data together, since we don't train on any of it
    """
    splits = [split] if split else ['train', 'dev_and_test']
    examples = []
    for s in splits:
        filename = os.path.join(PAWS_DIR, '{}.tsv'.format(s))
        with open(filename) as f:
            for i, line in enumerate(f):
                if i == 0: continue
                xid, q1, q2, label = line.strip().split('\t')
                guid = 'paws-{}-{}'.format(s, xid)
                examples.append(PairExample(
                        guid, ['{}_a'.format(guid), '{}_b'.format(guid)],
                        [q1, q2], int(label)))
    return examples
