import collections
import csv
import numpy as np
import os

#import celeba_reader
#import inat_reader
import reader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


class SymmetricWorld(object):
    def __init__(self):
        self.num_queries = 0
        self.symmetric = True
        self.tag = None

    def GetIdToText(self):
        return self.id_to_text
    
    def GetTotalNumTrainPairs(self):
        num_train_nodes = len(self.train_nodes)

        return num_train_nodes*(num_train_nodes-1) // 2

    def GetTotalNumDevPairs(self):
        num_dev_nodes = len(self.dev_nodes)
        
        return num_dev_nodes*(num_dev_nodes-1) // 2

    def GetTotalNumTestPairs(self):
        num_test_nodes = len(self.test_nodes)
        
        return num_test_nodes*(num_test_nodes-1) // 2

    def Query(self,pair):
        assert(pair[0] in self.train_nodes and pair[1] in self.train_nodes)
        self.num_queries += 1

        if pair in self.positive_pairs or (pair[1],pair[0]) in self.positive_pairs:
            label=1
        else:
            label=0

        return label

    def GetTrainPrimary(self):
        return self.train_nodes

    def GetDevPrimary(self):
        return self.dev_nodes

    def GetTestPrimary(self):
        return self.test_nodes

    def GetPositiveCosines(self, ids, embeddings):
        assert(len(ids) == embeddings.shape[0])
        id_to_index = {}
        for i in range(len(ids)):
            id_to_index[ids[i]] = i

        train_positives = sorted([pair for pair in self.positive_pairs if pair[0] in self.train_nodes and pair[1] in self.train_nodes])

        cosines = []
        for tp in train_positives:
            embedding0 = embeddings[id_to_index[tp[0]],:]
            embedding1 = embeddings[id_to_index[tp[1]],:]
            cosines.append( np.dot(embedding0, embedding1) )
        
        return cosines

    def GetTrainPositives(self):
        train_positives = [pair for pair in self.positive_pairs if pair[0] in self.train_nodes and pair[1] in self.train_nodes]
        return sorted(train_positives)

    def GetDevPositives(self):
        return [pair for pair in self.positive_pairs if pair[0] in self.dev_nodes and pair[1] in self.dev_nodes]

    def GetTestPositives(self):
        return [pair for pair in self.positive_pairs if pair[0] in self.test_nodes and pair[1] in self.test_nodes]

    def GetPositives(self,num):
        train_positives = self.GetTrainPositives()
        positive_sample = np.random.choice(len(train_positives), num, replace=False)
        return [train_positives[i] for i in positive_sample]

    def GetNumTrainPositives(self):
        train_positives = [pair for pair in self.positive_pairs if pair[0] in self.train_nodes and pair[1] in self.train_nodes]
        return len(train_positives)

    def GetDPRCP(self, test=False):
        dprcp_pairs = []
        dprcp_tags = [] 
        if test:
            filename = "{}/{}_tprcp.tsv".format(DATA_DIR, self.tag)
        else:
            filename = "{}/{}_dprcp.tsv".format(DATA_DIR, self.tag)

        with open(filename, 'r') as f:
            for line in f.readlines():
                p1, p2, tag = line[:-1].split('\t')
                dprcp_pairs.append( (p1, p2) )
                dprcp_tags.append( tag )

        return dprcp_pairs, dprcp_tags

class QQPWorld(SymmetricWorld):

    def __init__(self):
        super(QQPWorld, self).__init__()
        self.positive_pairs = reader.ReadPairs("{}/qqp_transitive_closure_pos_pairs.tsv".format(DATA_DIR))
        self.stated_negative_pairs = reader.ReadPairs("{}/qqp_stated_neg_pairs.tsv".format(DATA_DIR))
        self.id_to_text = reader.ReadQuestions("{}/qqp_questions.tsv".format(DATA_DIR))
        self.train_nodes = reader.ReadNodes("{}/qqp_train.tsv".format(DATA_DIR))
        self.dev_nodes = reader.ReadNodes("{}/qqp_dev.tsv".format(DATA_DIR))
        self.test_nodes = reader.ReadNodes("{}/qqp_test.tsv".format(DATA_DIR))
        self.faiss_nlist = 1000
        self.tag = "qqp"
        return

    def GetStatedNegatives(self, num):
        train_negatives = sorted([
            pair for pair in self.stated_negative_pairs
            if pair[0] in self.train_nodes and pair[1] in self.train_nodes])
        negative_sample = np.random.choice(len(train_negatives), num, replace=False)
        return [train_negatives[i] for i in negative_sample]

    def GetNumTrainStatedNegatives(self):
        train_negatives = [pair for pair in self.stated_negative_pairs 
                           if pair[0] in self.train_nodes and pair[1] in self.train_nodes]
        return len(train_negatives)

class INatWorld(SymmetricWorld):
    def __init__(self):
        super(INatWorld, self).__init__()
        self.positive_pairs = inat_reader.ReadPairs(inat_reader.POS_PAIRS_FILE)
        self.train_nodes = inat_reader.ReadNodes(os.path.join(inat_reader.DATA_DIR, 'inat_train.tsv'))
        self.dev_nodes = inat_reader.ReadNodes(os.path.join(inat_reader.DATA_DIR, 'inat_dev.tsv'))
        self.test_nodes = inat_reader.ReadNodes(os.path.join(inat_reader.DATA_DIR, 'inat_test.tsv'))
        # Just set id_to_text to be identity mapping, since loader handles actual image loading
        self.id_to_text = {}
        for img_id in self.train_nodes:
            self.id_to_text[img_id] = img_id
        for img_id in self.dev_nodes:
            self.id_to_text[img_id] = img_id
        for img_id in self.test_nodes:
            self.id_to_text[img_id] = img_id
        self.tag = 'inat'
        self.queries = 0
        self.faiss_nlist = 1000

class CelebAWorld(SymmetricWorld):
    def __init__(self):
        super(CelebAWorld, self).__init__()
        self.positive_pairs = celeba_reader.ReadPairs(celeba_reader.POS_PAIRS_FILE)
        self.train_nodes = celeba_reader.ReadNodes(os.path.join(celeba_reader.DATA_DIR, 'celeba_train.tsv'))
        self.dev_nodes = celeba_reader.ReadNodes(os.path.join(celeba_reader.DATA_DIR, 'celeba_dev.tsv'))
        self.test_nodes = celeba_reader.ReadNodes(os.path.join(celeba_reader.DATA_DIR, 'celeba_test.tsv'))
        # Just set id_to_text to be identity mapping, since loader handles actual image loading
        self.id_to_text = {}
        for img_id in self.train_nodes:
            self.id_to_text[img_id] = img_id
        for img_id in self.dev_nodes:
            self.id_to_text[img_id] = img_id
        for img_id in self.test_nodes:
            self.id_to_text[img_id] = img_id
        self.tag = 'celeba'
        self.queries = 0
        self.faiss_nlist = 1000
        
class AsymmetricWorld(object):
    # Convention: Asymmetric world does train/dev splits on primary ID, 
    #             uses same secondary ID set for all splits
    def __init__(self):
        self.num_queries = 0
        self.symmetric = False

    def Query(self,pair):
        assert(pair[0] in self.train_primary_ids and pair[1] in self.secondary_ids)
        self.num_queries += 1

        if pair in self.positive_pairs:
            label=1
        else:
            label=0

        return label
    
    def GetTrainPrimary(self):
        return self.train_primary_ids

    def GetTrainSecondary(self):
        return self.secondary_ids

    def GetDevPrimary(self):
        return self.dev_primary_ids

    def GetDevSecondary(self):
        return self.secondary_ids

    def GetTestPrimary(self):
        return self.test_primary_ids

    def GetTestSecondary(self):
        return self.secondary_ids

    def GetIdToText(self):
        return self.id_to_text

    def GetTotalNumTrainPairs(self):
        return len(self.train_primary_ids) * len(self.secondary_ids)

    def GetTotalNumDevPairs(self):
        return len(self.dev_primary_ids) * len(self.secondary_ids)

    def GetTotalNumTestPairs(self):
        return len(self.test_primary_ids) * len(self.secondary_ids)


    def GetPositives(self,num):
        train_positives = [pair for pair in self.positive_pairs if pair[0] in self.train_primary_ids and pair[1] in self.secondary_ids]
        positive_sample = np.random.choice(len(train_positives), num, replace=False)
        return [train_positives[i] for i in positive_sample]

    def GetTrainPositives(self):
        train_positives = [pair for pair in self.positive_pairs if pair[0] in self.train_primary_ids and pair[1] in self.secondary_ids]
        return sorted(train_positives)

    def GetNumTrainPositives(self):
        train_positives = [pair for pair in self.positive_pairs if pair[0] in self.train_primary_ids and pair[1] in self.secondary_ids]
        return len(train_positives)

    def GetDevPositives(self):
        return [pair for pair in self.positive_pairs if pair[0] in self.dev_primary_ids and pair[1] in self.secondary_ids]

    def GetTestPositives(self):
        return [pair for pair in self.positive_pairs if pair[0] in self.test_primary_ids and pair[1] in self.secondary_ids]

    def GetDPRCP(self, test=False):
        dprcp_pairs = []
        dprcp_tags = [] 
        if test:
            filename = "{}/{}_tprcp.tsv".format(DATA_DIR, self.tag)
        else:
            filename = "{}/{}_dprcp.tsv".format(DATA_DIR, self.tag)

        with open(filename, 'r') as f:
            for line in f.readlines():
                p1, p2, tag = line[:-1].split('\t')
                dprcp_pairs.append( (p1, p2) )
                dprcp_tags.append( tag )

        return dprcp_pairs, dprcp_tags

WikiQAExample = collections.namedtuple('WikiQAExample', ['qid', 'sid', 'question', 'sentence', 'label'])

def read_wikiqa(filename):
    data = []
    with open(filename) as f: 
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for r in reader:
            sentence = '{} . {}'.format(r['DocumentTitle'], r['Sentence'])
            data.append(WikiQAExample(r['QuestionID'], r['SentenceID'], r['Question'], sentence, int(r['Label'])))
    return data
        
class WikiQAWorld(AsymmetricWorld):
    def __init__(self):
        super(WikiQAWorld, self).__init__()
        self.tag = 'wikiqa'
        self.faiss_nlist = 100
        self.train_data = read_wikiqa(os.path.join(DATA_DIR, 'wikiqa', 'WikiQA-train.tsv'))
        self.dev_data = read_wikiqa(os.path.join(DATA_DIR, 'wikiqa', 'WikiQA-dev.tsv'))
        self.test_data = read_wikiqa(os.path.join(DATA_DIR, 'wikiqa', 'WikiQA-test.tsv'))
        self.all_data = read_wikiqa(os.path.join(DATA_DIR, 'wikiqa', 'WikiQA.tsv'))
        self.positive_pairs = set((x.qid, x.sid) for x in self.all_data if x.label == 1)
        self.stated_negative_pairs = set((x.qid, x.sid) for x in self.all_data if x.label == 0)
        self.id_to_text = {}

        # Use sentences from all splits
        for x in self.all_data:  
            self.id_to_text[x.qid] = x.question
            self.id_to_text[x.sid] = x.sentence
        self.secondary_ids = set(x.sid for x in self.all_data)

        # Use the train/dev split for questions
        self.train_primary_ids = set(x.qid for x in self.train_data)
        self.dev_primary_ids = set(x.qid for x in self.dev_data)
        self.test_primary_ids = set(x.qid for x in self.test_data)

    def GetStatedNegatives(self, num):
        train_negatives = sorted([
            pair for pair in self.stated_negative_pairs
            if pair[0] in self.train_primary_ids])
        negative_sample = np.random.choice(len(train_negatives), num, replace=False)
        return [train_negatives[i] for i in negative_sample]

    def GetNumTrainStatedNegatives(self):
        train_negatives = [pair for pair in self.stated_negative_pairs 
                           if pair[0] in self.train_primary_ids]
        return len(train_negatives)


def get_world(name):
    if name == 'qqp': 
        return QQPWorld()
    elif name == 'wikiqa':
        return WikiQAWorld()
    elif name == 'inat':
        return INatWorld()
    elif name == 'celeba':
        return CelebAWorld()
    else:
        raise ValueError(name)
