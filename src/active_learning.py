import argparse
import json
import os
import sys
import shutil
import util
import random
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, LoggingHandler, models
from sentence_transformers.readers import InputExample
import logging
from bisect import bisect_left

import evaluation
from sentence_embedder import SentenceEmbedder
#from image_embedder import ImageEmbedder
import get_close_pairs
import worlds

np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


class ActiveLearningAgent:
    def __init__(self, world, output_dir, smaller_train=None, uncertainty_sampling=True, faiss_nprobe=10):
        self.world = world
        self.symmetric = world.symmetric

        self.id_to_text = world.GetIdToText()
        self.trained_model = None
        
        self.train_primary_ids = list(world.GetTrainPrimary())
        if smaller_train:
            self.train_primary_ids = self.train_primary_ids[:smaller_train]
        self.train_primary_texts = [self.id_to_text[tpid] for tpid in self.train_primary_ids]
        self.train_primary_embeddings = None

        self.dev_primary_ids = list(world.GetDevPrimary())
        self.dev_primary_texts = [self.id_to_text[dpid] for dpid in self.dev_primary_ids]

        self.test_primary_ids = list(world.GetTestPrimary())
        self.test_primary_texts = [self.id_to_text[tpid] for tpid in self.test_primary_ids]
 
        if not self.symmetric:
            self.train_secondary_ids = list(world.GetTrainSecondary())
            if smaller_train:
                self.train_secondary_ids = self.train_secondary_ids[:smaller_train]
            self.train_secondary_texts = [self.id_to_text[tsid] for tsid in self.train_secondary_ids]
            self.train_secondary_embeddings = None

            self.dev_secondary_ids = list(world.GetDevSecondary())
            self.dev_secondary_texts = [self.id_to_text[dsid] for dsid in self.dev_secondary_ids]

            self.test_secondary_ids = list(world.GetTestSecondary())
            self.test_secondary_texts = [self.id_to_text[tsid] for tsid in self.test_secondary_ids]

        self.labeled_set = []
        self.labels = []
        self.labeled_set_weights = []

        self.output_dir = output_dir
        self.faiss_nprobe = faiss_nprobe


    def NormalizeEmbeddings(self):
        for i in range(self.train_primary_embeddings.shape[0]):
            self.train_primary_embeddings[i,:] /= np.linalg.norm( self.train_primary_embeddings[i,:] )

        if not self.symmetric:
            for i in range(self.train_secondary_embeddings.shape[0]):
                self.train_secondary_embeddings[i,:] /= np.linalg.norm( self.train_secondary_embeddings[i,:] )
            
    def GetInitialEmbedding(self, encode_batch_size):
        if self.world.tag == 'inat' or self.world.tag == 'celeba':
            image_embedder = ImageEmbedder(self.world.tag, None)
            image_embedder.init_model()
            logging.info("Getting initial embedding")
            logging.info("Primary:")
            self.train_primary_embeddings = image_embedder.embed(self.train_primary_texts)
            if not self.symmetric:
                logging.info("Secondary:")
                self.train_secondary_embeddings = image_embedder.embed(self.train_secondary_texts)
        else:
            word_embedding_model = models.BERT('bert-base-uncased')
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            
            logging.info("Getting initial embedding")
            logging.info("Primary:")
            self.train_primary_embeddings = np.array(model.encode(self.train_primary_texts, batch_size=encode_batch_size))
            if not self.symmetric:
                logging.info("Secondary:")
                self.train_secondary_embeddings = np.array(model.encode(self.train_secondary_texts, batch_size=encode_batch_size))

        self.NormalizeEmbeddings()

        return
    
    def GetDevEmbedding(self, test=False):
        """Only used by WriteDPRCP."""
        if test:
            primary_texts = self.test_primary_texts
            if not self.symmetric:
                secondary_texts = self.test_secondary_texts
        else:
            primary_texts = self.dev_primary_texts
            if not self.symmetric:
                secondary_texts = self.dev_secondary_texts

        if self.world.tag == 'inat' or self.world.tag == 'celeba':
            image_embedder = ImageEmbedder(self.world.tag, None)
            image_embedder.init_model()
            logging.info("Getting {} embedding".format('test' if test else 'dev'))
            logging.info("Primary:")
            primary_embs = image_embedder.embed(primary_texts)
            if not self.symmetric:
                logging.info("Secondary:")
                secondary_embs = image_embedder.embed(secondary_texts)
        else:
            word_embedding_model = models.BERT('bert-base-uncased')
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            
            logging.info("Getting {} embedding".format('test' if test else 'dev'))
            logging.info("Primary:")
            primary_embs = np.array(model.encode(primary_texts))
            if not self.symmetric:
                logging.info("Secondary:")
                secondary_embs = np.array(model.encode(secondary_texts))

        # Normalize
        for i in range(primary_embs.shape[0]):
            primary_embs[i,:] /= np.linalg.norm(primary_embs[i,:] )

        if not self.symmetric:
            for i in range(secondary_embs.shape[0]):
                secondary_embs[i,:] /= np.linalg.norm( secondary_embs[i,:] )

        if test:
            self.test_primary_embeddings = primary_embs
            if not self.symmetric:
                self.test_secondary_embeddings = secondary_embs
        else:
            self.dev_primary_embeddings = primary_embs
            if not self.symmetric:
                self.dev_secondary_embeddings = secondary_embs

    def AddOraclePositives(self, num):
        positive_pairs = self.world.GetPositives(num)
        for pair in positive_pairs:
            self.labeled_set.append(pair)
            self.labels.append(1)
            self.labeled_set_weights.append(1)

        return

    def AddStatedNegatives(self, num):
        negative_pairs = self.world.GetStatedNegatives(num)
        for pair in negative_pairs:
            self.labeled_set.append(pair)
            self.labels.append(0)
            self.labeled_set_weights.append(1)

    def AddRandomStated(self, num):
        train_positives = self.world.GetPositives(self.world.GetNumTrainPositives())
        stated_negatives = self.world.GetStatedNegatives(self.world.GetNumTrainStatedNegatives())
        all_stated = sorted([(x, 1) for x in train_positives] + [(x, 0) for x in stated_negatives])
        sample = np.random.choice(len(all_stated), num, replace=False)
        for i in sample:
            pair, label = all_stated[i]
            self.labeled_set.append(pair)
            self.labels.append(label)
            self.labeled_set_weights.append(1)
        
    def AddRandomNegatives(self, num):
        already_labeled = set(self.labeled_set)
        if self.symmetric:
            for pair in self.labeled_set:
                already_labeled.add( (pair[1], pair[0]) )

        for _ in range(num):
            while True:
                if self.symmetric:
                    id1 = random.choice(self.train_primary_ids)
                    id2 = random.choice(self.train_primary_ids)
                    if id1 != id2 and (id1,id2) not in already_labeled:
                        new_pair = (id1, id2)
                        break
                else:
                    id1 = random.choice(self.train_primary_ids)
                    id2 = random.choice(self.train_secondary_ids)
                    if (id1, id2) not in already_labeled:
                        new_pair = (id1, id2)
                        break

            already_labeled.add(new_pair)
            self.labeled_set.append(new_pair)
            self.labels.append(0)
            self.labeled_set_weights.append(1)

        return

        
    def AddRandomExamples(self, num):
        already_labeled = set(self.labeled_set)
        if self.symmetric:
            for pair in self.labeled_set:
                already_labeled.add( (pair[1], pair[0]) )

        for _ in range(num):
            while True:
                if self.symmetric:
                    id1 = random.choice(self.train_primary_ids)
                    id2 = random.choice(self.train_primary_ids)
                    if id1 != id2 and (id1,id2) not in already_labeled:
                        new_pair = (id1, id2)
                        break
                else:
                    id1 = random.choice(self.train_primary_ids)
                    id2 = random.choice(self.train_secondary_ids)
                    if (id1, id2) not in already_labeled:
                        new_pair = (id1, id2)
                        break

            already_labeled.add(new_pair)
            self.labeled_set.append(new_pair)
            self.labels.append(self.world.Query(new_pair))
            self.labeled_set_weights.append(1)

        return

    def RemoveAlreadyLabeled(self, cosine_pairs):
        already_labeled = set(self.labeled_set)
        if self.symmetric:
            for pair in self.labeled_set:
                already_labeled.add( (pair[1], pair[0]) )
        
        return [cp for cp in cosine_pairs if cp[1] not in already_labeled]

    def CheckCalibration(self, primary_embeddings, secondary_embeddings, split, close_pairs, 
                         num_bins=20, num_random_negatives=1e6):
        calibration_data = dict()  # Convention: id1 < id2 if symmetric
        def add_pair(id1, id2, label):
            if self.symmetric:
                if id1 == id2: return False # Don't do anything if id1 == id2
                key = (min(id1, id2), max(id1, id2))
            else:
                key = (id1, id2)
            if key in calibration_data:
                return False
            else:
                calibration_data[key] = label
                return True

        # Split-specific things
        if split == 'train':
            num_pairs = self.world.GetTotalNumTrainPairs()
            positives_id = self.world.GetTrainPositives()
            primary_ids = self.train_primary_ids
            if self.symmetric:
                secondary_ids = primary_ids
            else:
                secondary_ids = self.train_secondary_ids
        elif split == 'dev':
            num_pairs = self.world.GetTotalNumDevPairs()
            positives_id = self.world.GetDevPositives()
            primary_ids = self.dev_primary_ids
            if self.symmetric:
                secondary_ids = primary_ids
            else:
                secondary_ids = self.dev_secondary_ids
        else:
            raise ValueError(split)
        primary_id_to_index = {uid: i for i, uid in enumerate(primary_ids)}
        if self.symmetric:
            secondary_id_to_index = primary_id_to_index
        else:
            secondary_id_to_index = {uid: i for i, uid in enumerate(secondary_ids)}
        positives = [(primary_id_to_index[id1], secondary_id_to_index[id2])
                     for id1, id2 in positives_id]

        # Add positives and close negatives
        for ind1, ind2 in positives:
            add_pair(ind1, ind2, 'positive')
        for ind1, ind2 in close_pairs:
            add_pair(ind1, ind2, 'close')
        
        # Get random negatives
        random_weight = (num_pairs - len(calibration_data)) / num_random_negatives
        cur_rand_neg = 0
        while cur_rand_neg <  num_random_negatives:
            ind1 = random.randrange(len(primary_ids))
            ind2 = random.randrange(len(secondary_ids))
            if add_pair(ind1, ind2, 'random'):
                cur_rand_neg += 1

        # Bin by p(Y|X) = sigmoid(logit)
        bin_totals = [0] * num_bins
        bin_positives = [0] * num_bins
        bin_centers = [(1 + 2 * i) / (2 * num_bins) for i in range(num_bins)]
        for (ind1, ind2), label in calibration_data.items():
            embedding1 = primary_embeddings[ind1]
            embedding2 = secondary_embeddings[ind2]
            cosine = np.dot(embedding1, embedding2)
            logit = self.trained_model.get_logit(cosine)
            pred_p_y = 1 / (1 + np.exp(-logit))
            bin_index = int(num_bins * pred_p_y)
            if bin_index == num_bins:  #  Numerical precision
                bin_index = num_bins - 1
            if label == 'random':
                bin_totals[bin_index] += random_weight
            elif label == 'close':
                bin_totals[bin_index] += 1
            elif label == 'positive':
                bin_totals[bin_index] += 1
                bin_positives[bin_index] += 1
            else:
                raise ValueError(label)

        print('Calibration plot data:')
        print('bin\tmodel_prob\ttrue_prob\tnum_pos\tnum_total\tfrac_total')
        total_weight = sum(bin_totals)
        squared_calib_error = 0.0
        for i in range(num_bins):
            if bin_totals[i] == 0:
                true_prob = 0
            else:
                true_prob = bin_positives[i] / bin_totals[i]
            frac_total = bin_totals[i] / total_weight
            squared_calib_error += frac_total * (true_prob - bin_centers[i])**2
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(i, bin_centers[i], true_prob, bin_positives[i], bin_totals[i], frac_total))
        calib_error = np.sqrt(squared_calib_error)
        print('Calibration error (L2): {}'.format(calib_error))


    def GetBatchCertainty(self, cosine_pairs, batch_size):
        logging.info('Initial cosine_pairs:       {}'.format(len(cosine_pairs)))
        unlabeled_cosine_pairs = self.RemoveAlreadyLabeled(cosine_pairs)
        logging.info('After RemoveAlreadyLabeled: {}'.format(len(unlabeled_cosine_pairs)))

        unlabeled_cosine_pairs.sort(reverse=True)

        most_certain = unlabeled_cosine_pairs[:batch_size]

        return [mc[1] for mc in most_certain]
        

    def GetBatchUncertainty(self, cosine_pairs, batch_size):
        unlabeled_cosine_pairs = self.RemoveAlreadyLabeled(cosine_pairs)

        cosines, pairs = zip(*unlabeled_cosine_pairs)
        scores = np.abs(self.trained_model.get_logit(cosines))
        unlabeled_score_pairs = list(zip(scores,pairs))

        unlabeled_score_pairs.sort()

        most_uncertain = unlabeled_score_pairs[:batch_size]

        return [mu[1] for mu in most_uncertain]


    def QueryCertain(self, batch_size, sampling_technique, distance_metric):
        new_labels = 0
        
        if self.trained_model is not None:
            logging.info("Embedding training set")        
            logging.info("Primary:")
            self.train_primary_embeddings = np.array(self.trained_model.embed(self.train_primary_texts))
            if not self.symmetric:
                logging.info("Secondary:")
                self.train_secondary_embeddings = np.array(self.trained_model.embed(self.train_secondary_texts))
        
            if distance_metric == 'cosine':
                self.NormalizeEmbeddings()

        if sampling_technique == "oracle":
            if self.symmetric:
                oracle_positive_cosines = self.world.GetPositiveCosines(self.train_primary_ids, self.train_primary_embeddings)
            else:
                assert(False) #not implemented yet


        logging.info("Getting Candidates")
        if self.symmetric:
            pairs = get_close_pairs.faiss_get_pairs(self.train_primary_embeddings, self.train_primary_embeddings, self.world.faiss_nlist, self.faiss_nprobe)
        else:
            pairs = get_close_pairs.faiss_get_pairs(self.train_primary_embeddings, self.train_secondary_embeddings, self.world.faiss_nlist, self.faiss_nprobe)

        if self.trained_model:
            logging.info("Checking calibration on entire train set")
            self.CheckCalibration(
                    self.train_primary_embeddings,
                    self.train_primary_embeddings if self.symmetric else self.train_secondary_embeddings,
                    'train', pairs)

        logging.info("Filtering Candidates")
        cosine_faiss_pairs = []
        faiss_pairs = set()

        for pair in pairs:
            id1 = self.train_primary_ids[pair[0]]
            if self.symmetric:
                id2 = self.train_primary_ids[pair[1]]
            else:
                id2 = self.train_secondary_ids[pair[1]]
            
            if id1 != id2:
                embedding1 = self.train_primary_embeddings[pair[0]]
                if self.symmetric:
                    embedding2 = self.train_primary_embeddings[pair[1]]
                else:
                    embedding2 = self.train_secondary_embeddings[pair[1]]

                cosine = np.dot( embedding1, embedding2 )

                cosine_faiss_pairs.append( (cosine, (id1, id2) ) )
                faiss_pairs.add( (id1, id2) )

        logging.info("Choosing new examples")
                
        if sampling_technique == "certainty":
            pairs_to_label = self.GetBatchCertainty(cosine_faiss_pairs, batch_size)
            weights = [1 for i in range(len(pairs_to_label))]
        elif sampling_technique == "uncertainty":
            pairs_to_label = self.GetBatchUncertainty(cosine_faiss_pairs, batch_size)
            weights = [1 for i in range(len(pairs_to_label))]
        else:
            raise ValueError(sampling_technique)
        
        assert(len(pairs_to_label) == batch_size)

        for pair in pairs_to_label:
            self.labeled_set.append(pair)
            self.labels.append( self.world.Query(pair) )
        
        for weight in weights:
            self.labeled_set_weights.append(weight)


        new_batch_labels = self.labels[-batch_size:]
        print("Num positive selected: {}".format(sum(new_batch_labels)))
        print("Num total selected: {}".format(batch_size))

        # Dump collected data to file
        data_dump_file = os.path.join(self.output_dir, 'collected_data_{}.tsv'.format(len(self.labels)))
        print('Writing data to {}'.format(data_dump_file))
        with open(data_dump_file, 'w') as f:
            for (qid1, qid2), label in zip(self.labeled_set, self.labels):
                print('{}\t{}\t{}'.format(qid1, qid2, label), file=f)


    def Train(self, args, is_last=False):
        num_labels = len(self.labels)
        assert(len(self.labeled_set) == len(self.labels))
       
        logging.info("Converting Training Data") 
        input_examples = []
        for i in range(len(self.labels)):
            pair = self.labeled_set[i]
            label = self.labels[i]
            weight = self.labeled_set_weights[i]
            guid = "{}-{}".format(pair[0],pair[1])
            id1 = pair[0]
            id2 = pair[1]
            if self.world.tag == 'inat' or self.world.tag == 'celeba':  # images
                ex = (id1, id2, label)
            else:  # text
                text1 = self.id_to_text[pair[0]]
                text2 = self.id_to_text[pair[1]]
                ex = util.PairExample(guid,[id1,id2],[text1,text2],label)
            for j in range(weight):
                input_examples.append(ex)

        logging.info("Training")
        if self.world.tag == 'inat' or self.world.tag == 'celeba':  # images
            self.trained_model = ImageEmbedder(
                    self.world.tag,
                    "{}/labels{}".format(self.output_dir, num_labels),
                    min_epochs=args.min_epochs, 
                    min_train_updates=args.min_train_updates,
                    #batch_size=args.batch_size,
                    #learning_rate=args.learning_rate,
                    dev_frac=args.dev_frac,
                    loss_type=args.loss_type,
                    distance_metric=args.distance_metric,
                    logit_scale=args.logit_scale,
                    #weight_decay=args.weight_decay,
                    dont_save_model=args.dont_save_model or (not is_last and args.save_last_model))
            self.trained_model.train(input_examples)
        else:
            self.trained_model = SentenceEmbedder(
                    "{}/labels{}".format(self.output_dir, num_labels),
                    min_epochs=args.min_epochs, 
                    min_train_updates=args.min_train_updates,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    dev_frac=args.dev_frac,
                    pooling_mode=args.pooling_mode,
                    loss_type=args.loss_type,
                    distance_metric=args.distance_metric,
                    logit_scale=args.logit_scale,
                    bias_scale=args.bias_scale,
                    weight_decay=args.weight_decay,
                    dont_save_model=args.dont_save_model or (not is_last and args.save_last_model))
            self.trained_model.train(input_examples)

    def Evaluate(self, results_dir, distance_metric, test=False):
        assert(self.trained_model is not None)
        split = 'test' if test else 'dev'
            
        dev_ids_to_embed = set()
        
        dprcp_pairs, dprcp_tags = self.world.GetDPRCP(test=test)

        for pair in dprcp_pairs:
            dev_ids_to_embed.add(pair[0])
            dev_ids_to_embed.add(pair[1])
        dev_ids_to_embed = list(dev_ids_to_embed)

        text_to_embed = [self.id_to_text[idd] for idd in dev_ids_to_embed]
        
        logging.info("Embedding {} set".format(split))
        dev_embeddings = self.trained_model.embed(text_to_embed)


        dev_id_to_normalized_embedding = {}
        for i in range(len(dev_embeddings)):
            if distance_metric == 'cosine':
                dev_id_to_normalized_embedding[dev_ids_to_embed[i]] = dev_embeddings[i] / np.linalg.norm(dev_embeddings[i])
            else:
                dev_id_to_normalized_embedding[dev_ids_to_embed[i]] = dev_embeddings[i]

        logging.info("Writing {}PRCP...".format('T' if test else 'D'))

        score_tags = []
        for i in tqdm(range(len(dprcp_pairs)), desc='Writing PRCP'):
            pair = dprcp_pairs[i]
            cosine = np.dot(dev_id_to_normalized_embedding[pair[0]],
                            dev_id_to_normalized_embedding[pair[1]])
            logit = self.trained_model.get_logit(cosine)
            score_tags.append((logit, dprcp_tags[i]))
        scores_np = np.array([s for s, t in score_tags], dtype=np.float32)
        np.save(os.path.join(results_dir, 'pred_{}_pr_curve.npy'.format(split)), scores_np)

        pr_results = evaluation.evaluate_precision_recall(score_tags, self.world, split)
        logging.info('PR curve {} results: {}'.format(split, json.dumps(pr_results)))
        with open(os.path.join(results_dir, 'results_pr.json'), 'w') as f:
            json.dump(pr_results, f, indent=2)

        return


    def WriteDPRCP(self, num_random_negatives, test=False):
        if test:
            pos_pairs = set(self.world.GetTestPositives()) 
        else:
            pos_pairs = set(self.world.GetDevPositives()) 
        
        self.GetDevEmbedding(test=test)
        if test:
            primary_ids = self.test_primary_ids
            primary_embs = self.test_primary_embeddings
            if not self.symmetric:
                secondary_ids = self.test_secondary_ids
                secondary_embs = self.test_secondary_embeddings
        else:
            primary_ids = self.dev_primary_ids
            primary_embs = self.dev_primary_embeddings
            if not self.symmetric:
                secondary_ids = self.dev_secondary_ids
                secondary_embs = self.dev_secondary_embeddings

        if self.symmetric:
            faiss_pairs = get_close_pairs.faiss_get_pairs(primary_embs, primary_embs, self.world.faiss_nlist, self.faiss_nprobe)
            faiss_pairs = [(primary_ids[pair[0]], primary_ids[pair[1]]) for pair in faiss_pairs]
            secondary_ids = primary_ids
        else:
            faiss_pairs = get_close_pairs.faiss_get_pairs(primary_embs, secondary_embs, self.world.faiss_nlist, self.faiss_nprobe)
            faiss_pairs = [(primary_ids[pair[0]], secondary_ids[pair[1]]) for pair in faiss_pairs]
        faiss_pairs = set([pair for pair in faiss_pairs if (pair not in pos_pairs and (pair[1],pair[0]) not in pos_pairs and pair[0] != pair[1])])
        
        off_limits = pos_pairs.union(faiss_pairs)
        neg_pairs = set()

        random.seed(0)
        for i in range(num_random_negatives):
            while True:
                q1 = random.choice(primary_ids)
                q2 = random.choice(secondary_ids)
                if q1 !=  q2 and (q1,q2) not in off_limits and (q2,q1) not in off_limits:
                    break
            off_limits.add( (q1,q2) )
            neg_pairs.add( (q1,q2) )
             
        
        pair_tags = [(pair,"random") for pair in neg_pairs] + [(pair,"faiss") for pair in faiss_pairs] + [(pair,"positive") for pair in pos_pairs]
        random.seed(0)
        random.shuffle(pair_tags)

        if test:
            out_file = "{}/{}_tprcp.tsv".format(worlds.DATA_DIR,self.world.tag)
        else:
            out_file = "{}/{}_dprcp.tsv".format(worlds.DATA_DIR,self.world.tag)
        with open(out_file, "w") as f:
            for pair, tag in pair_tags:
                f.write("{}\t{}\t{}\n".format(pair[0], pair[1], tag))

        return

def main(args):
    # Seed RNG
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    world = worlds.get_world(args.world)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    agent = ActiveLearningAgent(world, args.output_dir, smaller_train=args.smaller_train, faiss_nprobe=10)


    if args.al_scaling == 'exponential':
        batch_sizes = [args.al_init_batch // 2**i * 3**i for i in range(args.al_batches)]
    elif args.al_scaling == 'linear':
        batch_sizes = [args.al_init_batch * (i+1) for i in range(args.al_batches)]
    elif args.al_scaling == 'constant':
        batch_sizes = [args.al_init_batch] * args.al_batches
    else:
        raise ValueError(args.al_scaling)


    num_labels = 0
    # Can use 4 * args.batch_size--main batch size is for pair + backward pass
    agent.GetInitialEmbedding(4 * args.batch_size)
    logging.info('Allocated after GetInitialEmbedding: {} MiB'.format(torch.cuda.memory_allocated() / 2**20))
    for i, batch_size in enumerate(batch_sizes):
        if num_labels == 0 and args.seed_with_stated:
            agent.AddRandomStated(batch_size)
            data_dump_file = os.path.join(agent.output_dir, 'collected_data_{}.tsv'.format(len(agent.labels)))
            print('Writing data to {}'.format(data_dump_file))
            with open(data_dump_file, 'w') as f:
                for (qid1, qid2), label in zip(agent.labeled_set, agent.labels):
                    print('{}\t{}\t{}'.format(qid1, qid2, label), file=f)
        elif num_labels == 0 and args.sampling_technique == 'uncertainty':
            agent.QueryCertain(batch_size, 'certainty', args.distance_metric)
        else:
            agent.QueryCertain(batch_size,args.sampling_technique, args.distance_metric)
        agent.Train(args, is_last=(i == len(batch_sizes) - 1))
        num_labels += batch_size
        logging.info('Allocated after Train: {} MiB'.format(torch.cuda.memory_allocated() / 2**20))
        if not args.skip_pr_eval:
            agent.Evaluate(os.path.join(args.output_dir, 'labels{}'.format(num_labels)),
                           args.distance_metric)
            logging.info('Allocated after Evaluate: {} MiB'.format(torch.cuda.memory_allocated() / 2**20))

def get_parser():
    parser = argparse.ArgumentParser()
    # Required arguemnts
    parser.add_argument('--world', required=True, choices=['qqp', 'wikiqa', 'inat', 'celeba'])
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sampling_technique', required=True, choices = ['certainty', 'uncertainty'])

    # Active learning batch sizes
    parser.add_argument('--al_scaling', choices=['exponential', 'linear', 'constant'], 
                        default='exponential')
    parser.add_argument('--al_init_batch', type=int, default=2048)
    parser.add_argument('--al_batches', type=int, default=10)

    # SentenceEmbedder architecture
    parser.add_argument('--distance_metric', default='cosine', choices=['cosine', 'l2', 'dot'])
    parser.add_argument('--pooling_mode', default='mean', choices=['mean', 'cls', 'max'])

    # SentenceEmbedder training
    parser.add_argument('--min_epochs', type=int, default=2)
    parser.add_argument('--min_train_updates', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--loss_type', default='sgdbn', choices=['logistic', 'sgdbn'])
    parser.add_argument('--logit_scale', type=float, default=1e4)
    parser.add_argument('--bias_scale', type=float, default=1e2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dont_save_model', action='store_true')
    parser.add_argument('--save_last_model', action='store_true')

    # Data
    parser.add_argument('--dev_frac', type=float, default=0.0, help='Use this fraction of data as dev in each iteration of active learning')
    parser.add_argument('--smaller_train', type=int, default=None)
    parser.add_argument('--seed_with_stated', action='store_true', help='Seed with stated data instead of static retrieval')

    # Other
    parser.add_argument('--skip_pr_eval', action='store_true')
    parser.add_argument('--rng_seed', type=int, default=0)
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
