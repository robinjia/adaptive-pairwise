"""SentenceEmbedder model code."""
from datetime import datetime
import json
import logging
import math
import numpy as np
import os
import random
import shutil
import sys
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_transformers import LoggingHandler, SentenceTransformer, SentencesDataset, models, losses 
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator
from sentence_transformers.util import batch_to_device
from transformers import BertModel

import evaluation
from model_utils import *

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

class NoDropoutBERT(models.BERT):
    """sentence_transformers BERT model but without dropout"""
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, do_lower_case: bool = True,
                 loading=False):
        super(NoDropoutBERT, self).__init__(model_name_or_path, max_seq_length=max_seq_length,
                                            do_lower_case=do_lower_case)
        if not loading:  # Only do this for initializing model, not loading
            self.bert = None
            self.bert = BertModel.from_pretrained(model_name_or_path,
                                                  hidden_dropout_prob=0.0, 
                                                  attention_probs_dropout_prob=0.0)

    @classmethod
    def load(cls, input_path):
        with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
            config = json.load(fIn)
        return cls(model_name_or_path=input_path, loading=True, **config)


def make_evaluator(evaluators):
    if len(evaluators) == 0:
        return None
    elif len(evaluators) == 1:
        return evaluators[0]
    else:
        return SequentialEvaluator(evaluators)

class MonotoneLinearLayer(nn.Module):
    def __init__(self):
        super(MonotoneLinearLayer, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.abs(self.weight) * x + self.bias

class CosineLogisticLoss(nn.Module):
    def __init__(self, model, distance_metric='cosine', logit_scale=1e3):
        super(CosineLogisticLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.logit_scale = logit_scale

        self.classifier = nn.Linear(1, 1, bias=True)  # bias=True is default, but just to emphasize
        nn.init.constant_(self.classifier.weight, 0.001)  # Initialize where high cosine sim = paraphrase
        nn.init.zeros_(self.classifier.bias)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, sentence_features, labels=None, return_both=False):
        reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        rep_a, rep_b = reps  # B, d
        distance = get_distance(rep_a, rep_b, self.distance_metric)
        logits = self.logit_scale * self.classifier(distance.view(-1, 1)).view(-1)  # B,
        if labels is None:
            return logits
        else:
            loss = self.loss_func(logits, labels.float())
            if return_both:
                return logits, loss
            else:
                return loss

class CosineLogisticBNLoss(nn.Module):
    """Like CosineLogisticLoss but with batchnorm"""
    def __init__(self, model, distance_metric='cosine', logit_scale=1e3):
        super(CosineLogisticBNLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.logit_scale = logit_scale

        self.batchnorm = nn.BatchNorm1d(1, momentum=None, eps=1e-8, affine=False)  # No transformation
        self.linear = MonotoneLinearLayer()
        nn.init.constant_(self.linear.weight, 10 / logit_scale)  # Initialize to net weight = 10
        nn.init.constant_(self.linear.bias, -50 / logit_scale)  # Initialize to net weight = -50
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, sentence_features, labels=None, return_both=False):
        reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        rep_a, rep_b = reps  # B, d
        distance = get_distance(rep_a, rep_b, self.distance_metric)
        logits = self.logit_scale * self.linear(self.batchnorm(distance.view(-1, 1)).view(-1)) # B,
        if labels is None:
            return logits
        else:
            loss = self.loss_func(logits, labels.float())
            if return_both:
                return logits, loss
            else:
                return loss

class DistanceEmbedder(nn.Module):
    """Helper module that converts pairs of sentence features to distance.""" 
    def __init__(self, model, distance_metric='cosine'):
        super(DistanceEmbedder, self).__init__()
        self.model = model
        self.distance_metric = distance_metric

    def forward(self, sentence_features):
        reps = [self.model(sf)['sentence_embedding'] for sf in sentence_features]
        rep_a, rep_b = reps  # B, d
        distance = get_distance(rep_a, rep_b, self.distance_metric)
        return distance


class AccuracyEvaluator(SentenceEvaluator):
    """Evaluate accuracy.

    NOTE: this somewhat departs from the SentenceEvaluator implementations
    already in the SentenceTransformer library, because this class needs access
    to the loss function in addition to the sentence embedding model.
    We make this work by passing the entire SentenceEmbedder at initialization.
    """
    def __init__(self, dataset, processed_data, embedder, split):
        self.dataset = dataset
        self.processed_data = processed_data
        self.embedder = embedder
        self.split = split

        self.device = embedder.model.device
        self.dataloader = DataLoader(processed_data, shuffle=False, batch_size=embedder.batch_size)
        self.dataloader.collate_fn = embedder.model.smart_batching_collate

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        self.embedder.loss_model.eval()

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"
        logging.info('Running AccuracyEvaluator on {} dataset{}'.format(self.split, out_txt))

        self.dataloader.collate_fn = model.smart_batching_collate
        logits = []
        loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='Predicting in AccuracyEvaluator'):
                features, label_ids = batch_to_device(batch, self.device)
                cur_logits, cur_loss = self.embedder.loss_model(features, labels=label_ids, return_both=True)
                logits.extend(cur_logits.tolist())
                loss += cur_loss.item() * len(cur_logits)  # mean -> sum

        pred_labels = {}
        for example, pred in zip(self.dataset, logits):
            pred_labels[tuple(example.sent_ids)] = int(pred > 0)
        results = evaluation.evaluate_accuracy(self.dataset, pred_labels)
        results['avg_loss'] = loss / len(self.dataset)
        logging.info('Results for AccuracyEvaluator on {} dataset{}'.format(self.split, out_txt))
        logging.info(json.dumps(results))
        if self.embedder.loss_type == 'sgdbn':
            logging.info('Monotone classifier weights: weight={}, bias={}'.format(
                    self.embedder.loss_model.linear.weight,
                    self.embedder.loss_model.linear.bias))
        elif self.embedder.loss_type == 'logistic':
            logging.info('Classifier weights: {}'.format(list(self.embedder.loss_model.classifier.named_parameters())))

        if output_path:
            if epoch != -1:
                if steps != -1:
                    filename_txt = '_epoch{}_steps{}'.format(epoch, steps)
                else:
                    filename_txt = '_epoch{}'.format(epoch)
            else:
                filename_txt = ''
            out_basename = 'accuracy_{}{}.json'.format(self.split, filename_txt)
            with open(os.path.join(output_path, out_basename), 'w') as f:
                json.dump(results, f)
        return results['accuracy']  # Use accuracy as main metric


class SentenceEmbedder(object):
    def __init__(self, output_dir, min_epochs=1, min_train_updates=0,
                 max_seq_length=128, batch_size=16, pooling_mode='mean',
                 loss_type='logistic', distance_metric='cosine', logit_scale=1e4,
                 bias_scale=1e2, learning_rate=2e-5, dev_frac=0, weight_decay=1e-2,
                 fit_logistic_num_examples=2048, dont_save_model=False, max_retries=3, 
                 retry_threshold=0.6):
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.min_epochs = min_epochs  # Train for at least this many epochs
        self.min_train_updates = min_train_updates  # Do at least this many train updates (examples, not batches) 
        self.pooling_mode = pooling_mode
        self.loss_type = loss_type
        self.distance_metric = distance_metric
        self.logit_scale = logit_scale
        self.bias_scale = bias_scale
        self.learning_rate = learning_rate
        self.dev_frac = dev_frac
        self.weight_decay = weight_decay
        self.fit_logistic_num_examples = fit_logistic_num_examples
        self.dont_save_model = dont_save_model
        self.max_retries = max_retries  # Retry training this many times
        self.retry_threshold = retry_threshold  # Retry if final loss > retry_threshold

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.loss_model = None

    def save(self, path):
        self.model.save(os.path.join(path, 'model'))
        if self.loss_type == 'logistic':
            torch.save(self.loss_model.classifier.state_dict(), os.path.join(path, 'classifier.pt'))
        elif self.loss_type == 'sgdbn':
            torch.save(self.loss_model.batchnorm.state_dict(), os.path.join(path, 'batchnorm.pt'))
            torch.save(self.loss_model.linear.state_dict(), os.path.join(path, 'monotone.pt'))

    def load(self, path):
        logging.info('Loading SentenceEmbedder from {}'.format(path))
        word_embedding_model = NoDropoutBERT.load(os.path.join(path, 'model', '0_NoDropoutBERT'))
        pooling_model = models.Pooling.load(os.path.join(path, 'model', '1_Pooling'))
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                         device=self.device)
        if self.loss_type == 'sgdbn':
            self.loss_model = CosineLogisticBNLoss(model=self.model, 
                                                   distance_metric=self.distance_metric, 
                                                   logit_scale=self.logit_scale)
            self.loss_model.batchnorm.load_state_dict(torch.load(os.path.join(path, 'batchnorm.pt')))
            self.loss_model.linear.load_state_dict(torch.load(os.path.join(path, 'monotone.pt')))
        elif self.loss_type == 'logistic':
            self.loss_model = CosineLogisticLoss(model=self.model, 
                                                 distance_metric=self.distance_metric, 
                                                 logit_scale=self.logit_scale)
            self.loss_model.classifier.load_state_dict(torch.load(os.path.join(path, 'classifier.pt')))
        else:
            raise ValueError(self.loss_type)
        self.loss_model.to(device=self.model.device)
        self.logit_weight = 1.0
        self.logit_bias = 0.0

    def train_and_embed(self, dataset, questions, save_dir=None):
        self.train(dataset, save_dir=save_dir)
        return self.embed(questions)

    def fit_logistic(self, dataloader, T=10000, lr_init=1e0, return_normalized=False, return_loss=False):
        # Embed the training dataset 
        dist_emb = DistanceEmbedder(self.model, self.distance_metric)
        dist_emb.eval()
        x_features = []
        y_labels = []
        dataloader.collate_fn = self.model.smart_batching_collate
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Embedding in fit_logistic()'):
                features, label_ids = batch_to_device(batch, self.model.device)
                distances = dist_emb(features)
                x_features.extend(distances.tolist())
                y_labels.extend(label_ids.tolist())

        return fit_logistic_helper(x_features, y_labels, T=T, lr_init=lr_init, 
                                   return_normalized=return_normalized, 
                                   return_loss=return_loss)

    def _save_data(self, dataset, filename):
        # Dump collected data to file
        with open(filename, 'w') as f:
            for ex in dataset:
                qid1, qid2 = ex.sent_ids
                print('{}\t{}\t{}'.format(qid1, qid2, ex.label), file=f)

    def _init_model(self):
        logging.info('Initializing model.') 
        word_embedding_model = NoDropoutBERT('bert-base-uncased', 
                                             max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=self.pooling_mode == 'mean',
                                       pooling_mode_cls_token=self.pooling_mode == 'cls',
                                       pooling_mode_max_tokens=self.pooling_mode == 'max')
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                         device=self.device)

    def _init_loss_model(self):
        if self.loss_type == 'sgdbn':
            self.loss_model = CosineLogisticBNLoss(self.model, 
                                                   distance_metric=self.distance_metric, 
                                                   logit_scale=self.logit_scale)
        elif self.loss_type == 'logistic':
            self.loss_model = CosineLogisticLoss(self.model, 
                                                 distance_metric=self.distance_metric, 
                                                 logit_scale=self.logit_scale)
        else:
            raise ValueError(self.loss_type)

    def train(self, train_dataset, save_dir=None, dev_dataset=None, eval_on_train=True,
              keep_prev_init=False):
        """Train model on the given binary labeled dataset.
        
        Args:
            train_dataset: List[InputExample] containing training examples
            save_dir: place to save model parameters and outputs (default=self.output_dir)
        """
        if not save_dir:
            save_dir = self.output_dir

        num_dev = int(round(self.dev_frac * len(train_dataset)))
        if num_dev > 0:
            random.shuffle(train_dataset)
            dev_dataset = train_dataset[:num_dev]
            train_dataset = train_dataset[num_dev:]
            logging.info('Created splits: train={}, dev={}'.format(len(train_dataset), len(dev_dataset)))
        if not self.model or not keep_prev_init:
            self._init_model()
        else:
            logging.info('Initializing from previous model.')

        logging.info('Preparing training data.') 
        evaluators = []
        train_processed = SentencesDataset(train_dataset, self.model)
        train_dataloader = DataLoader(train_processed, shuffle=True, batch_size=self.batch_size,
                                      drop_last=True)  # Small batches mess with batch norm 
        logistic_processed = None
        logistic_dataloader = None
        if eval_on_train:
            evaluators.append(AccuracyEvaluator(train_dataset, train_processed, self, 'train'))
        if dev_dataset:
            logging.info('Preparing dev data.') 
            dev_processed = SentencesDataset(dev_dataset, self.model)
            evaluators.append(AccuracyEvaluator(dev_dataset, dev_processed, self, 'dev'))

        self._init_loss_model()
        num_epochs = max(self.min_epochs, math.ceil(self.min_train_updates / len(train_dataset)))

        def try_train():
            logging.info('Training for {} epoch(s).'.format(num_epochs)) 
            # Prepare evaluator
            evaluator = make_evaluator(evaluators)

            # Borrowed from training_stsbenchmark_bert.py
            # 10% of train data for warm-up
            warmup_steps = math.ceil(len(train_processed) * num_epochs / self.batch_size*0.1)
            self.model.fit(train_objectives=[(train_dataloader, self.loss_model)],
                           evaluator=evaluator,
                           epochs=num_epochs,
                           save_best_model=False,
                           warmup_steps=warmup_steps,
                           weight_decay=self.weight_decay,
                           optimizer_params={'lr': self.learning_rate, 
                                             'eps': 1e-6, 'correct_bias': False},
                           output_path=save_dir)
            torch.cuda.empty_cache()

            # Set the logit_weight and logit_bias
            if self.loss_type == 'logistic':
                self.logit_weight = self.logit_scale * self.loss_model.classifier.weight.detach().cpu().numpy()[0]
                self.logit_bias = self.logit_scale * self.loss_model.classifier.bias.detach().cpu().numpy()
                logging.info('Final classifier: w={}, b={}, boundary={}'.format(
                        self.logit_weight, self.logit_bias, - self.logit_bias / self.logit_weight))
            elif self.loss_type == 'sgdbn':
                # real_w * x + real_b = scale * ((x - mu) / (sqrt(var + eps)) * gamma + beta)
                # real_w = scale * gamma / sqrt(var + eps)
                # real_b = scale * (beta - mu * gamma / sqrt(var + eps))
                bn = self.loss_model.batchnorm
                mean_bn = bn.running_mean.detach().cpu().numpy()[0]
                var_bn = bn.running_var.detach().cpu().numpy()[0] 
                denom = np.sqrt(var_bn + bn.eps)
                w_bn = np.abs(self.loss_model.linear.weight.detach().cpu().numpy())[0]
                self.logit_weight = self.logit_scale * w_bn / denom
                self.logit_bias = self.logit_scale * (
                        self.loss_model.linear.bias.detach().cpu().numpy() - mean_bn * w_bn / denom)
                logging.info('Final batchnorm stats: mean={}, var={}'.format(mean_bn, var_bn))
                logging.info('Final batchnorm classifier: w={}, b={}, boundary={}'.format(
                        self.logit_weight, self.logit_bias, - self.logit_bias / self.logit_weight))

            # Recalibrate!
            logging.info('Recalibrating')
            recal_w, recal_b, loss = self.fit_logistic(train_dataloader, return_loss=True)  # Find best weights on whole training dataset
            self.logit_weight = np.float64(recal_w)
            self.logit_bias = np.float64(recal_b)
            self.loss_model.eval()
            if save_dir and not self.dont_save_model:
                self.save(save_dir)
            if save_dir and num_dev > 0:
                self._save_data(train_dataset, os.path.join(save_dir, 'train_split.tsv'))
                self._save_data(dev_dataset, os.path.join(save_dir, 'dev_split.tsv'))
            return loss

        # If something goes very wrong in training, restart training
        # This didn't wind up triggering in any of the runs we did for the paper results,
        # but it did occasionally trigger during development.
        # Not sure this is the best way to deal with it, but leaving it here. - Robin
        for i in range(self.max_retries):
            loss = try_train()
            if loss <= self.retry_threshold:  # Worst loss is ln(2) = 0.693
                break
            logging.info('Loss was {} > {}, retrying training...'.format(loss, self.retry_threshold))
            shutil.rmtree(save_dir)

            # Re-initialize model
            self._init_model()
            self._init_loss_model()


    def predict(self, dataset):
        """Predict on the given dataset using the current trained model.

        Args:
            dataset: List[InputExample] containing test examples.
        """
        self.loss_model.eval()

        logging.info('Preparing evaluation data.') 
        eval_data = SentencesDataset(dataset, self.model)
        eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=self.batch_size)
        eval_dataloader.collate_fn = self.model.smart_batching_collate

        logging.info('Starting prediction.')
        preds = []
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc='Predicting'):
                features, label_ids = batch_to_device(batch, self.model.device)
                preds.extend(self.loss_model(features).tolist())
        assert(len(preds) == len(dataset))
        pred_dict = {}
        for example, pred in zip(dataset, preds):
            pred_dict[tuple(example.sent_ids)] = pred
        return pred_dict

    def embed(self, questions):
        """Embed a list of questions using the current trained model.
        
        Args:
            questions: List[String], each string being a question.
        Returns:
            numpy array of size (len(questions), embedding_size)
        """
        self.loss_model.eval()
        # Can use 4 * self.batch_size--main batch size is for pair + backward pass
        embs = self.model.encode(questions, batch_size=4 * self.batch_size)
        return embs

    def get_logit(self, cos_sim):
        """Convert a cosine similarity into a (pre-sigmoid) logit.
        
        Should work if cos_sim is any numpy array, returns numpy array of same size.
        """
        return self.logit_weight * np.array(cos_sim) + self.logit_bias
