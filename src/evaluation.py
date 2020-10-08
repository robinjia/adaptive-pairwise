"""Evaluation code for QQP resplit data."""
import collections
import logging
import os

from util import DATA_DIR

RECALL_LANDMARKS = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]

def evaluate_accuracy(dataset, pred_labels):
    assert(all(v == 0 or v == 1 for v in pred_labels.values()))
    num_correct = tp = fp = fn = tn = 0
    for ex in dataset:
        pred = pred_labels[tuple(ex.sent_ids)]
        num_correct += (ex.label == pred)
        tp += (ex.label == 1 and pred == 1)
        fp += (ex.label == 0 and pred == 1)
        fn += (ex.label == 1 and pred == 0)
        tn += (ex.label == 0 and pred == 0)
    if tp == 0:
        precision = recall = f1 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    return {'accuracy': num_correct / len(dataset), 
            'num_examples': len(dataset),
            'true_pos': tp, 
            'false_pos': fp, 
            'false_neg': fn,
            'true_neg': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1}

def evaluate_precision_recall(score_tag, world, split):
    if world.symmetric:
        if split == 'dev':
            nodes = world.GetDevPrimary()
        elif split == 'test':
            nodes = world.GetTestPrimary()
        else:
            raise NotImplementedError
        num_questions = len(nodes)
        total_pairs = num_questions*(num_questions-1)//2
    else:
        if split == 'dev':
            primary_nodes = world.GetDevPrimary()
            secondary_nodes = world.GetDevSecondary()
        elif split == 'test':
            primary_nodes = world.GetTestPrimary()
            secondary_nodes = world.GetTestSecondary()
        else:
            raise NotImplementedError
        total_pairs = len(primary_nodes) * len(secondary_nodes)


    counts = {"random": 0, "faiss": 0, "positive": 0}
    for score, tag in score_tag:
        counts[tag] += 1
    random_weight = (total_pairs - counts["faiss"] - counts["positive"]) / counts["random"]
    print("Random: {}".format(counts["random"]))
    print("Faiss: {}".format(counts["faiss"]))
    print("Positive: {}".format(counts["positive"]))
    print("random_weight: {}".format(random_weight))

    score_tag = sorted(score_tag, reverse=True)
    precisions = [1.0]  # Add 1.0, following scikit-learn convention
    recalls = [0.0]  # Add 1.0, following scikit-learn convention
    TP = 0
    FN = counts["positive"]
    FP = 0
    results = {'num_positive': counts['positive'], 'total_pairs': total_pairs}
    avg_prec = 0.0
    best_num_errors = FN + FP

    # For 0-1 loss, assuming scores are logits
    real_FN = 0
    real_FP = 0

    for score, tag in score_tag:
        if tag == 'positive':
            TP += 1.0
            FN -= 1.0
            if score < 0:
                real_FN += 1
        elif tag == 'faiss':
            FP += 1.0
            if score >= 0:
                real_FP += 1
        else:  # tag == 'random'
            FP += random_weight
            if score >= 0:
                real_FP += random_weight
        if TP > 0:  # Avoid big spikes at low recall
            precisions.append( TP/(TP+FP) )
            recalls.append( TP/(TP+FN) )
        if tag == 'positive':
            avg_prec += precisions[-1] * (recalls[-1] - recalls[-2])
        cur_num_errors = FN + FP
        if cur_num_errors < best_num_errors:
            best_num_errors = cur_num_errors
        for x in RECALL_LANDMARKS:
            if recalls[-1] >= x and recalls[-2] < x:
                results['P@R=%d%%' % int(100 * x)] = precisions[-1]
    results['avg_prec'] = avg_prec
    results['num_errors'] = real_FN + real_FP
    results['acc'] = 1 - (real_FN + real_FP) / total_pairs
    results['best_num_errors'] = best_num_errors
    results['best_acc'] = 1 - best_num_errors / total_pairs
    results['random_weight'] = random_weight
    return results

def evaluate_wikiqa(score_qid_label, thresh=0):
    """Compute standard WikiQA evaluation metrics.

    First, binary accuracy at thresh and optimal thresh.
    
    Second, MAP, MRR, and P@1 when excluding questions without answers (ansOnly).

    Third, question-level precision, recall, and F1 with all questions (allQ).
    This just looks at the highest predicted sentence for each question, 
    or none if none of them exceed the threshold.
    This is the recommended metric from the WikiQA paper.
    We compute metrics at the given threshold, and at best threshold.
    """
    results = {}
    N = len(score_qid_label)
    qid_to_score_label = collections.defaultdict(list)
    for score, qid, label in score_qid_label:
        qid_to_score_label[qid].append((score, label)) 
    for score_label in qid_to_score_label.values():
        score_label.sort(reverse=True)  # Sort by score decreasing

    # Binary Accuracy
    num_correct = sum(int(s > thresh) == y for s, q, y in score_qid_label)
    cur_num_correct = sum(y for s, q, y in score_qid_label)
    best_num_correct = cur_num_correct
    best_thresh = float('-inf')
    for score, qid, label in sorted(score_qid_label):  # Sort by score increasing
        cur_num_correct += -1 if label else 1
        if cur_num_correct > best_num_correct:
            best_num_correct = cur_num_correct
            best_thresh = score
    results['acc'] = num_correct / N
    results['acc_best'] = best_num_correct / N
    results['acc_best_thresh'] = best_thresh

    # ansOnly MAP/MRR
    logging.info('WikiQA: {} examples, {} questions'.format(N, len(qid_to_score_label)))
    inv_ranks = []
    avg_precs = []
    for qid, score_label in qid_to_score_label.items():
        # Clean setting, following Garg et al. (AAAI 2020):
        # Remove questions that are all negative or all positive
        if all(y == 0 for s, y in score_label) or all(y == 1 for s, y in score_label): 
            continue  # Skip questions with no answers, following prior work
        rank = None
        cur_precs = []
        for i, (score, label) in enumerate(score_label):
            if label:
                if not rank:
                    rank = i + 1
                cur_precs.append((1 + len(cur_precs)) / (1 + i))
        inv_ranks.append(1 / rank)
        avg_precs.append(sum(cur_precs) / len(cur_precs))
    logging.info('WikiQA: {} examples, {} questions, {} clean questions'.format(
            N, len(qid_to_score_label), len(inv_ranks)))
    results['ansOnly_map'] = sum(avg_precs) / len(avg_precs)
    results['ansOnly_mrr'] = sum(inv_ranks) / len(inv_ranks)

    # allQ precision/recall/F1 (matches official WikiQA evaluation)
    gold_positives = sum(any(y for s, y in score_label) for score_label in qid_to_score_label.values())
    tp = 0
    fp = 0
    top_score_label = []
    for qid, score_label in qid_to_score_label.items():
        has_ans = any(y for s, y in score_label)  # If the question is answerable
        top_score, top_label = score_label[0]
        top_score_label.append((top_score, top_label))
        if top_score > thresh:  # Predict answer at given thresh
            if top_label:
                tp += 1
            else:
                fp += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / gold_positives
    results['allQ_precision'] = prec
    results['allQ_recall'] = recall
    results['allQ_f1'] = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    # allQ best precision/recall/F1
    top_score_label.sort(reverse=True)  # Sort by decreasing score
    cur_tp = 0
    cur_fp = 0
    best_prec, best_recall, best_f1 = 0, 0, 0
    best_thresh = top_score_label[0][0]
    for i, (score, label) in enumerate(top_score_label):
        if label:
            cur_tp += 1
        else:
            cur_fp += 1
        cur_prec = cur_tp / (cur_tp + cur_fp)
        cur_recall = cur_tp / gold_positives
        cur_f1 = 2 * cur_prec * cur_recall / (cur_prec + cur_recall) if cur_tp else 0.0
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_prec = cur_prec
            best_recall = cur_recall
            # Threshold should be such that predicting > threshold gives you this number
            if i+1 < len(top_score_label):
                best_thresh = top_score_label[i+1][0]
            else:
                best_thresh = float('-inf')
    results['allQ_bestf1_f1'] = best_f1
    results['allQ_bestf1_precision'] = best_prec
    results['allQ_bestf1_recall'] = best_recall
    results['allQ_bestf1_threshold'] = best_thresh

    return results


if __name__ == '__main__':
    import argparse
    import json
    import numpy as np
    import sys
    import worlds
    parser = argparse.ArgumentParser()
    parser.add_argument('world')
    parser.add_argument('pred_file')
    parser.add_argument('--test', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    split = 'test' if args.test else 'dev'
    world = worlds.get_world(args.world)
    scores_np = np.load(args.pred_file)
    dprcp_pairs, dprcp_tags = world.GetDPRCP(test=args.test)
    score_tags = list(zip(scores_np, dprcp_tags))
    pr_results = evaluate_precision_recall(score_tags, world, split)
    print(json.dumps(pr_results))

