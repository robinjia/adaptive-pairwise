"""Final evaluation runs."""
import argparse
import json
import os
from sklearn.metrics import average_precision_score
import sys

from active_learning import get_parser, ActiveLearningAgent
import evaluation
#from image_embedder import ImageEmbedder
from sentence_embedder import SentenceEmbedder
import util
import worlds

def main(args):
    world = worlds.get_world(args.world)
    agent = ActiveLearningAgent(world, None, faiss_nprobe=10)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.world == 'inat' or args.world == 'celeba':  # images
        agent.trained_model = ImageEmbedder(
                args.world, None,
                loss_type=args.loss_type,
                distance_metric=args.distance_metric,
                logit_scale=args.logit_scale)
    else:
        agent.trained_model = SentenceEmbedder(
                None, min_epochs=args.min_epochs, 
                batch_size=args.batch_size,
                pooling_mode=args.pooling_mode,
                loss_type=args.loss_type,
                distance_metric=args.distance_metric,
                logit_scale=args.logit_scale,
                bias_scale=args.bias_scale)
    agent.trained_model.load(args.load_dir)
    if args.paws_qqp or args.evaluation_file:
        if args.paws_qqp:
            dataset = util.load_paws_qqp_data()
        else:
            dataset = util.load_custom_data(world, args.evaluation_file)
        preds = agent.trained_model.predict(dataset)  # Returns logits
        hard_preds = {k: int(v > 0) for k, v in preds.items()}
        results = evaluation.evaluate_accuracy(dataset, hard_preds)
        print(json.dumps(results))

        if args.paws_qqp:
            # Evaluate average precision
            y_true = [ex.label for ex in dataset]
            y_score = [preds[tuple(ex.sent_ids)] for ex in dataset]
            print('Average Precision: {}'.format(average_precision_score(y_true, y_score)))
    else:
        agent.Evaluate(args.output_dir, args.distance_metric, test=args.test)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--load_dir')
    parser.add_argument('--evaluation_file', help='Evaluate on this file instead of PR curve')
    parser.add_argument('--paws_qqp', action='store_true', help='Evaluate on PAWS QQP data')
    parser.add_argument('--test', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    main(args)
