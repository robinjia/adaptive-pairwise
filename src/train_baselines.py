"""Used for SentenceEmbedder baselines and debugging."""
import argparse
import sys

from active_learning import *
import worlds

def read_data_cache(filename):
    labeled_set = []
    labels = []
    with open(filename) as f:
        for line in f:
            qid1, qid2, label = line.strip().split('\t')
            labeled_set.append((qid1, qid2))
            labels.append(int(label))
    return labeled_set, labels

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

    agent.labeled_set = []
    agent.labels = []
    if args.add_data_file:
        cached_set, cached_labels = read_data_cache(args.add_data_file)
        agent.labeled_set.extend(cached_set)
        agent.labels.extend(cached_labels)
        for i in range(len(cached_labels)):
            agent.labeled_set_weights.append(1)
    if args.add_all_positives:
        agent.AddOraclePositives(world.GetNumTrainPositives())
    elif args.add_positives:
        agent.AddOraclePositives(args.add_positives)
    if args.add_all_stated_negatives:
        agent.AddStatedNegatives(world.GetNumTrainStatedNegatives())
    elif args.add_stated_negatives:
        agent.AddStatedNegatives(args.add_stated_negatives)
    if args.add_stated_examples:
        agent.AddRandomStated(args.add_stated_examples)
    if args.add_random_examples:
        agent.AddRandomExamples(args.add_random_examples)
    if args.add_random_negatives:
        agent.AddRandomNegatives(args.add_random_negatives)
    if args.query_batch:
        agent.GetInitialEmbedding(4 * args.batch_size)
        agent.QueryCertain(args.query_batch, args.sampling_technique, args.distance_metric)
    else:
        # Dump collected data to file; agent.QueryCertain() does this too
        data_dump_file = os.path.join(agent.output_dir, 'collected_data_{}.tsv'.format(len(agent.labels)))
        print('Writing data to {}'.format(data_dump_file))
        with open(data_dump_file, 'w') as f:
            for (qid1, qid2), label in zip(agent.labeled_set, agent.labels):
                print('{}\t{}\t{}'.format(qid1, qid2, label), file=f)

    print('Read {} examples (+={}, -={})'.format(len(agent.labels), 
          sum(1 for x in agent.labels if x == 1), sum(1 for x in agent.labels if x == 0)))

    # Train and evaluate
    agent.Train(args)
    logging.info('Allocated after Train: {} MiB'.format(torch.cuda.memory_allocated() / 2**20))
    if not args.skip_pr_eval:
        agent.Evaluate(os.path.join(args.output_dir, 'labels{}'.format(len(agent.labels))),
                       args.distance_metric)
        logging.info('Allocated after Evaluate: {} MiB'.format(torch.cuda.memory_allocated() / 2**20))

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--add_data_file')
    parser.add_argument('--add_positives', type=int, default=0)
    parser.add_argument('--add_all_positives', action='store_true')
    parser.add_argument('--add_stated_negatives', type=int, default=0)
    parser.add_argument('--add_stated_examples', type=int, default=0)
    parser.add_argument('--add_all_stated_negatives', action='store_true')
    parser.add_argument('--add_random_examples', type=int, default=0)
    parser.add_argument('--add_random_negatives', type=int, default=0)
    parser.add_argument('--query_batch', type=int, default=0)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    main(args)
