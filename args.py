import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="subgraph Bernoulli embedding")

    # training scheme
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size_con', default=4096, type=int)
    parser.add_argument('--batch_size', default=512, type=int)

    parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=100, type=int)

    # model
    parser.add_argument('--dim', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--k', type=int, default=5,
                        help='Subgraph size. Default is 5.')
    parser.add_argument('--ns', type=int, default=10,
                        help='Number of negative samples. Default is 10.')
    parser.add_argument('--lam', default=0.01, type=float, help="parameter of temporal smooth")
    parser.add_argument('--dataset', default='sx-askubuntu',
                        help='Name of dataset')
    parser.add_argument('--node_num', type=int, default=159316,
                        help='graph size.')
    parser.add_argument('--T', type=int, default=3, help='Number of snapshots.')

    args = parser.parse_args()
    return args
