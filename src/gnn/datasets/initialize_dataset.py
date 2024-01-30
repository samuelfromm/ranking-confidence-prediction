
import argparse
from af_ranking_dataset import *
from torch_geometric.loader import DataLoader


def add_arguments(parser):
    parser.add_argument('--dataset', type=str, help='Should be a class')
    parser.add_argument('--info_file', type=str, help='The info file for the dataset')
    parser.add_argument('--dataset_dir', type=str, help="the directory containing the dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize dataset."
    )
    add_arguments(parser)
    args = parser.parse_args()

    dataset = eval(f"{args.dataset}(root='{args.dataset_dir}', info_file='{args.info_file}')")


    print(f'Length: {len(dataset)}')
    example=dataset[0]
    print(f'First example: {example}')
    print(f'Shape first example:\n node {example.x.shape} \n edge_index {example.edge_index.shape} \n edge_attr {example.edge_attr.shape}')
    print(f'Data type:\n node {example.x.dtype} \n edge_index {example.edge_index.dtype} \n edge_attr {example.edge_attr.dtype}')



    print(f'Data type: DockQ {example.DockQ.dtype}')
    print(f'Data type: ranking_confidence {example.ranking_confidence.dtype}')

    print(f'Example PDBID: {example.PDBID}')
    print(f'Example identifier: {example.identifier}')

    # Test batching

    loader = DataLoader(dataset, batch_size=3)
    batch = next(iter(loader))

    print(batch)


if __name__ == "__main__":
    exit(main())
