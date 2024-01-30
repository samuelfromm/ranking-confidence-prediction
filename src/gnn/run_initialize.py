import os

run = os.system


def main():
    root_dir = os.getcwd()
    root_dir = os.path.join(
        root_dir.split("af-ranking-confidence")[0], "af-ranking-confidence"
    )

    dataset = "AFRankingDatasetBinnedDistances"
    info_file = "dataset_scores.csv"
    dataset_dir = "gnn_data/AFRankingDatasetBinnedDistances"

    dataset_dir = os.path.join(root_dir, dataset_dir)

    runstr = f"python {root_dir}/src/gnn/datasets/initialize_dataset.py --dataset_dir {dataset_dir} --dataset {dataset} --info_file {info_file}"

    print(runstr)
    run(runstr)


if __name__ == "__main__":
    main()
