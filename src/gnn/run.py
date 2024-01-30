import os

run = os.system


def main():
    dataset = "AFRankingDatasetBinnedDistances"
    info_file = "dataset_scores.csv"
    data_dir = f"/home/sfromm/git/af-ranking-confidence/gnn_data/{dataset}"
    model = "DeepGCN"
    seed = 12345
    dataset_size = 0
    epochs = 2000
    num_gpus = 2
    batch_size = 16
    num_workers = 0
    learning_rate = 0.01
    training_target = "DockQ"
    weight_decay = 0

    mode = "train"  # train evaluate

    model_num_layers = 1
    model_num_hidden_channels = 8
    model_dropout_rate = 0.1
    model_dropout_rate_DeepGCNLayer = 0.1

    model_dir = f"/home/sfromm/git/af-ranking-confidence/gnn_data/checkpoints/{model}_{dataset}_n{dataset_size}_ngpu{num_gpus}_bs{batch_size}_lr{learning_rate}_wd{weight_decay}_mdr{model_dropout_rate}_ml{model_num_layers}_mhc{model_num_hidden_channels}_mdrGCN{model_dropout_rate_DeepGCNLayer}_{training_target}"
    scoredir = f"/home/sfromm/git/af-ranking-confidence/gnn_data/evaluation"

    runstr = f"python gnn.py --dataset_dir {data_dir} --dataset {dataset} --info_file {info_file} --epochs={epochs} --model {model} --seed {seed} --save_model_dir {model_dir} --batch_size {batch_size}  --dataset_size_DEBUG {dataset_size} --num_gpus={num_gpus}  --num_workers={num_workers} --training_target={training_target} --mode={mode} --pin_memory --learning_rate={learning_rate} --weight_decay={weight_decay} --model_dropout_rate={model_dropout_rate} --model_num_layers={model_num_layers} --model_num_hidden_channels={model_num_hidden_channels} --model_dropout_rate_DeepGCNLayer={model_dropout_rate_DeepGCNLayer}"

    print(runstr)
    run(runstr)


if __name__ == "__main__":
    main()
