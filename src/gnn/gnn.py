import argparse
import os
import shutil
import random
import numpy as np
import time
import datetime
import tqdm
import sys
import pandas as pd

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp

import lightning as L

from models import *
from datasets import *


# modified for adding dummy node
from utils.io import load_config, save_config
from utils.log import init_logger, close_logger


from torch.utils.tensorboard import SummaryWriter


# DEBUG (avoid messages)
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


# Set default dtype
torch.set_default_dtype(torch.float32)
# On aeserv22a
# torch.set_float32_matmul_precision("high")


def add_arguments(parser):
    # for data config
    parser.add_argument(
        "--dataset_dir", type=str, help="the directory containing the dataset"
    )
    parser.add_argument("--dataset", type=str, help="Should be a class")
    parser.add_argument("--info_file", type=str, help="The info file for the dataset")

    # for specifying special models
    parser.add_argument("--model", type=str, help="the model structure in models.py")
    parser.add_argument("--save_model_dir", type=str, default="checkpoints")

    # for model config
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="learning rate"
    )
    parser.add_argument(
        "--step_size", type=float, default=50, help="scheduler step size"
    )
    # parser.add_argument("--gamma", type=float, default=0.5, help="scheduler gamma")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--epochs", type=int, default=500, help="maximum number of epochs"
    )
    # parser.add_argument(
    #    "--patience", type=int, default=50, help="patience for early stopping"
    # )

    parser.add_argument(
        "--model_dropout_rate", type=float, default=0.1, help="dropout rate used"
    )
    parser.add_argument("--model_num_layers", type=int, default=2, help=" ")
    parser.add_argument("--model_num_hidden_channels", type=int, default=8, help="")
    parser.add_argument(
        "--model_dropout_rate_DeepGCNLayer", type=float, default=0.1, help=""
    )

    # For training
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_gpus", type=int, default=-1)

    parser.add_argument("--training_target", type=str, default=None)

    # DEBUG only
    parser.add_argument(
        "--dataset_size_DEBUG",
        type=int,
        default=-1,
        help="restrict size of the dataset, DEBUG only",
    )

    parser.add_argument(
        "--mode", type=str, default="train", help="'train' or 'evaluate'"
    )


def run_training(hparams):
    # Timing
    start_time = time.time()

    # Initialize fabric
    if hparams["num_gpus"] == 0:
        fabric = L.Fabric(accelerator="cpu")
        world_size = 0
    else:
        world_size = torch.cuda.device_count()  # number of available GPUs
        if 0 < hparams["num_gpus"] <= world_size:
            world_size = hparams["num_gpus"]
        fabric = L.Fabric(accelerator="cuda", devices=world_size, strategy="ddp")
    fabric.launch()

    # Initialize logger and writer, save config
    if fabric.global_rank == 0:
        os.makedirs(hparams["save_model_dir"], exist_ok=True)
        logger = init_logger(
            log_file=os.path.join(hparams["save_model_dir"], "log.txt"),
            log_tag=hparams["model"],
        )
        writer = SummaryWriter(hparams["save_model_dir"])
        save_config(hparams, os.path.join(hparams["save_model_dir"], "config.json"))

    #
    if fabric.global_rank == 0:
        logger.info(f"Using {world_size} GPUs.")

    # Set seeds
    fabric.seed_everything(hparams["seed"])

    # Initialize dataset
    if fabric.global_rank == 0:
        logger.info(f"Running on dataset {hparams['dataset']}")
    dataset = eval(
        f"{hparams['dataset']}(root='{hparams['dataset_dir']}', info_file='{hparams['info_file']}')"
    )
    # DEBUG
    if hparams["dataset_size_DEBUG"] > 0:
        dataset = torch.utils.data.Subset(dataset, range(hparams["dataset_size_DEBUG"]))

    # Split dataset into training, validation and test set
    split = [0.6, 0.2, 0.2]
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=split,
        generator=torch.Generator().manual_seed(hparams["seed"]),
    )
    if fabric.global_rank == 0:
        logger.info(
            f"Training examples: {len(train_dataset)} Validation examples: {len(valid_dataset)} Test examples: {len(test_dataset)}"
        )

    # Configure dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
        pin_memory=hparams["pin_memory"],
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=hparams["num_workers"],
        pin_memory=hparams["pin_memory"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=hparams["num_workers"],
        pin_memory=hparams["pin_memory"],
    )

    model_params = {k: v for k, v in hparams.items() if k.startswith("model_")}
    # Initialize model
    model = eval(
        hparams["model"]
        + f"(num_node_features={dataset[0].num_node_features}, num_edge_features={dataset[0].num_edge_features}, model_params={model_params})"
    )

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    # Set up model and optimizer for accelerated training
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, valid_loader, test_loader = fabric.setup_dataloaders(
        train_loader, valid_loader, test_loader
    )

    def train_epoch(data_loader, training_target=None):
        # TRAIN
        model.train()
        train_loss = []

        start_load_time = time.time()
        for batch_id, batch in enumerate(data_loader):
            # DEBUG START
            # print(batch.x)
            # tensor[tensor!=0] = 0
            # DEBUG END

            if batch_id == 0:
                print(
                    f"rank{fabric.global_rank}: data loading time {datetime.timedelta(seconds=time.time() - start_load_time)}"
                )
            pred = model(batch)
            # Calculating the loss and gradients
            optimizer.zero_grad()
            if training_target is None:
                target = batch.y
            else:
                target = eval(f"batch.{training_target}")
            loss = loss_fn(pred, target)
            fabric.backward(loss)
            optimizer.step()
            train_loss.append(loss.detach().view(-1))

        train_loss = torch.cat(train_loss, dim=0)
        train_loss = train_loss.mean()
        return train_loss

    @torch.no_grad()
    def test(data_loader, training_target=None):
        # TEST
        model.eval()

        total_loss = []
        for batch in data_loader:
            pred = model(batch)
            if training_target is None:
                target = batch.y
            else:
                target = eval(f"batch.{training_target}")
            loss = loss_fn(pred, target)
            total_loss.append(loss.detach().view(-1))

        total_loss = torch.cat(total_loss, dim=0)
        total_loss = total_loss.mean()
        return total_loss

    best_epoch = 0
    best_val_loss = 1e10
    for epoch in range(1, hparams["epochs"] + 1):
        # Measure training time
        if fabric.global_rank == 0:
            start_train_epoch_time = time.time()

        # Training
        train_loss = train_epoch(
            data_loader=train_loader, training_target=hparams["training_target"]
        )

        # Wait for all processes to finish training
        fabric.barrier()

        fabric.all_reduce(
            data=train_loss, group=None, reduce_op="mean"
        )  # Calculate the average of all train_loss values across all ranks

        if fabric.global_rank == 0:
            # Logging
            if logger:
                logger.info(
                    f"epoch: {epoch:0>5d}/{hparams['epochs']:0>5d} \t train_loss: {train_loss:6.3f} \t time: {datetime.timedelta(seconds=time.time() - start_train_epoch_time)}"
                )
            if writer:
                writer.add_scalar("train/train-loss-epoch", train_loss, epoch)

        # Save model
        if fabric.global_rank == 0:
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            fabric.save(os.path.join(hparams["save_model_dir"], f"{epoch}.ckpt"), state)

        # Evaluation
        if epoch % 1 == 0:
            # Measure eval time
            if fabric.global_rank == 0:
                start_eval_epoch_time = time.time()

            valid_loss = test(valid_loader, training_target=hparams["training_target"])

            # Wait for all processes to finish evaluating
            fabric.barrier()

            fabric.all_reduce(
                data=valid_loss, group=None, reduce_op="mean"
            )  # Calculate the average of all loss values across all ranks

            if fabric.global_rank == 0:
                if logger:
                    logger.info(
                        f"epoch: {epoch:0>5d}/{hparams['epochs']:0>5d} \t eval_loss: {valid_loss:6.3f} \t time: {datetime.timedelta(seconds=time.time() - start_eval_epoch_time)}"
                    )
                if writer:
                    writer.add_scalar("valid/valid-loss-epoch", valid_loss, epoch)
                # Save best model
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_epoch = epoch
                    shutil.copyfile(
                        os.path.join(hparams["save_model_dir"], f"{epoch}.ckpt"),
                        os.path.join(hparams["save_model_dir"], "best.ckpt"),
                    )

    # Wait for all processes to finish evaluating
    fabric.barrier()

    # Testing
    if fabric.global_rank == 0:  # We evaluate on a single GPU
        best_model = fabric.load(os.path.join(hparams["save_model_dir"], "best.ckpt"))
        model.load_state_dict(best_model["model_state_dict"])

        test_loss = test(
            data_loader=test_loader, training_target=hparams["training_target"]
        )
        if logger:
            logger.info(
                f"best_epoch: {best_epoch:0>5d}/{hparams['epochs']:0>5d} \t test_loss: {test_loss:6.3f}"
            )

    # Log time
    if fabric.global_rank == 0:
        logger.info(
            f"Elapsed time: {datetime.timedelta(seconds=time.time() - start_time)}."
        )

    if fabric.global_rank == 0:
        close_logger(logger)


def run_evaluation(hparams):
    # Initialize fabric
    fabric = L.Fabric(accelerator="cpu")
    fabric.launch()

    save_score_dir = os.path.join(hparams["save_model_dir"], "evaluation")

    # Initialize logger
    if fabric.global_rank == 0:
        os.makedirs(save_score_dir, exist_ok=True)
        logger = init_logger(
            log_file=os.path.join(save_score_dir, "log_eval.txt"),
            log_tag=f"{hparams['model']} EVAL",
        )

    # Set seeds
    fabric.seed_everything(hparams["seed"])

    # Initialize dataset
    if fabric.global_rank == 0:
        logger.info(f"Running on dataset {hparams['dataset']}")
    dataset = eval(
        f"{hparams['dataset']}(root='{hparams['dataset_dir']}', info_file='{hparams['info_file']}')"
    )
    # DEBUG
    if hparams["dataset_size_DEBUG"] > 0:
        dataset = torch.utils.data.Subset(dataset, range(hparams["dataset_size_DEBUG"]))

    # Split dataset into training, validation and test set
    split = [0.6, 0.2, 0.2]
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=split,
        generator=torch.Generator().manual_seed(hparams["seed"]),
    )
    if fabric.global_rank == 0:
        logger.info(
            f"Training examples: {len(train_dataset)} Validation examples: {len(valid_dataset)} Test examples: {len(test_dataset)}"
        )

    # Configure dataloaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=hparams["num_workers"],
        pin_memory=hparams["pin_memory"],
    )

    model_params = {k: v for k, v in hparams.items() if k.startswith("model_")}
    # Initialize model
    model = eval(
        hparams["model"]
        + f"(num_node_features={dataset[0].num_node_features}, num_edge_features={dataset[0].num_edge_features}, model_params={model_params})"
    )

    # Initialize loss function
    loss_fn = nn.MSELoss()

    # Set up model
    model = fabric.setup(model)
    test_loader = fabric.setup_dataloaders(test_loader)

    @torch.no_grad()
    def test(data_loader, training_target=None):
        # TEST
        model.eval()

        total_loss = []
        for batch in data_loader:
            pred = model(batch)
            if training_target is None:
                target = batch.y
            else:
                target = eval(f"batch.{training_target}")
            loss = loss_fn(pred, target)
            total_loss.append(loss.detach().view(-1))

        total_loss = torch.cat(total_loss, dim=0)
        total_loss = total_loss.mean()
        return total_loss

    # Compute test loss
    if fabric.global_rank == 0:  # We evaluate on a single GPU
        best_model = fabric.load(os.path.join(hparams["save_model_dir"], "best.ckpt"))
        model.load_state_dict(best_model["model_state_dict"])

        test_loss = test(
            data_loader=test_loader, training_target=hparams["training_target"]
        )
        if logger:
            logger.info(f"best_epoch:  test_loss: {test_loss:6.3f}")

    # Print model and parameters
    logger.info(f"model: {model}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Parameter {name}: {param.data}")

    @torch.no_grad()
    def predict_to_df(data_loader, training_target=None):
        if training_target is None:
            result_df = pd.DataFrame(
                columns=[
                    "PDBID",
                    "identifier",
                    "DockQ",
                    "ptm",
                    "iptm",
                    "ranking_confidence",
                    "target",
                    "prediction",
                ]
            )
        else:
            result_df = pd.DataFrame(
                columns=[
                    "PDBID",
                    "identifier",
                    "DockQ",
                    "ptm",
                    "iptm",
                    "ranking_confidence",
                    "prediction",
                ]
            )
        # Predict
        model.eval()

        for batch in data_loader:
            pred = model(batch)
            if training_target is None:
                target = batch.y
            # else:
            #    target = eval(f"batch.{training_target}")
            DockQscore = batch.DockQ.detach().numpy()[0][0]
            ptm = batch.ptm.detach().numpy()[0][0]
            iptm = batch.iptm.detach().numpy()[0][0]
            ranking_confidence = batch.ranking_confidence.detach().numpy()[0][0]
            prediction = pred.detach().numpy()
            if not (prediction.shape[0] == prediction.shape[1] == 1):
                sys.exit("ERROR: wrong output shape")
            else:
                prediction = prediction[0][0]
            if training_target is None:
                target = batch.y
                result_df.loc[len(result_df)] = [
                    batch.PDBID[0],
                    batch.identifier[0],
                    DockQscore,
                    ptm,
                    iptm,
                    ranking_confidence,
                    target,
                    prediction,
                ]
            else:
                result_df.loc[len(result_df)] = [
                    batch.PDBID[0],
                    batch.identifier[0],
                    DockQscore,
                    ptm,
                    iptm,
                    ranking_confidence,
                    prediction,
                ]
        return result_df

    # Compute predictions over dataset
    result_df = predict_to_df(
        data_loader=test_loader, training_target=hparams["training_target"]
    )
    result_df.to_csv(
        os.path.join(hparams["save_model_dir"], "evaluation", "eval.csv"), index=False
    )

    if fabric.global_rank == 0:
        close_logger(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    hparams = vars(args)

    if args.mode == "train":
        run_training(hparams)
    elif args.mode == "evaluate":
        run_evaluation(hparams)


# TODO
# EarlyStopping, Scheduler
# Profiler?
# Hide checkpointing/logging behind callbacks?
