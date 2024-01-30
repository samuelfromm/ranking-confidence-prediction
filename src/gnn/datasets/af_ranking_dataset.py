import pickle
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_sparse import SparseTensor

import torch.nn as nn
import os.path as osp
import pandas as pd

import confidence_tools
from typing import Optional, Tuple

# Node features: plddt/100 (NOTE: plddt score lies between 0 and 100 so here we scale the value directly)
# Edge fatures: aligned_confidence_probs

#'ligand', 'receptor', 'asym_id', 'plddt', 'distances', 'aligned_confidence_probs'

# Edge features: aligned_confidence_probs
# Node features: plddt

# Assumptions:
# The processed_features_pkl features are the output of the run_assemble_features.py function. In particular, ligand is always 1 and receptor is 2 in the asym_id. Also ligand comes first, then comes receptor.


class AFRankingDataset(Dataset):
    def __init__(
        self,
        root,
        info_file,
        transform=None,
        pre_transform=None,
    ):
        self.info_file = info_file
        self.datatype = "float32"
        super(AFRankingDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.info_file

    @property
    def processed_file_names(self):
        processed_files = []

        score_path = self.raw_paths[0]  # raw_paths = root/raw/raw_file_names
        score_df = pd.read_csv(score_path, sep=",")
        for idx in range(score_df.shape[0]):
            processed_files.append(f"data_{idx}.pt")
        return processed_files

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        idx = 0
        score_path = self.raw_paths[0]  # raw_paths = root/raw_dir/raw_file_names
        score_df = pd.read_csv(score_path, sep=",")

        # we have to replace "$" for itertuples to work as expected
        score_df.columns = score_df.columns.str.replace(r"[$]", "", regex=True)

        for row in score_df.itertuples():
            # read row
            PDBID = getattr(row, "PDBID")
            processed_features_pkl = getattr(
                row, "pair_pkl"
            )  # keys: ['ligand', 'receptor', 'asym_id', 'plddt', 'distances', 'aligned_confidence_probs']
            pair_DockQ_val = getattr(row, "pair_DockQ")
            pair_ptm_val = getattr(row, "pair_ptm")
            pair_iptm_val = getattr(row, "pair_iptm")
            pair_ranking_confidence_val = getattr(row, "pair_ranking_confidence")
            identifier = osp.split(processed_features_pkl)[1].split(".")[0]

            print(f"Processing PDBID {PDBID} with identifier {identifier}")

            result_features = confidence_tools.load_data_from_pkl(
                processed_features_pkl
            )

            # Get raw features
            aligned_confidence_probs = result_features["aligned_confidence_probs"]
            asym_id = result_features["asym_id"]
            plddt = result_features["plddt"]
            distances = result_features["distances"]
            # ligand = np.array(result_features["ligand"])
            # receptor = np.array(result_features["receptor"])
            DockQ = np.array([[pair_DockQ_val]])
            ptm = np.array([[pair_ptm_val]])
            iptm = np.array([[pair_iptm_val]])
            ranking_confidence = np.array([[pair_ranking_confidence_val]])

            # Process raw features
            data = self.create_graph_data(
                aligned_confidence_probs=aligned_confidence_probs,
                plddt=plddt,
                asym_id=asym_id,
                distances=distances,
                datatype=self.datatype,
            )

            data.DockQ = torch.as_tensor(DockQ, dtype=eval(f"torch.{self.datatype}"))
            data.ptm = torch.as_tensor(ptm, dtype=eval(f"torch.{self.datatype}"))
            data.iptm = torch.as_tensor(iptm, dtype=eval(f"torch.{self.datatype}"))
            data.ranking_confidence = torch.as_tensor(
                ranking_confidence, dtype=eval(f"torch.{self.datatype}")
            )

            # Add meta information
            data.PDBID = PDBID
            data.identifier = identifier

            # Save data
            torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data

    def create_graph_data(
        self,
        aligned_confidence_probs: np.ndarray,
        plddt: np.ndarray,
        asym_id: np.ndarray,
        distances: np.ndarray,
        datatype: str,
    ):
        """Creates a data structure for a given example.

        Args:
            aligned_confidence_probs: [num_res, num_res, num_bins] (see prediction_results['aligned_confidence_probs']) (computed internally using the output logits from
            PredictedAlignedErrorHead.)
            plddt: [num_res] the predicted LDDT scores
            asym_id: [num_res] the asymmetric unit ID
            distances: [num_res, num_res] a matrix containing distances between pairs of residues (in Å) (i.e. entry (i,j) contains the distance (in Å) between residue i and j)
            datatype: str the datatype (float32 or float64)

        Returns:
            data: torch_geometric.data.data.Data object
        """

        num_res = asym_id.shape[0]
        assert aligned_confidence_probs.shape[0] == num_res
        assert plddt.shape[0] == num_res
        assert distances.shape[0] == num_res
        assert distances.shape[1] == num_res

        # Add node features
        plddt_scaled = plddt / 100
        x = torch.as_tensor(plddt_scaled, dtype=eval(f"torch.{datatype}"))
        x = x.resize_(x.shape[0], 1)

        # Add edge features
        ## Compute edge attributes
        # edge attr (i,j) is 1 if residue i and j are part of the same chain and 0 otherwise
        interface_mask = asym_id[:, None] != asym_id[None, :]
        interface_mask = np.array(interface_mask, dtype=eval(f"np.{datatype}"))
        interface_mask = interface_mask[:, :, np.newaxis]

        # edge attr (i,j) containts the 1/(1+d^2) where d is the distance between residue i and j predicted by alphafold
        transformed_distances = 1 / (1 + distances**2)
        distance_adj = transformed_distances[:, :, np.newaxis]

        # edge attr (i,j) is 1 if residue i and j are "neighbours" in the amino acid chain and 0 otherwise
        if num_res == 1:
            aa_neighbour_adj = np.array([[0]])
        else:
            aa_neighbour_adj = np.diag(np.ones(num_res - 1), k=1) + np.diag(
                np.ones(num_res - 1), k=-1
            )
            for i in range(1, num_res):
                if asym_id[i] != asym_id[i - 1]:
                    aa_neighbour_adj[i, i - 1] = 0
                    aa_neighbour_adj[i - 1, i] = 0
        aa_neighbour_adj = aa_neighbour_adj[:, :, np.newaxis]

        ## Define adjacency matrix with edge attributes
        adj = aligned_confidence_probs
        adj = np.concatenate(
            (aligned_confidence_probs, interface_mask, distance_adj, aa_neighbour_adj),
            axis=-1,
        )
        adj = adj.astype(eval(f"np.{datatype}"))
        adj = torch.as_tensor(adj)

        ## Turn adjacency matrix into edge_attr and edge_index
        adj = SparseTensor.from_dense(adj)
        row, col, edge_attr = adj.coo()
        edge_index = torch.stack([row, col], dim=0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # #DEBUG
        # print(data.x.shape)
        # print(data.edge_attr.shape)
        # print(data.x.dtype)
        # print(data.edge_attr.dtype)

        return data


class AFRankingDatasetBinnedDistances(Dataset):
    def __init__(
        self,
        root,
        info_file,
        transform=None,
        pre_transform=None,
    ):
        self.info_file = info_file
        self.datatype = "float32"
        super(AFRankingDatasetBinnedDistances, self).__init__(
            root, transform, pre_transform
        )

    @property
    def raw_file_names(self):
        return self.info_file

    @property
    def processed_file_names(self):
        processed_files = []

        score_path = self.raw_paths[0]  # raw_paths = root/raw/raw_file_names
        score_df = pd.read_csv(score_path, sep=",")
        for idx in range(score_df.shape[0]):
            processed_files.append(f"data_{idx}.pt")
        return processed_files

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        idx = 0
        score_path = self.raw_paths[0]  # raw_paths = root/raw_dir/raw_file_names
        score_df = pd.read_csv(score_path, sep=",")

        # we have to replace "$" for itertuples to work as expected
        score_df.columns = score_df.columns.str.replace(r"[$]", "", regex=True)

        for row in score_df.itertuples():
            # read row
            PDBID = getattr(row, "PDBID")
            processed_features_pkl = getattr(
                row, "pair_pkl"
            )  # keys: ['ligand', 'receptor', 'asym_id', 'plddt', 'distances', 'aligned_confidence_probs']
            pair_DockQ_val = getattr(row, "pair_DockQ")
            pair_ptm_val = getattr(row, "pair_ptm")
            pair_iptm_val = getattr(row, "pair_iptm")
            pair_ranking_confidence_val = getattr(row, "pair_ranking_confidence")
            identifier = osp.split(processed_features_pkl)[1].split(".")[0]

            print(f"Processing PDBID {PDBID} with identifier {identifier}")

            result_features = confidence_tools.load_data_from_pkl(
                processed_features_pkl
            )

            # Get raw features
            aligned_confidence_probs = result_features["aligned_confidence_probs"]
            asym_id = result_features["asym_id"]
            plddt = result_features["plddt"]
            distances = result_features["distances"]
            # ligand = np.array(result_features["ligand"])
            # receptor = np.array(result_features["receptor"])
            DockQ = np.array([[pair_DockQ_val]])
            ptm = np.array([[pair_ptm_val]])
            iptm = np.array([[pair_iptm_val]])
            ranking_confidence = np.array([[pair_ranking_confidence_val]])

            # Process raw features
            data = self.create_graph_data(
                aligned_confidence_probs=aligned_confidence_probs,
                plddt=plddt,
                asym_id=asym_id,
                distances=distances,
                datatype=self.datatype,
            )

            data.DockQ = torch.as_tensor(DockQ, dtype=eval(f"torch.{self.datatype}"))
            data.ptm = torch.as_tensor(ptm, dtype=eval(f"torch.{self.datatype}"))
            data.iptm = torch.as_tensor(iptm, dtype=eval(f"torch.{self.datatype}"))
            data.ranking_confidence = torch.as_tensor(
                ranking_confidence, dtype=eval(f"torch.{self.datatype}")
            )

            # Add meta information
            data.PDBID = PDBID
            data.identifier = identifier

            # Save data
            torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data

    def create_graph_data(
        self,
        aligned_confidence_probs: np.ndarray,
        plddt: np.ndarray,
        asym_id: np.ndarray,
        distances: np.ndarray,
        datatype: str,
    ):
        """Creates a data structure for a given example.

        Args:
            aligned_confidence_probs: [num_res, num_res, num_bins] (see prediction_results['aligned_confidence_probs']) (computed internally using the output logits from
            PredictedAlignedErrorHead.)
            plddt: [num_res] the predicted LDDT scores
            asym_id: [num_res] the asymmetric unit ID
            distances: [num_res, num_res] a matrix containing distances between pairs of residues (in Å) (i.e. entry (i,j) contains the distance (in Å) between residue i and j)
            datatype: str the datatype (float32 or float64)

        Returns:
            data: torch_geometric.data.data.Data object
        """

        num_res = asym_id.shape[0]
        assert aligned_confidence_probs.shape[0] == num_res
        assert plddt.shape[0] == num_res
        assert distances.shape[0] == num_res
        assert distances.shape[1] == num_res

        # Add node features
        plddt_scaled = plddt / 100
        x = torch.as_tensor(plddt_scaled, dtype=eval(f"torch.{datatype}"))
        x = x.resize_(x.shape[0], 1)

        # Add edge features
        ## Compute edge attributes
        # edge attr (i,j) is 1 if residue i and j are part of the same chain and 0 otherwise
        interface_mask = asym_id[:, None] != asym_id[None, :]
        interface_mask = np.array(interface_mask, dtype=eval(f"np.{datatype}"))
        interface_mask = interface_mask[:, :, np.newaxis]

        # edge attr (i,j) containts the 1/(1+d^2) where d is the distance between residue i and j predicted by alphafold
        # transformed_distances = 1/(1+distances**2)
        # distance_adj =  transformed_distances[:,:,np.newaxis]
        distance_adj = calculate_binned_distances(distances, 31, 30)

        # edge attr (i,j) is 1 if residue i and j are "neighbours" in the amino acid chain and 0 otherwise
        if num_res == 1:
            aa_neighbour_adj = np.array([[0]])
        else:
            aa_neighbour_adj = np.diag(np.ones(num_res - 1), k=1) + np.diag(
                np.ones(num_res - 1), k=-1
            )
            for i in range(1, num_res):
                if asym_id[i] != asym_id[i - 1]:
                    aa_neighbour_adj[i, i - 1] = 0
                    aa_neighbour_adj[i - 1, i] = 0
        aa_neighbour_adj = aa_neighbour_adj[:, :, np.newaxis]

        ## Define adjacency matrix with edge attributes
        adj = aligned_confidence_probs
        adj = np.concatenate(
            (aligned_confidence_probs, interface_mask, distance_adj, aa_neighbour_adj),
            axis=-1,
        )
        adj = adj.astype(eval(f"np.{datatype}"))
        adj = torch.as_tensor(adj)

        ## Turn adjacency matrix into edge_attr and edge_index
        adj = SparseTensor.from_dense(adj)
        row, col, edge_attr = adj.coo()
        edge_index = torch.stack([row, col], dim=0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # #DEBUG
        # print(data.x.shape)
        # print(data.edge_attr.shape)
        # print(data.x.dtype)
        # print(data.edge_attr.dtype)

        return data


def calculate_binned_distances(
    distances: np.array,
    max_bin: Optional[float] = 31.0,
    num_bins: Optional[int] = 64,
) -> np.ndarray:
    """Computes predicted TM alignment or predicted interface TM alignment. The function behaves the same way as the predicted_tm_score function in confidence.py but allows for the use of aligned_confidence_probs instead of logits.

    Args:
        distances: [num_res, num_res] distances between residues
        max_bin:
        num_bins:

    Returns:
        binned_distances: [num_res, num_res, num_bins] The binned distances

    """
    raw_distances = distances[:,:,np.newaxis]
    bin_edges = np.linspace(0.0, max_bin, num_bins - 1)
    inds = np.searchsorted(bin_edges, raw_distances)
    binned_distances = (inds == np.arange(num_bins)).astype(int)
    return binned_distances
