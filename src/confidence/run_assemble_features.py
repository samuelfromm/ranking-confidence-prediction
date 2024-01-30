import confidence_tools
from typing import Optional
import pickle
import os
import sys
import argparse
import numpy as np


def assemble_features(
    results_pkl_path: str,
    pdb_file_path: str,
    ligand: list,
    receptor: Optional[list] = None,
    output_results_pkl_path: Optional[str] = None,
    model_nber: Optional[int] = 0,
) -> dict:
    """Assemble new features for ligand and receptor pair.


    Args:



    Returns:

    """

    assert os.path.isfile(results_pkl_path) and os.path.isfile(pdb_file_path)
    results = confidence_tools.load_data_from_pkl(results_pkl_path)
    pdb_structure = confidence_tools.load_pdb_structure(pdb_file_path)

    asym_id = confidence_tools.compute_asym_id_from_pdb(pdb_structure, model_nber)

    # 1D features of shape [num_res]
    features_one_dim = {"plddt": results["plddt"]}

    # 2D features of shape [num_res, num_res] or features of shape [num_res, num_res, ???]
    features_two_dim = {
        "distances": confidence_tools.compute_distances_from_pdb(
            pdb_structure, model_nber
        ),
        "aligned_confidence_probs": results["aligned_confidence_probs"],
        "sequence_neighbour_adj": get_sequence_neighbour(asym_id),
    }

    new_features = confidence_tools.create_ligand_receptor_features(
        asym_id=asym_id,
        ligand=ligand,
        receptor=receptor,
        features_one_dim=features_one_dim,
        features_two_dim=features_two_dim,
    )

    if not output_results_pkl_path is None:
        with open(output_results_pkl_path, "wb") as f:
            pickle.dump(new_features, f, protocol=4)

    return new_features


def get_sequence_neighbour(asym_id):
    """Create sequence neighbour feature

    Args:
        asym_id: [num_res] the asymmetric unit id

    Returns:
        aa_neighbour_adj: [num_res,num_res,1]sequence neighbour feature
    """
    num_res = asym_id.shape[0]
    # Create sequence neighbour feature (for all chains)
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

    return aa_neighbour_adj


def add_arguments(parser):
    parser.add_argument(
        "--afmodel_pdb",
        help="path to af PDB model file",
        type=str,
    )
    parser.add_argument(
        "--results_pkl",
        help="path to results pkl file",
    )
    parser.add_argument(
        "--output_pkl",
        help="path, where to store the new features",
    )
    parser.add_argument(
        "--ligand",
        nargs="*",
        type=float,
        help="a list of chain indexes which make up the ligand",
    )
    parser.add_argument(
        "--receptor",
        nargs="*",
        type=float,
        help="a list of chain indexes which make up the receptor",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Assemble new features for ligand receptor pair."
    )
    add_arguments(parser)
    args = parser.parse_args()

    assemble_features(
        results_pkl_path=args.results_pkl,
        pdb_file_path=args.afmodel_pdb,
        ligand=args.ligand,
        receptor=args.receptor,
        output_results_pkl_path=args.output_pkl,
    )


if __name__ == "__main__":
    main()
