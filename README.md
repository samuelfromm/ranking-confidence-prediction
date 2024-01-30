# ranking-confidence-prediction (example version)

Use machine learning to improve the predictive capabilities of the "ranking_confidence" score output by AlphaFold.


## Data pipeline

1. Run alphafold: AFM-v220-benchmark-set.sh
2. Process alphafold models, calculate scores and prepare features: calculate_scores_afbenchmark.sh
3. We filter the score file as follows (process_data_scores.py): We filter out examples with NaN values (see 2.) as well as examples with mismatched lengths of the model or native structure.
    (Note: The idea is that the seqres sequence (used in the alphafold model) should be a supersequence of the native PDB model sequence)
4. Using the examples collected in 3 we create the final score file, which can be used to create the dataset(s).

## Run training/evaluation

Use run.py



