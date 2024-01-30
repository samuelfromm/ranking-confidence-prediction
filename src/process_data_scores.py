import pandas as pd
import os
import argparse




def add_arguments(parser):
    parser.add_argument('--scores_path', type=str, help='Path to score csv')
    parser.add_argument('--outdir', type=str, help='outpath')

def main():
    parser = argparse.ArgumentParser(
        description="Process data scores."
    )
    add_arguments(parser)
    args = parser.parse_args()

    data_scores_df = pd.read_csv(args.scores_path, sep=",")
    print(f"Inital length: {data_scores_df.shape[0]}")

    # Drop rows with NaN values
    data_scores_df = data_scores_df.dropna()
    print(f"Drop NaN: {data_scores_df.shape[0]}")

    # Drop rows
    data_scores_df = data_scores_df[data_scores_df["$L1"] == data_scores_df["$L1_CUT"]]
    print(f"Drop NATIVE LENGTH reduction: {data_scores_df.shape[0]}")
    
    data_scores_df = data_scores_df[data_scores_df["$L1_CUT"] == data_scores_df["$L2_CUT"]]
    print(f"LNATIVE_cut == LMODEL_cut : {data_scores_df.shape[0]}")


    # sort so that 2mers come first, then 3mers, etc.
    data_scores_df = data_scores_df.sort_values(['$class'], ascending = [True])

    #data_scores_df = data_scores_df[['PDBID', 'dir', 'model', 'results', 'DockQ', 'TMscore']]
    print(f"Final length: {data_scores_df.shape[0]}")
    #print(f"Final columns: {list(data_scores_df.columns)}")

    # save file
    print(f"Saving file to {args.outdir}")
    data_scores_df.to_csv(os.path.join(args.outdir, f"scores_final.csv"), sep=',',index=False)



if __name__ == '__main__':
    main()

