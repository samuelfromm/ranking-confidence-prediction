#!/bin/bash -x


module load Anaconda/2021.05-nsc1
conda activate /proj/berzelius-2021-29/users/x_arnel/.conda/envs/afsample




##### PATHS #####
modeldir=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-confidence/data/afm-benchmark-models
datasetdir=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-confidence/afm-benchmark-dataset/
idfile=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-confidence/data/IDs.csv
#idfile=$modeldir/missing_af.csv #rerun

outdir=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-confidence/data/processed-afm-benchmark-models-data


scorefile=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-confidence/data/scores.csv
rm $scorefile
touch $scorefile

echo '$PDBID,$MODELNAME,$class,$oligocount,$native_pdb,$model_pdb,$model_pkl,$NATIVE_CUT,$MODEL_CUT,$pair_pkl,$TM1,$L1,$L2,$L1_CUT,$L2_CUT,$chain1,$modelchain1,$chain1id,$chain2,$modelchain2,$chain2id,$pair_DockQ,$pair_ptm,$pair_iptm,$pair_ranking_confidence' >> $scorefile

missingscores=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-confidence/data/scores_missing.csv
rm $missingscores
touch $missingscores

echo '$PDBID,$MODELNAME,$class,$oligocount,$native_pdb,$model_pdb,$model_pkl,$NATIVE_CUT,$MODEL_CUT,$reason' >> $missingscores

for LN in $(seq 1 $(cat $idfile | wc -l))
#for LN in $(seq 23 23)
do

        line=$(sed -n ${LN}p $idfile)

        pdbid=$(echo $line | cut -d ',' -f 1)
        oligocount=$(echo $line | cut -d ',' -f 2)
        class=$(echo $line | cut -d ',' -f 3 | head -c 4)


        baseidpath=$modeldir/$pdbid
        native_pdb=$datasetdir/${oligocount}mer/pdb_clean/${pdbid}_clean.pdb
        model_pdb=$baseidpath/unrelaxed_model_1_multimer_v2_pred_0.pdb
        model_pkl=$baseidpath/result_model_1_multimer_v2_pred_0.pkl


        info="$class,$oligocount"

        #echo  ${pdbid} ${native_pdb} ${model_pdb} ${model_pkl} ${outdir}
        bash calculate_score_singleid_afbenchmark.sh ${pdbid} ${native_pdb} ${model_pdb} ${model_pkl} ${outdir} ${scorefile} ${missingscores} ${info}
        # PDBID=$1
        # native_pdb=$2
        # model_pdb=$3        # e.g. path/to/unrelaxed_model_1_multimer_v2_pred_0.pdb
        # model_pkl=$4
        # outdir=$5
        # scorefile=$6
        # missing_scorefile=$7
        # info=$8
done