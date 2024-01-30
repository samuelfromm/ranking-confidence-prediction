#!/bin/bash -x

##### PATHS #####
outdir=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-score/data/afm-benchmark-models
datasetdir=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-score/afm-benchmark-dataset/
idfile=$datasetdir/IDs_complete.csv


LN=$(( SLURM_ARRAY_TASK_ID + 1 ))
line=$(sed -n ${LN}p $idfile)




missingidfile=$outdir/missing_af.csv
existingidfile=$outdir/existing_af.csv

rm $existingidfile
rm $missingidfile
touch ${missingidfile}
touch ${existingidfile}


for POS in $(seq 1 $(cat $idfile | wc -l))
do 
    line=$(sed -n ${POS}p $idfile)

    pdbid=$(echo $line | cut -d ',' -f 1)
    oligocount=$(echo $line | cut -d ',' -f 2)
    class=$(echo $line | cut -d ',' -f 3 | head -c 4)

    echo "checking $line"

    MSAS=$outdir
    model=$MSAS/$pdbid/unrelaxed_model_1_multimer_v2_pred_0.pdb


    if [ ! -f $model ]
    then
        echo $line >> ${missingidfile}
    else
        echo $line >> ${existingidfile}
    fi

done
