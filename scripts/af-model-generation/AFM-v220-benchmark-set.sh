#!/bin/bash -x
#SBATCH -A berzelius-2023-131
#SBATCH --output=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-score/logs/out/af_%A_%a.out
#SBATCH --error=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-score/logs/error/af_%A_%a.err
#SBATCH --array=1-246%90
#SBATCH --gpus=1
#SBATCH -t 12:00:00

export NVIDIA_VISIBLE_DEVICES='all'
export TF_FORCE_UNIFIED_MEMORY='1'
export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'


##### AF2 CONFIGURATION #####
COMMON="/proj/berzelius-2021-29"
AFHOME=$COMMON"/af2-v2.2.0/alphafold" 			# Path of AF2-multimer-mod directory.
SINGULARITY=$AFHOME"/AF_data_v220/alphafold_v220.sif" 	# Path of singularity image.
PARAM=$AFHOME"/AF_data_v220/" 				# path of param folder containing AF2 Neural Net parameters.



##### PATHS #####
outdir=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-score/data/afm-benchmark-models
datasetdir=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-score/afm-benchmark-dataset/
#idfile=$datasetdir/ID_info.csv
idfile=$outdir/missing_af.csv #rerun




LN=$(( SLURM_ARRAY_TASK_ID )) #offset by 1 to exclude header
line=$(sed -n ${LN}p $idfile)

pdbid=$(echo $line | cut -d ',' -f 1)
oligocount=$(echo $line | cut -d ',' -f 2)
class=$(echo $line | cut -d ',' -f 3 | head -c 4)

fasta=$datasetdir/${oligocount}mer/seqres/${pdbid}.fasta

iddir=$outdir/$pdbid
[ ! -d $iddir ] && mkdir -p $iddir
[ ! -d $iddir/msas ] && cp -rs $datasetdir/${oligocount}mer/msa_${oligocount}mer_${class}/$pdbid/msas $iddir/msas # copy the directory structure and create symlinks of all files

echo "Running $PDBID"

SECONDS=0

##### TO JUST FOLD, GIVEN AN AF DEFAULT FOLDER STRUCTURE WITH MSAS #####
singularity exec --nv --bind $COMMON:$COMMON $SINGULARITY \
python3 $AFHOME/run_alphafold.py \
        --fasta_paths=$fasta \
        --model_preset=model_1_multimer_v2 \
        --output_dir=$outdir \
        --data_dir=$PARAM \
        --num_multimer_predictions_per_model=1



echo "$(($SECONDS/3600))h $((($SECONDS/60)%60))m $(($SECONDS%60))s"