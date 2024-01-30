#!/bin/bash -x


#module load Anaconda/2021.05-nsc1
#conda activate /proj/berzelius-2021-29/users/x_arnel/.conda/envs/afsample

BASEDIR=/proj/berzelius-2021-29/users/x_safro/git/af-ranking-confidence

# Software
MMALIGN=/proj/berzelius-2021-29/users/x_arnel/bin/MMalign
DOCKQ=/proj/berzelius-2021-29/users/x_zhwen/programs/DockQ/DockQ.py

# utils folder
bioutils=/proj/berzelius-2021-29/users/x_safro/git/bioutils

# Scripts
pDockQ_src=$bioutils/src/pDockQ_multimer.py
pDockQv2_src=$bioutils/src/pDockQv2.py
MMalign_wrapper=$bioutils/src/MMalign_wrapper.py
align_cut_renumber=$bioutils/src/align_cut_renumber.py
run_DockQ_multi=$bioutils/src/run_DockQ_multi.py
scores_from_pickle=$bioutils/src/read_scores_from_pkl.py

# Task specific
get_in_contact_chains=$bioutils/src/get_in_contact_chains.py
run_assemble_features=$BASEDIR/src/confidence/run_assemble_features.py
run_calculate_predicted_tm=$BASEDIR/src/confidence/run_calculate_predicted_tm.py
create_chain_dict=$BASEDIR/src/confidence/create_chain_dict.py

PDBID=$1
native_pdb=$2
model_pdb=$3        # e.g. path/to/unrelaxed_model_1_multimer_v2_pred_0.pdb
model_pkl=$4
outdir=$5
scorefile=$6
missing_scorefile=$7
info=$8

echo "Running scoring for ID ${PDBID}"

#########



MODELNAME=$(echo ${model_pdb} | rev | cut -d '/' -f 1 | rev | cut -d '.' -f 1) # e.g. unrelaxed_model_1_multimer_v2_pred_0
MODEL=${model_pdb}
NATIVE=${native_pdb}
pklFILE=${model_pkl}


output_dir=${outdir}/$PDBID
mkdir -p ${output_dir}


#echo $MODELNAME $MODEL $NATIVE $pklFILE
if [ -f "$NATIVE" ] && [ -f "$MODEL" ] && [ -f "$pklFILE" ]
then
        
        ##############   MMalign ##############
        MMalign_output=$(python3 ${MMalign_wrapper} -N $NATIVE -M $MODEL --MMalign $MMALIGN)
        #echo ${MMalign_output}
        # example output
        # PDBchain1;A:B:C:,PDBchain2;A:B::C,TM1;0.3119,TM2;0.2528,RMSD;3.78,ID1;0.327,ID2;0.262,IDali;0.941,L1;639,L2;798,Lali;222

        CHAINS1=$(echo ${MMalign_output} | cut -d ',' -f 1 | cut -d ';' -f 2)
        CHAINS2=$(echo ${MMalign_output} | cut -d ',' -f 2 | cut -d ';' -f 2)
        TM1=$(echo ${MMalign_output} | cut -d ',' -f 3 | cut -d ';' -f 2)
        #TM2=$(echo ${MMalign_output} | cut -d ',' -f 4 | cut -d ';' -f 2)
        L1=$(echo ${MMalign_output} | cut -d ',' -f 9 | cut -d ';' -f 2)
        L2=$(echo ${MMalign_output} | cut -d ',' -f 10 | cut -d ';' -f 2)
        #Lali=$(echo ${MMalign_output} | cut -d ',' -f 11 | cut -d ';' -f 2)

        echo -e "\t $PDBID TMscore: $TM1"

        ##############   RUN cut_and_align    ##############

        MODEL_CUT=${output_dir}/${MODELNAME}_cut_model.pdb
        NATIVE_CUT=${output_dir}/${MODELNAME}_cut_native.pdb
        DockQ_alignment_score=$(python3 $align_cut_renumber -N $NATIVE -M $MODEL -n ${NATIVE_CUT} -m ${MODEL_CUT} -A $CHAINS1 $CHAINS2 --rename_chains_rule native | grep "min_alignment_score" | cut -d ' ' -f 2 | xargs)

        #echo -e "\t $PDBID DockQ_align: ${DockQ_alignment_score}"

        ##############   MMalign - CUT  ##############
        MMalign_output_cut=$(python3 ${MMalign_wrapper} -N ${NATIVE_CUT} -M ${MODEL_CUT} --MMalign $MMALIGN)
        #echo ${MMalign_output}
        # example output
        # PDBchain1;A:B:C:,PDBchain2;A:B::C,TM1;0.3119,TM2;0.2528,RMSD;3.78,ID1;0.327,ID2;0.262,IDali;0.941,L1;639,L2;798,Lali;222

        #TM1_CUT=$(echo ${MMalign_output_cut} | cut -d ',' -f 3 | cut -d ';' -f 2)
        #TM2_CUT=$(echo ${MMalign_output_cut} | cut -d ',' -f 4 | cut -d ';' -f 2)
        L1_CUT=$(echo ${MMalign_output_cut} | cut -d ',' -f 9 | cut -d ';' -f 2)
        L2_CUT=$(echo ${MMalign_output_cut} | cut -d ',' -f 10 | cut -d ';' -f 2)
        #Lali_CUT=$(echo ${MMalign_output_cut} | cut -d ',' -f 11 | cut -d ';' -f 2)

        #echo -e "\t $PDBID TMscore: ${TM1_CUT}"

        echo -e "\t $PDBID length native cut (length model cut): ${L1_CUT} (${L2_CUT})"



        ##############   Compute chain ID map    ##############
        
        ## TODO
        #The current restrictions on alignments are too strict. For instance, for dimers an alignment A:B: A::B would still be ok as one could match up A and A and so B and B must be matched togetehr

        native_chains_str=$CHAINS1
        model_chains_str=$CHAINS2
        nber_improper_alns=$(grep -o '::' <<< "$native_chains_str $model_chains_str" | grep -c .) # count number of occurences of '::'

        if [ "$nber_improper_alns" -gt "0" ] # Check if the alignment is proper
        then
                echo "ERROR: Improper alignment ($CHAINS1 $CHAINS2)"
                #echo "native: $CHAINS1"
                #echo "model:  $CHAINS2"
                reason="improper alignment"
                echo "$PDBID,$MODELNAME,$info,$native_pdb,$model_pdb,$model_pkl,$NATIVE_CUT,$MODEL_CUT,$reason" >> $missing_scorefile

        else

                native_chains_str=$(echo $CHAINS1 | sed "s/::/:/g")
                model_chains_str=$(echo $CHAINS2 | sed "s/::/:/g")


                IFS=':' read -a native_chains <<< "$native_chains_str" ; unset IFS   # create array containing native chains
                IFS=':' read -a model_chains <<< "$model_chains_str" ; unset IFS
                IFS=$'\n'; sorted_native_chains=($(sort <<< "${native_chains[*]}")); unset IFS
                IFS=$'\n'; sorted_model_chains=($(sort <<< "${model_chains[*]}")); unset IFS

                declare -A chain_id_dict_model
                i=$(( 0 ))
                for chain in "${sorted_model_chains[@]}"
                do
                        i=$(( i + 1 ))
                        chain_id_dict_model[$chain]=$i
                done

                declare -A aln_dict
                for i in "${!native_chains[@]}"
                do
                        aln_dict[${native_chains[$i]}]=${model_chains[$i]}
                done

                declare -A native_to_model_idx
                for i in "${!native_chains[@]}"
                do
                        native_to_model_idx[${native_chains[$i]}]=${chain_id_dict_model[${model_chains[$i]}]}
                done

                echo -e "\t $PDBID Alignment: $CHAINS1 $CHAINS2"
                #for i in "${!native_to_model_idx[@]}"; do echo "${i}: ${native_to_model_idx[$i]}"; done
        

                ##############    in contact pair DockQ  ##############

                in_contact_pairs=$(python3 ${get_in_contact_chains} -pdb ${NATIVE} -max_dist 5)
                #output: A,B;A,C;A,F;B,C;B,I;B,F;B,J;B,K;C,F;E,F;E,J;E,G;I,J;I,K;F,J;F,G;J,G;J,K;
                echo -e "\t $PDBID in_contact_pairs: $in_contact_pairs"

                if [[ "$in_contact_pairs" != "No in contact chains found!" ]]
                then
                        OLDIFS=$IFS
                        IFS=";"
                        for pair in ${in_contact_pairs}
                        do
                                chain1=$(echo $pair | cut -d ',' -f 1)
                                chain2=$(echo $pair | cut -d ',' -f 2)
                                pair_DockQ=$(python3 ${DOCKQ} ${MODEL_CUT} ${NATIVE_CUT} -native_chain1 ${chain1} -model_chain1 ${chain1} -native_chain2 ${chain2} -model_chain2 ${chain2} -no_needle | grep 'DockQ' | cut -d ' ' -f 2 | xargs)
                                echo -e "\t $PDBID DockQ: $pair_DockQ pair: $pair"
                                
                                chain1id=${native_to_model_idx[$chain1]}
                                chain2id=${native_to_model_idx[$chain2]}

                                modelchain1=${aln_dict[$chain1]}
                                modelchain2=${aln_dict[$chain2]}

                                pair_pkl=${output_dir}/${MODELNAME}_ligand_n${chain1}_${chain1id}_receptor_n${chain2}_${chain2id}.pkl
                                python3 ${run_assemble_features} --afmodel_pdb $MODEL --results_pkl $pklFILE --output_pkl ${pair_pkl} --ligand ${chain1id} --receptor ${chain2id}

                                confidence_metrics=$(python3 ${run_calculate_predicted_tm} --results_pkl ${pair_pkl})
                                pair_ptm=$(echo $confidence_metrics | cut -d ',' -f 1 | cut -d ':' -f 2)
                                pair_iptm=$(echo $confidence_metrics | cut -d ',' -f 2 | cut -d ':' -f 2)
                                pair_ranking_confidence=$(echo $confidence_metrics | cut -d ',' -f 3 | cut -d ':' -f 2)


                                echo -e "\t $PDBID ptm: $pair_ptm iptm: $pair_iptm ranking_confidence: $pair_ranking_confidence pair: $pair ids: $chain1id,$chain2id"

                                ##############   Write output   ##############
                                echo $PDBID,$MODELNAME,$info,$native_pdb,$model_pdb,$model_pkl,$NATIVE_CUT,$MODEL_CUT,$pair_pkl,$TM1,$L1,$L2,$L1_CUT,$L2_CUT,$chain1,$modelchain1,$chain1id,$chain2,$modelchain2,$chain2id,$pair_DockQ,$pair_ptm,$pair_iptm,$pair_ranking_confidence >> $scorefile

                        done
                        IFS=$OLDIFS
                else
                        reason="no in contact chains"
                        echo "$PDBID,$MODELNAME,$info,$native_pdb,$model_pdb,$model_pkl,$NATIVE_CUT,$MODEL_CUT,$reason" >> $missing_scorefile
                fi
        fi
else
        echo "Not all files exist for $MODELNAME"
fi

