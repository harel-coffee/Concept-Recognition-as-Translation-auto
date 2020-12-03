#!/usr/bin/env bash

##evaluation files
evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"

##path to the gold standard bionlp for evaluation
gold_standard_bionlp_path='/craft-st-2019/gold-annotations/craft-ca/'

##all ontologies of interest
ontologies='CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON'
##output path for concept normalization metrics results
output_path='.../Concept-Recognition-as-Translation/Evaluation_Files/Concept_Norm_Files/'

##all experiments with no duplicates first since it is a reference for the rest!
experiments='no_duplicates,random_ids,shuffled_ids,alphabetical'

##concept norm file path for all the predictions
concept_norm_file_path='.../Concept-Recognition-as-Translation/Concept_Norm_Files/'

##set up the gold standard files to be able to evaluate everything - includes experiments
python3 gs_spans_for_concept_normalization.py -evaluation_files=$evaluation_files -gold_standard_bionlp_path=$gold_standard_bionlp_path -ontologies=$ontologies -output_path=$output_path --experiments=$experiments --concept_norm_file_path=$concept_norm_file_path



##run the concept normalization pipeline (eval_open_nmt)
##evaluation path with folders
eval_path='.../Concept-Recognition-as-Translation/Evaluation_Files/'
concept_norm_files='Concept_Norm_Files/'
results_concept_norm_files='Results_concept_norm_files/'

##the ending of the model we use for all ontologies
declare -a mod=('model-char_step_100000')

##file headers and prefixes common among all ontologies
gs='gs_'
char_file='_combo_src_file_char.txt'
char_file_output='_pred.txt'


##loop over each ontology model for open_nmt and run the concept normalization pipeline (open nmt)
declare -a ont=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')

for i in "${ont[@]}"
  do
    echo "$i"
      for j in "${mod[@]}"
      do
        ##runs the opennmt model for each ontology
        onmt_translate -model $eval_path$concept_norm_files$i/$i-$j.pt -src $eval_path$concept_norm_files$i/$gs$i$char_file -output $eval_path$results_concept_norm_files$i/$gs$i-$j$char_file_output -replace_unk #-verbose #$i/
      done
#    fi
  done



##calculate metrics for the predicted vs the gold standard
##gold standard and predicted paths
gold_standard_path='.../Concept-Recognition-as-Translation/Evaluation_Files/Concept_Norm_Files/'
predicted_path='.../Concept-Recognition-as-Translation/Evaluation_Files/Results_concept_norm_files/'
##concept id file suffix with the concepts for both predicted and gold standard
concept_id_file='_combo_tgt_concept_ids_char'
##output path for the metrics files
output_path='.../Concept-Recognition-as-Translation/concept_system_output/'
##filename prefixes for each gold standard and prediction file respectively
gs_file_name='_combo_tgt_concept_ids_char'
pred_file_name='-model-char_step_100000_pred'
##original training folder
full_files='full_files'

##calculate the concept normalization metrics and output the report for the full files training regime
python3 calculate_concept_normalization_metrics.py -ontologies=$ontologies -gold_standard_path=$gold_standard_path -gs_file_name=$gs_file_name -predicted_path=$predicted_path -pred_file_name=$pred_file_name -output_path=$output_path --training_path=$concept_norm_file_path --experiments=$full_files


##Run the same pipeline for each experiment done for only the core set for each ontology
##loop over all experiments to run and get output on gold standard
##experiment output folder
experiment_output='.../Concept-Recognition-as-Translation/Results_concept_norm_files/'
##all experiments
declare -a arr=('no_duplicates' 'random_ids' 'shuffled_ids' 'alphabetical')
##folder with the model
seq_2_seq='seq_2_seq_output'
dash='_'

##running all open_nmt models for each experiment for each ontology
for a in "${arr[@]}"
    do
    for i in "${ont[@]}"
      do
        echo "$i"
          for j in "${mod[@]}"
          do
            ##runs the opennmt model
            onmt_translate -model $concept_norm_file_path$i/$a/$seq_2_seq/$i-$j.pt -src $concept_norm_file_path$i/$a/$gs$i$char_file -output $experiment_output$i/$gs$a$dash$i-$j$char_file_output -replace_unk #-verbose #$i/
          done
      done
done


##calculate metrics for experiments
experiments='no_duplicates,random_ids,shuffled_ids,alphabetical'

##calculate the concept normalization metrics comparing predicted to the gold standard
python3 calculate_concept_normalization_metrics.py -ontologies=$ontologies -gold_standard_path=$concept_norm_file_path -gs_file_name=$gs_file_name -predicted_path=$experiment_output -pred_file_name=$pred_file_name -output_path=$output_path --experiments=$experiments --training_path=$concept_norm_file_path


##calculate metrics for ConceptMapper comparison - baseline comparison
experiments='conceptmapper'
conceptmapper_path='.../Concept-Recognition-as-Translation/Concept-Recognition-as-Translation-ConceptMapper-Baseline/data/output/core/'
gold_standard_path='.../Concept-Recognition-as-Translation/Evaluation_Files/Concept_Norm_Files/'

python3 calculate_concept_normalization_metrics.py -ontologies=$ontologies -gold_standard_path=$gold_standard_path -gs_file_name=$gs_file_name -predicted_path=$conceptmapper_path -pred_file_name=$pred_file_name -output_path=$output_path --experiments=$experiments

