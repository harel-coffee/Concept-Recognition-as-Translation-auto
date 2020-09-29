#!/usr/bin/env bash


ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON"

eval_path='../Output_Folders/Evaluation_Files/'
results_concept_norm_files='Results_concept_norm_files/'
concept_norm_files='Concept_Norm_Files/'
concept_system_output='concept_system_output/'
gold_standard='gold_standard/'


evaluation_files="11532192,17696610"

evaluate='True'



##run the open_nmt to predict
#run_eval_open_nmt.sh


##concept system output
python3 eval_concept_system_output.py -ontologies=$ontologies -concept_norm_results_path=$eval_path$results_concept_norm_files -concept_norm_link_path=$eval_path$concept_norm_files -output_file_path=$eval_path$concept_system_output -gold_standard_path=$gold_standard -eval_path=$eval_path -evaluation_files=$evaluation_files -evaluate=$evaluate


