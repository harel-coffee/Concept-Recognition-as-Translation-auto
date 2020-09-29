#!/usr/bin/env bash


ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON"

eval_path='/Output_Folders/Evaluation_Files/'

results_concept_norm_files='Results_concept_norm_files/'
concept_norm_files='Concept_Norm_Files/'
concept_system_output='concept_system_output/'
gold_standard='/craft-st-2019/gold-annotations/craft-ca/'


evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"


evaluate='True'



##run the open_nmt to predict
#run_eval_open_nmt.sh


##concept system output
python3 eval_concept_system_output.py -ontologies=$ontologies -concept_norm_results_path=$eval_path$results_concept_norm_files -concept_norm_link_path=$eval_path$concept_norm_files -output_file_path=$eval_path$concept_system_output -gold_standard_path=$gold_standard -eval_path=$eval_path -evaluation_files=$evaluation_files -evaluate=$evaluate


