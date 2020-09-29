#!/usr/bin/env bash
source bin/activate

##NEED TO RUN LSTM-ELMO AND BIOBERT ON A SUPERCOMPUTER - FIJI

craft_path='../Concept-Recognition-as-Translation/CRAFT-3.1.3/'
concept_recognition_path='../Concept-Recognition-as-Translation/'

eval_path='../Concept-Recognition-as-Translation/Output_Folders/Evaluation_Files/'

concept_system_output='concept_system_output/'

article_folder='Articles/txt/' #want files.txt
tokenized_files='Tokenized_Files/'
save_models_path='Models/SPAN_DETECTION/'
results_span_detection='Results_span_detection/'
concept_norm_files='Concept_Norm_Files/'
pmcid_sentence_files_path='PMCID_files_sentences/'

concept_annotation='concept-annotation/'

ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON"

evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"


gold_standard='False'

algos='LSTM_ELMO' ##CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTM_ELMO, BIOBERT


#python3 eval_preprocess_docs.py -craft_path=$craft_path -concept_recognition_path=$concept_recognition_path -eval_path=$eval_path -concept_system_output=$concept_system_output -article_folder=$article_folder -tokenized_files=$tokenized_files -concept_annotation=$concept_annotation -ontologies=$ontologies -evaluation_files=$evaluation_files --gold_standard=$gold_standard


python3 eval_span_detection.py -ontologies=$ontologies -excluded_files=$evaluation_files -tokenized_file_path=$eval_path$tokenized_files -save_models_path=$concept_recognition_path$save_models_path -output_path=$eval_path$results_span_detection  -algos=$algos --gold_standard=$gold_standard  --pmcid_sentence_files_path=$pmcid_sentence_files_path #--all_lcs_path=$all_lcs_path




#python3 eval_preprocess_concept_norm_files.py -ontologies=$ontologies -results_span_detection_path=$eval_path$results_span_detection -concept_norm_files_path=$eval_path$concept_norm_files -evaluation_files=$evaluation_files


##run the open_nmt to predict
#run_eval_open_nmt.sh
