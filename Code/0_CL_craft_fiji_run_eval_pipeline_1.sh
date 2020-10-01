#!/usr/bin/env bash
source bin/activate

##NEED TO RUN LSTM-ELMO ON A SUPERCOMPUTER IDEALLY WITH A GPU - FIJI

##path to CRAFT
craft_path='../Concept-Recognition-as-Translation/CRAFT-3.1.3/'

##path to the concept recognition project
concept_recognition_path='../Concept-Recognition-as-Translation/'

##path to the evaluation files where all output will be stored during the evaluation
eval_path='../Concept-Recognition-as-Translation/Output_Folders/Evaluation_Files/'

##Folders for inputs and outputs 
concept_system_output='concept_system_output/' #the folder for the final output of the full concept recognition run
article_folder='Articles/txt/' #the folder with the PMC Articles text files
tokenized_files='Tokenized_Files/' #preprocessed article files to be word tokenized for BIO- format
save_models_path='Models/SPAN_DETECTION/' #all the saved models for span detection
results_span_detection='Results_span_detection/' #results from the span detection runs 
concept_norm_files='Concept_Norm_Files/' #the processed spans detected for concept normalization on the character level
pmcid_sentence_files_path='PMCID_files_sentences/' #the sentence files for the PMC articles
concept_annotation='concept-annotation/' #the concept annotations for CRAFT

##list of the ontologies of interest
ontologies="CL"

##list of the files to run through the concept recognition pipeline
evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"

##if a gold standard exists (true or false)
gold_standard='False'

##the span detection algorithm to use
algos='LSTM_ELMO' ##CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTM_ELMO, BIOBERT


##preprocess the articles (word tokenize) to prepare for span detection
#python3 eval_preprocess_docs.py -craft_path=$craft_path -concept_recognition_path=$concept_recognition_path -eval_path=$eval_path -concept_system_output=$concept_system_output -article_folder=$article_folder -tokenized_files=$tokenized_files -concept_annotation=$concept_annotation -ontologies=$ontologies -evaluation_files=$evaluation_files --gold_standard=$gold_standard

##runs the span detection models for LSTM-ELMO on supercomputer
python3 eval_span_detection.py -ontologies=$ontologies -excluded_files=$evaluation_files -tokenized_file_path=$eval_path$tokenized_files -save_models_path=$concept_recognition_path$save_models_path -output_path=$eval_path$results_span_detection  -algos=$algos --gold_standard=$gold_standard  --pmcid_sentence_files_path=$pmcid_sentence_files_path #--all_lcs_path=$all_lcs_path



##process the spans to run through concept normalization
#python3 eval_preprocess_concept_norm_files.py -ontologies=$ontologies -results_span_detection_path=$eval_path$results_span_detection -concept_norm_files_path=$eval_path$concept_norm_files -evaluation_files=$evaluation_files


##run the open_nmt to predict
#run_eval_open_nmt.sh
