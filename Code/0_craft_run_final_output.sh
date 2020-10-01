#!/usr/bin/env bash


##evaluation path
eval_path='/Output_Folders/Evaluation_Files/'

##list of all ontologies of interest
ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON"

##the full concept recognition output folder
concept_system_output='concept_system_output/'

##evaluation files we are working with
evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"

##the complete final output for each span detection algorithm in the format for the final performance analysis using Docker
final_output='/Output_Folders/concept_system_output/'

##the span detection algorithm to use
algo='LSTM_ELMO' #CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTM_ELMo, BIOBERT

##the corresponding model name for each span detection algorithm in order
algo_filename_info='LSTM_ELMO_model_weights' #crf_model_full, LSTM_model, LSTM_CRF_model, char_embeddings_LSTM_model, LSTM_ELMO_model_weights, biobert_model

##run the final output formatting for final performance analysis
python3 0_craft_final_output.py -ontologies=$ontologies -eval_path=$eval_path -evaluation_files=$evaluation_files -concept_system_output=$concept_system_output -final_output=$final_output -algo=$algo -algo_filename_info=$algo_filename_info


