#!/usr/bin/env bash



eval_path='/Output_Folders/Evaluation_Files/'

ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON"

concept_system_output='concept_system_output/'

evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"

final_output='/Output_Folders/concept_system_output/'

algo='LSTM_ELMO' #CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTM_ELMo, BIOBERT
algo_filename_info='LSTM_ELMO_model_weights' #crf_model_full, LSTM_model, LSTM_CRF_model, char_embeddings_LSTM_model

python3 0_craft_final_output.py -ontologies=$ontologies -eval_path=$eval_path -evaluation_files=$evaluation_files -concept_system_output=$concept_system_output -final_output=$final_output -algo=$algo -algo_filename_info=$algo_filename_info


