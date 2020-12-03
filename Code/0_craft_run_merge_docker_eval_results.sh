#!/usr/bin/env bash

##ontologies of interest
ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON"

##final output path
final_output_path='.../Concept-Recognition-as-Translation/Output_Folders/concept_system_output/'

##all algorithms
algo='CRF,LSTM,LSTM_CRF,CHAR_EMBEDDINGS,LSTM_ELMO,BIOBERT' #CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTM_ELMo, BIOBERT

##merge the results
python3 0_craft_merge_docker_eval_results.py -ontologies=$ontologies -final_output_path=$final_output_path -algo=$algo