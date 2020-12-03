#!/usr/bin/env bash

##list of evaluation files - pmcids
evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"

##gold standard bionlp path
gold_standard_bionlp_path='.../Concept-Recognition-as-Translation/craft-st-2019/gold-annotations/craft-ca/'
##need to add the ontology

##predicted bionlp path
predicted_bionlp_path='.../Concept-Recognition-as-Translation/Output_Folders/concept_system_output/'
##need to add algo/ontology

##all algorithms to evaluate
algo_list='CRF,LSTM,LSTM_CRF,CHAR_EMBEDDINGS,LSTM_ELMO,BIOBERT'

##all ontologies
ontologies='CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON'

##output path for results
output_path='.../Concept-Recognition-as-Translation/Output_Folders/concept_system_output/'


##calculate the span detection metrics on the 30 held out documents
python3 calculate_span_detection_metrics.py -evaluation_files=$evaluation_files -gold_standard_bionlp_path=$gold_standard_bionlp_path -predicted_bionlp_path=$predicted_bionlp_path -algo_list=$algo_list -ontologies=$ontologies -output_path=$output_path