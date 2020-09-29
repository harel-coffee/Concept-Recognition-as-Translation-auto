#!/usr/bin/env bash


craft_st_path='/craft-st-2019/'
concept_recognition_path='../Concept-Recognition-as-Translation/'

eval_path='/Output_Folders/Evaluation_Files/'

concept_system_output='concept_system_output/'

article_folder='Articles/txt/' #want files.txt
tokenized_files='Tokenized_Files/'
results_span_detection='Results_span_detection/'
concept_norm_files='Concept_Norm_Files/'
pmcid_sentence_files_path='PMCID_files_sentences/'

concept_annotation='concept-annotation/'

ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON"

evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"


gold_standard='false'

algos='LSTM_ELMO' #CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTM_ELMO, BIOBERT
#algos='BIOBERT'



lstm_elmo='LSTM_ELMO'
if [ $algos == $lstm_elmo ]; then
    declare -a arr=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')

    results_span_detection='/Output_Folders/Evaluation_Files/Results_span_detection/'
    local_results='/Output_Folders/Evaluation_Files/Results_span_detection/'

    for i in "${arr[@]}"
    do
        echo $i
        results_path=$results_span_detection$i
        local_path=$local_results$i
        # scp mabo1182@fiji.colorado.edu:$results_path/* $local_path

    done



    ##preprocess to get all the concepts for the next steps

    eval_path='/Output_Folders/Evaluation_Files/'

    results_span_detection='Results_span_detection/'
    concept_norm_files='Concept_Norm_Files/'


    python3 eval_preprocess_concept_norm_files.py -ontologies=$ontologies -results_span_detection_path=$eval_path$results_span_detection -concept_norm_files_path=$eval_path$concept_norm_files -evaluation_files=$evaluation_files



fi