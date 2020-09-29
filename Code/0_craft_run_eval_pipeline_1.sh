#!/usr/bin/env bash


craft_st_path='../Concept-Recognition-as-Translation/craft-st-2019/'
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


python3 eval_preprocess_docs.py -craft_path=$craft_st_path -concept_recognition_path=$concept_recognition_path -eval_path=$eval_path -concept_system_output=$concept_system_output -article_folder=$article_folder -tokenized_files=$tokenized_files -pmcid_sentence_files=$pmcid_sentence_files_path -concept_annotation=$concept_annotation -ontologies=$ontologies -evaluation_files=$evaluation_files --gold_standard=$gold_standard





biobert='BIOBERT'
lstm_elmo='LSTM_ELMO'

if [ $algos == $biobert ]; then

    ##creates the biobert test.tsv file
    python3 eval_span_detection.py -ontologies=$ontologies -excluded_files=$evaluation_files -tokenized_file_path=$eval_path$tokenized_files -save_models_path=$concept_recognition_path$save_models_path -algos=$algos -output_path=$eval_path$results_span_detection --pmcid_sentence_files_path=$pmcid_sentence_files_path --gold_standard=$gold_standard --all_lcs_path=$all_lcs_path



    ##move test.tsv file to supercomputer (fiji) for predictions for BIOBERT
    prediction_file='../Concept-Recognition-as-Translation/Output_Folders/Evaluation_Files/Tokenized_Files/BIOBERT'
    # scp $prediction_file/test.tsv mabo1182@fiji.colorado.edu:/Users/mabo1182/negacy_project/Evaluation_Files/Tokenized_Files/BIOBERT/

    #GO TO FIJI_RUN_EVAL_BIOBERT!
    ##TODO: biobert run classification algorithm - fiji_run_eval_biobert.sh on fiji!!!
    #sbatch GPU_run_fiji_eval_biobert.sbatch - runs fiji_run_eval_biobert.sh



    ##BRING ALL OUTPUT LOCALLY FOR BIOBERT and run 0_craft_run_eval_biobert_pipeline_1.5.sh

##Move all files to supercomputer (fiji) for predictions for LSTM-ELMo
elif [ $algos == $lstm_elmo ]; then
    tokenized_files_updated='Tokenized_Files'
    # fiji_path='/Users/mabo1182/negacy_project/Evaluation_Files/'
    # scp $eval_path$tokenized_files_updated/* mabo1182@fiji.colorado.edu:$fiji_path$tokenized_files_updated/

    pmcid_sentence_files_path_updated='PMCID_files_sentences'
    # scp $eval_path$pmcid_sentence_files_path_updated/* mabo1182@fiji.colorado.edu:$fiji_path$pmcid_sentence_files_path_updated/




else

python3 eval_span_detection.py -ontologies=$ontologies -excluded_files=$evaluation_files -tokenized_file_path=$eval_path$tokenized_files -save_models_path=$concept_recognition_path$save_models_path -algos=$algos -output_path=$eval_path$results_span_detection --pmcid_sentence_files_path=$pmcid_sentence_files_path --gold_standard=$gold_standard --all_lcs_path=$all_lcs_path


python3 eval_preprocess_concept_norm_files.py -ontologies=$ontologies -results_span_detection_path=$eval_path$results_span_detection -concept_norm_files_path=$eval_path$concept_norm_files -evaluation_files=$evaluation_files



##run the open_nmt to predict
#run_eval_open_nmt.sh

fi