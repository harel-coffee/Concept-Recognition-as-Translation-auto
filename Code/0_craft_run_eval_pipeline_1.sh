#!/usr/bin/env bash

##path to the held out 30 documents for gold standard evaluation
craft_st_path='../Concept-Recognition-as-Translation/craft-st-2019/'

##path to the concept recognition project
concept_recognition_path='../Concept-Recognition-as-Translation/'

##path to the evaluation files where all output will be stored during the evaluation
eval_path='../Output_Folders/Evaluation_Files/'

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
ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON" 

##list of the files to run through the concept recognition pipeline
evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"

##if a gold standard exists (true or false)
gold_standard='False'

##the span detection algorithm to use
algos='LSTM_ELMO' ##CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTM_ELMO, BIOBERT

##preprocess the articles (word tokenize) to prepare for span detection
python3 eval_preprocess_docs.py -craft_path=$craft_st_path -concept_recognition_path=$concept_recognition_path -eval_path=$eval_path -concept_system_output=$concept_system_output -article_folder=$article_folder -tokenized_files=$tokenized_files -pmcid_sentence_files=$pmcid_sentence_files_path -concept_annotation=$concept_annotation -ontologies=$ontologies -evaluation_files=$evaluation_files --gold_standard=$gold_standard





biobert='BIOBERT'
lstm_elmo='LSTM_ELMO'

if [ $algos == $biobert ]; then

    ##creates the biobert test.tsv file
    python3 eval_span_detection.py -ontologies=$ontologies -excluded_files=$evaluation_files -tokenized_file_path=$eval_path$tokenized_files -save_models_path=$concept_recognition_path$save_models_path -algos=$algos -output_path=$eval_path$results_span_detection --pmcid_sentence_files_path=$pmcid_sentence_files_path --gold_standard=$gold_standard 

    ## 1. Move ONTOLOGY_test.tsv (where ONTOLOGY are all the ontologies) file to supercomputer for predictions (Fiji)
    ## 2. On the supercomputer run 0_craft_fiji_run_eval_biobert.sh
    ## 3. Move the biobert models local to save for each ontology
    ## 4. Move label_test.txt and token_test.txt locally for each ontology
    ## 5. Run 0_craft_run_eval_biobert_pipepine_1.5.sh to process the results from BioBERT



##Run lstm-elmo on supercomputer because issues locally (ideally with GPUs)
elif [ $algos == $lstm_elmo ]; then
    tokenized_files_updated='Tokenized_Files'
    pmcid_sentence_files_path_updated='PMCID_files_sentences'

    ## 1. Move tokenized files to supercomputer (fiji)
    ## 2. Move sentence files (PMCID_files_sentences/) to supercomputer (fiji)
    ## 3. Run 0_craft_fiji_run_eval_pipeline_1.sh (ONTOLOGY is the ontologies of choice) on supercomputer 
    ## 4. Move the /Output_Folders/Evaluation_Files/Results_span_detection/ files for LSTM_ELMO local: ONTOLOGY_LSTM_ELMO_model_weights_local_PMCARTICLE.txt where ONTOLOGY is the ontology of interest and PMCARTICLE is the PMC article ID
    ## 5. Run 0_craft_run_eval_LSTM_ELMO_pipeline_1.5.sh to process the results from LSTM_ELMO


## the rest of the span detection algorithms can be run locally
else

    ##runs the span detection models locally
    python3 eval_span_detection.py -ontologies=$ontologies -excluded_files=$evaluation_files -tokenized_file_path=$eval_path$tokenized_files -save_models_path=$concept_recognition_path$save_models_path -algos=$algos -output_path=$eval_path$results_span_detection --pmcid_sentence_files_path=$pmcid_sentence_files_path --gold_standard=$gold_standard

    ##process the spans to run through concept normalization
    python3 eval_preprocess_concept_norm_files.py -ontologies=$ontologies -results_span_detection_path=$eval_path$results_span_detection -concept_norm_files_path=$eval_path$concept_norm_files -evaluation_files=$evaluation_files



##run the open_nmt to predict
#run_eval_open_nmt.sh

fi