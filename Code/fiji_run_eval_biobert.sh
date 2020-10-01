#!/usr/bin/env bash

##Move test.tsv file to supercomputer fiji for predictions




##array of all the ontologies
declare -a arr=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')


##biobert path
biobert_path='/BIOBERT/'
##model path
python_scripts='/Models/SPAN_DETECTION/'


##loop over each ontology and run the corresponding BioBERT model
for i in "${arr[@]}"
do
    ##Need to get the global step to know the number for the model run
    echo $i
    eval_results_path=$python_scripts$i$biobert_path
    algo='BIOBERT'

    ##get the global step number
    python3 biobert_model_eval_result.py -eval_results_path=$eval_results_path -ontology=$i -algo=$algo
    global_step_file='global_step_num.txt'
    eval_global_step_file=$eval_results_path$global_step_file


    global_step=$(<$eval_global_step_file)
    echo $global_step

    ##the output folder for the results of running BioBERT for span detection
    results_span_detection='/Output_Folders/Evaluation_Files/Results_span_detection/'
    biobert='/BIOBERT/'

    ##Articles to run BioBERT span detection on
    NER_DIR='/Output_Folders/Evaluation_Files/Tokenized_Files/BIOBERT/'

    ##Output files
    OUTPUT_DIR=$results_span_detection$i$biobert
    model='model.ckpt-' ##need highest number found - need to gather this from eval_results


    ##the original base model for BioBERT for running the algorithm
    biobert_original='/Code/biobert_v1.0_pubmed_pmc/'


    ## Run BioBERT span detection algorithm
    #https://blog.insightdatascience.com/using-bert-for-state-of-the-art-pre-training-for-natural-language-processing-1d87142c29e7
    python3 biobert/run_ner.py --do_train=true --do_predict=true --vocab_file=$biobert_original/vocab.txt --bert_config_file=$biobert_original/bert_config.json --init_checkpoint=$OUTPUT_DIR$model$global_step  --mmax_seq_length=410 --num_train_epochs=1.0 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR


done


##move label_test.txt and token_test.txt locally to do ner_detokenize and create dataframe for next steps - back to run_eval_biobert_pipepine_1.5.sh

