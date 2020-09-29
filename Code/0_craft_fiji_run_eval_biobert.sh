#!/usr/bin/env bash

####move test.tsv file to fiji for predictions
#prediction_file='/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/Evaluation_Files/Tokenized_Files/BIOBERT/test.tsv'
#scp $prediction_file mabo1182@fiji.colorado.edu:/Users/mabo1182/negacy_project/Evaluation_Files/Tokenized_Files/BIOBERT/





##biobert run classification algorithm
declare -a arr=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')
#declare -a arr=('CHEBI')

##loop over each ontology and run the corresponding model
biobert_path='/BIOBERT/'
# output='output/'

save_models_path='/Output_Folders/Models/SPAN_DETECTION/'

for i in "${arr[@]}"
do
    echo $i
    eval_results_path=$save_models_path$i$biobert_path
    algo='BIOBERT'

    ##get the global step num for the final algorithm from the results of training!
    python3 biobert_model_eval_result.py -eval_results_path=$eval_results_path -ontology=$i -algo=$algo
    global_step_file='global_step_num.txt'
    eval_global_step_file=$eval_results_path$global_step_file

    global_step=$(<$eval_global_step_file)
    echo $global_step

    results_span_detection='Output_Folders/Evaluation_Files/Results_span_detection/'
    biobert='/BIOBERT/'

#    model_dir='/Users/mabo1182/negacy_project/span_detection_models/'
#    models='/models/BIOBERT/output/'
#    #/Users/mabo1182/negacy_project/span_detection_models/CHEBI/models/BIOBERT/output
#    BIOBERT_DIR=$model_dir$i$models

    NER_DIR='Output_Folders/Evaluation_Files/Tokenized_Files/BIOBERT/'
    #/Users/mabo1182/negacy_project/Evaluation_Files/Results_span_detection/CHEBI/BIOBERT'
    OUTPUT_DIR=$results_span_detection$i$biobert
    model='model.ckpt-' ##TODO: need highest number found - need to gather this from eval_results

    biobert_original='Code/biobert_v1.0_pubmed_pmc/'


    #https://blog.insightdatascience.com/using-bert-for-state-of-the-art-pre-training-for-natural-language-processing-1d87142c29e7
    python3 biobert/run_ner.py --do_train=true --do_predict=true --vocab_file=$biobert_original/vocab.txt --bert_config_file=$biobert_original/bert_config.json --init_checkpoint=$OUTPUT_DIR$model$global_step  --mmax_seq_length=410 --num_train_epochs=1.0 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR


done


###move label_test.txt and token_test.txt locally to do ner_detokenize and create dataframe for next steps - back to run_eval_pipeline_1.sh

