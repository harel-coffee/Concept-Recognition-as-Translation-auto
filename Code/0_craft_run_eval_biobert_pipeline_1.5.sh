#!/usr/bin/env bash


#craft evaluation path
craft_st_path='/craft-st-2019/'
##path to all files
concept_recognition_path='../Concept-Recognition-as-Translation/'
##evaluation path
eval_path='/Output_Folders/Evaluation_Files/'
##concept system output path
concept_system_output='concept_system_output/'
##All folders for results
article_folder='Articles/txt/' #want files.txt
tokenized_files='Tokenized_Files/'
results_span_detection='Results_span_detection/'
concept_norm_files='Concept_Norm_Files/'
pmcid_sentence_files_path='PMCID_files_sentences/'
##annotation folder
concept_annotation='concept-annotation/'
##all ontologies of interest
ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON"
##pmcid evaluation files
evaluation_files="11319941,11604102,14624252,14675480,14691534,15018652,15070402,15238161,15328538,15560850,15615595,15619330,15784609,15850489,15882093,16026622,16027110,16410827,16517939,16611361,16787536,16800892,16968134,17029558,17201918,17206865,17465682,17503968,17565376,17677002"

##is there a gold standard or not for evaluation
gold_standard='false'

##the algorithm we are focusing on - specifically BioBERT due to running
algos='BIOBERT'  ##CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTM_ELMO, BIOBERT



##FOR BIOBERT
biobert='BIOBERT'
if [ $algos == $biobert ]; then
    ##0. you have just run the BioBERT models on fiji for span detection
    ##1. Bring the results files local for each ontology - all files with "*_test.txt*"
    declare -a arr=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')

    ##loop over each ontology and run the corresponding model
    results_span_detection='/Output_Folders/Evaluation_Files/Results_span_detection/'
    biobert='/BIOBERT'
    local_results='/Output_Folders/Evaluation_Files/Results_span_detection/'
    
    ##Grab all files with "*_test.txt*" local with fiji 
    for i in "${arr[@]}"
    do
       echo $i
       results_path=$results_span_detection$i$biobert
       local_path=$local_results$i$biobert
       # scp USERNAME@fiji.colorado.edu:$results_path/*_test.txt* $local_path

    done


    ##2. Detokenize all BioBERT results files (updated the detokenize script)

    biotags='B,I,O-,O' #ordered for importance
    gold_standard='false'
    true='true'

    declare -a ont=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')

    ##loop over each ontology and reformat the BioBERT output files to match the input
    for i in "${ont[@]}"
    do
        echo "$i"
        tokenized_files='Tokenized_Files'
        results_span_detection='Results_span_detection/'
        NER_DIR=$eval_path$tokenized_files$biobert
        OUTPUT_DIR=$eval_path$results_span_detection$i$biobert

        ##detokenize the bioBERT results files
        python3 biobert_ner_detokenize_updated.py --token_test_path=$OUTPUT_DIR/token_test.txt --label_test_path=$OUTPUT_DIR/label_test.txt --answer_path=$NER_DIR/test.tsv --output_dir=$OUTPUT_DIR --biotags=$biotags --gold_standard=$gold_standard

        echo 'DONE WITH TEST.TSV'


        ##if gold standard then we also want the gold standard information using the ontology_test.tsv files
        if [ $gold_standard == $true ]; then
            ont_test='_test.tsv'
            python3 biobert_ner_detokenize_updated.py --token_test_path=$OUTPUT_DIR/token_test.txt --label_test_path=$OUTPUT_DIR/label_test.txt --answer_path=$NER_DIR/$i$ont_test --output_dir=$OUTPUT_DIR --biotags=$biotags --gold_standard=$gold_standard


            ##classification report if gold standard
            python3 biobert_classification_report.py --ner_conll_results_path=$OUTPUT_DIR/ --biotags=$biotags --ontology=$i --output_path=$OUTPUT_DIR/ --gold_standard=$gold_standard

            #copy the classification report to the main results with ontology name
            biobert_class_report='_biobert_local_eval_files_classification_report.txt'
            cp $OUTPUT_DIR/biobert_classification_report.txt $eval_path$results_span_detection$i/$i$biobert_class_report


        fi

    done


    tokenized_files='Tokenized_Files/'
    results_span_detection='Results_span_detection/'
    biobert_prediction_results=$eval_path$results_span_detection

    ##create the evaluation dataframe!
    python3 biobert_eval_dataframe_output.py -ontologies=$ontologies -excluded_files=$evaluation_files -tokenized_file_path=$eval_path$tokenized_files -biobert_prediction_results=$biobert_prediction_results -output_path=$biobert_prediction_results -algos=$algos --pmcid_sentence_files_path=$pmcid_sentence_files_path

fi





##preprocess to get all the concepts for the next steps
python3 eval_preprocess_concept_norm_files.py -ontologies=$ontologies -results_span_detection_path=$eval_path$results_span_detection -concept_norm_files_path=$eval_path$concept_norm_files -evaluation_files=$evaluation_files


##run the open_nmt to predict
#run_eval_open_nmt.sh
