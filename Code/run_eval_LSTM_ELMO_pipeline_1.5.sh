#!/usr/bin/env bash


craft_path='/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/CRAFT-3.1.3/'
concept_recognition_path='/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/'

eval_path='/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/Evaluation_Files/'

concept_system_output='concept_system_output/'

article_folder='Articles/txt/' #want files.txt
tokenized_files='Tokenized_Files/'
python_scripts='PythonScripts/'
results_span_detection='Results_span_detection/'
concept_norm_files='Concept_Norm_Files/'
pmcid_sentence_files_path='PMCID_files_sentences/'

concept_annotation='concept-annotation/'

ontologies="CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON"
#ontologies="CHEBI"

evaluation_files="11532192,17696610"

gold_standard='True'

algos='LSTM_ELMO' #CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTM_ELMO, BIOBERT
#algos='BIOBERT'

all_lcs_path='/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/0_Gold_Standard_Annotation/Ontologies/Ontology_Of_Ignorance_all_cues_2020-03-27.txt'


lstm_elmo='LSTM_ELMO'
if [ $algos == $lstm_elmo ]; then
    declare -a arr=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')

    results_span_detection='/Users/mabo1182/negacy_project/Evaluation_Files/Results_span_detection/'
    local_results='/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/Evaluation_Files/Results_span_detection/'

    for i in "${arr[@]}"
    do
        echo $i
        results_path=$results_span_detection$i
        local_path=$local_results$i
        scp mabo1182@fiji.colorado.edu:$results_path/* $local_path

    done



##preprocess to get all the concepts for the next steps


eval_path='/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/Evaluation_Files/'

results_span_detection='Results_span_detection/'
concept_norm_files='Concept_Norm_Files/'



python3 eval_preprocess_concept_norm_files.py -ontologies=$ontologies -results_span_detection_path=$eval_path$results_span_detection -concept_norm_files_path=$eval_path$concept_norm_files -evaluation_files=$evaluation_files



fi