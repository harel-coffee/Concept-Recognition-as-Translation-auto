#!/usr/bin/env bash

##biobert model directory and model name
BIOBERT_DIR='../Code/biobert_v1.0_pubmed_pmc' #pubmed and pmc is the closest model to our data
biobert_model='biobert_model.ckpt' #the beginning of the model name up until .ckpt


##output information for the biobert models
span_detection_models='/Models/SPAN_DETECTION/'
biobert='/BIOBERT'
output='/output'

##the biotags we are using
biotags='B,I,O,O-'

##array of all the ontologies of interest
declare -a ont=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')
# declare -a ont=('CHEBI')


##loop over each ontology model and train them
for i in "${ont[@]}"
  do
    echo "$i"
    ##named entitity recognition directory and output directory for the final models
    NER_DIR=$span_detection_models$i$biobert
    OUTPUT_DIR=$span_detection_models$i$biobert$output

    #need to delete all the stuff in the folder so that it will retrain the model
	rm -rf $OUTPUT_DIR
	mkdir $OUTPUT_DIR/

	#run the ner stuff with BERT tuning
	python3 run_ner.py --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/$biobert_model --mmax_seq_length=410 --train_batch_size=32 --eval_batch_size=8 --predict_batch_size=8 --learning_rate=1e-5 --num_train_epochs=10.0 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR


	#detokenize to put the BERT format back together into Conll format (updated the original detokenized file)
	python3 biobert_ner_detokenize_updated.py --token_test_path=$OUTPUT_DIR/token_test.txt --label_test_path=$OUTPUT_DIR/label_test.txt --answer_path=$NER_DIR/test.tsv --output_dir=$OUTPUT_DIR

  done

