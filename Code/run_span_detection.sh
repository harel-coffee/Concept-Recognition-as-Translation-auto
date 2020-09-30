#!/usr/bin/env bash

##list of ontologies that have annotations to preproess
ontologies='CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON'

##list of excluded files from training
excluded_files='11532192,17696610'

##BIO tags that we use from the BIO- format
biotags='B,I,O,O-'

##BIOtags to prioritize
closer_biotags='B,I'

##Path to the BIO- format tokenized files that were preprocessed
tokenized_file_path='../Output_Folders/Tokenized_Files/'

##Path to for where to save the models
save_models_path='../Models/SPAN_DETECTION/'

##the algorithm to use
algo='LSTM' #CRF, LSTM, LSTM-CRF, char_embeddings, LSTM_ELMO, BIOBERT

##Corpus we are using
corpus='CRAFT'

##LSTM hyperparameter options for training
batch_size_list='18,36,53,106'
optimizer_list='rmsprop'
loss_list='categorical_crossentropy'
epochs_list='10,100' #10, 100, 500, 1000
neurons_list='3,12' #3, 12, 25, 50


##true or false to save the model
save_model='True'

##using the crf hyperparameters or not
crf_hyperparameters='True'


##Run span detection training for all ontologies over the preprocessed tokenized files
python3 span_detection.py -ontologies=$ontologies -excluded_files=$excluded_files -biotags=$biotags -closer_biotags=$closer_biotags -tokenized_file_path=$tokenized_file_path -save_models_path=$save_models_path -algo=$algo -corpus=$corpus --batch_size_list=$batch_size_list --optimizer_list=$optimizer_list --loss_list=$loss_list --epochs_list=$epochs_list --neurons_list=$neurons_list --save_model=$save_model