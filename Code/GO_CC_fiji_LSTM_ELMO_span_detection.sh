#!/usr/bin/env bash
source bin/activate


##ontologie of interest - separate to parallelize
# ontologies='CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON' ##all ontologies
ontologies='GO_CC'
##list of excluded files
excluded_files='11532192,17696610'
##biotags
biotags='B,I,O,O-'
closer_biotags='B,I'


##file path to the tokenized files to run span detection over
tokenized_file_path='/Output_Folders/Tokenized_Files/'

##location of the models we are running
save_models_path='/Modles/SPAN_DETECTION/'


algo='LSTM_ELMO' #CRF, LSTM, LSTM-CRF, char_embeddings, LSTM_ELMO
corpus='craft' #corpus

##hyperparameters for LSTM - tuning parameters for batch size, number of epochs and number of neurons.
optimizer_list='rmsprop' 
loss_list='categorical_crossentropy'
batch_size_list='18,36,53,106'
epochs_list='200,300'  
neurons_list='25,50'   
gpu_count='1' #max 4 GPUs with fiji


#running the LSTM-ELMO models for span detection
python3 span_detection.py -ontologies=$ontologies -excluded_files=$excluded_files -biotags=$biotags -closer_biotags=$closer_biotags -tokenized_file_path=$tokenized_file_path -save_models_path=$save_models_path -algo=$algo -corpus=$corpus --gpu_count=$gpu_count