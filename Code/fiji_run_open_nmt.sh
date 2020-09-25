#!/usr/bin/env bash

##NEED A SUPER COMPUTER FOR THIS!

data_files='../Output_Folders/Concept_Norm_Files'


src_file='_combo_src_file_char.txt'

tgt_file='_combo_tgt_concept_ids_char.txt'

src_val='_combo_src_file_char_val.txt'

tgt_val='_combo_tgt_concept_ids_char_val.txt'


declare -a arr=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')

## now loop through the above array
for i in "${arr[@]}"
do
	echo "$i"

	##process the model
	onmt_preprocess -train_src $data_files/$i/full_files/$i$src_file -train_tgt $data_files/$i/full_files/$i$tgt_file -valid_src $data_files/$i/full_files/$i$src_val -valid_tgt $data_files/$i/full_files/$i$tgt_val -save_data $data_files/$i/seq_2_seq_output/$i-char

	##train the model
	onmt_train -data $data_files/$i/seq_2_seq_output/$i-char -save_model $data_files/$i/seq_2_seq_output/$i-model-char -model_type='text' -encoder_type='rnn' -decoder_type='rnn' -rnn_type='LSTM' -save_checkpoint_steps=5000 -valid_steps=10000 -train_steps=100000 -early_stopping=10000 -optim='sgd'

done






