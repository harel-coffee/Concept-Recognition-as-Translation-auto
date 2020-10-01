#!/usr/bin/env bash

##evaluation path
eval_path='../Output_Folders/Evaluation_Files/'
##concept normalization folder
concept_norm_files='Concept_Norm_Files/'
##results from concept normalization folder
results_concept_norm_files='Results_concept_norm_files/'

##name of the model we are using for openNMT
declare -a mod=('model-char_step_100000')

##name of character source file that we want to predict the concept IDs for on the character level
char_file='_combo_src_file_char.txt'

##the output extension name for the predictions
char_file_output='_pred.txt'


##loop over each ontology openNMT model and run it for concept normalization
declare -a ont=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')

for i in "${ont[@]}"
  do
    echo "$i"
      for j in "${mod[@]}"
      do
        ##runs the opennmt model for each ontology
        onmt_translate -model $eval_path$concept_norm_files$i/$i-$j.pt -src $eval_path$concept_norm_files$i/$i$char_file -output $eval_path$results_concept_norm_files$i/$i-$j$char_file_output -replace_unk #-verbose #$i/
      done
  done
