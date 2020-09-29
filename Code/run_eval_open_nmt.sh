#!/usr/bin/env bash

eval_path='../Output_Folders/Evaluation_Files/'

concept_norm_files='Concept_Norm_Files/'
results_concept_norm_files='Results_concept_norm_files/'


declare -a mod=('model-char_step_100000')

char_file='_combo_src_file_char.txt'

char_file_output='_pred.txt'


##loop over each ontology model and run them
declare -a ont=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')

for i in "${ont[@]}"
  do
    echo "$i"
#    if [ $i = 'CHEBI' ]; then
      for j in "${mod[@]}"
      do
        ##runs the opennmt model
        onmt_translate -model $eval_path$concept_norm_files$i/$i-$j.pt -src $eval_path$concept_norm_files$i/$i$char_file -output $eval_path$results_concept_norm_files$i/$i-$j$char_file_output -replace_unk #-verbose #$i/
      done
#    fi
  done
