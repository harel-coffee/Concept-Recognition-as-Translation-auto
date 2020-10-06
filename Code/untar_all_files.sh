#!/usr/bin/env bash

ontologies=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON' 'CHEBI_EXT' 'CL_EXT' 'GO_BP_EXT' 'GO_CC_EXT' 'GO_MF_EXT' 'MOP_EXT' 'NCBITaxon_EXT' 'PR_EXT' 'SO_EXT' 'UBERON_EXT')

cd '/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/Concept-Recognition-as-Translation/'
pwd

save_filename_1='Models-SPAN_DETECTION-'
save_filename_2='-BIOBERT.tar.gz'

models='Models/SPAN_DETECTION/'
biobert='/BIOBERT/'


##extract the biober models first for each ontology
for ont in ${ontologies[@]}; do
	echo $ont
	tar -xzvf $save_filename_1$ont$save_filename_2 $models$ont$biobert

done


##extract the rest of the files 
all_files=('Code-biobert_v1.0_pubmed_pmc.tar.gz' 'Output_Folders-Concept_Norm_Files.tar.gz' 'Output_Folders-Evaluation_Files.tar.gz' 'Output_Folders-Tokenized_Files.tar.gz')

for file in ${all_files[@]}; do
	echo $file
	tar -xzvf $file
done