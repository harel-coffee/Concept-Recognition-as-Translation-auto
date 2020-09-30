#!/usr/bin/env bash

##path to the main craft documents for training
craft_path='../CRAFT-3.1.3/'

##folder to the articles within the craft path
articles='articles/txt/' #want files.txt

##folder to the concept annotations within the craft path
concept_annotation='concept-annotation/'

##list of ontologies that have annotations to preproess
ontologies='CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON'

##output path for the BIO- format files that are tokenized
output_path='../Output_Folders/Tokenized_Files/'

##folder name for sentence files
pmcid_sentence_path='PMCID_files_sentences'

##corpus name - craft here
corpus='craft'

##list of excluded files from training if you want - default is None
excluded_files='11532192,17696610'




##path to the concept norm file path
concept_norm_files_path='../Output_Folders/Concept_Norm_Files/'
full_files_path='full_files/'


##path to the obos
obo_file_path=$craft_path$concept_annotation


##preprocess the full documents to BIO- format
python3 preprocess_docs.py -craft_path=$craft_path -articles=$articles -concept_annotation=$concept_annotation -ontologies=$ontologies -output_path=$output_path -pmcid_sentence_path=$pmcid_sentence_path -corpus=$corpus



##run the obo_addition scripts
python3 concept_normalization_obo_addition.py -obo_file_path=$obo_file_path -concept_norm_files_path=$concept_norm_files_path -ontologies=$ontologies


##preprocess the annotation files for the concept normalizaton translation
python3 concept_normalization_preprocess_full.py -tokenized_file_path=$output_path -excluded_files=$excluded_files -ontologies=$ontologies -concept_norm_files_path=$concept_norm_files_path -full_files_path=$full_files_path




