#!/usr/bin/env bash

##all ontologies including core and core+extensions sets
ontologies='CHEBI,CHEBI_EXT,CL,CL_EXT,GO_BP,GO_BP_EXT,GO_CC,GO_CC_EXT,GO_MF,GO_MF_EXT,MOP,MOP_EXT,NCBITaxon,NCBITaxon_EXT,PR,PR_EXT,SO,SO_EXT,UBERON,UBERON_EXT'

##training path
training_annotation_path='.../Concept-Recognition-as-Translation/Output_Folders/Tokenized_Files/'

##obo addition path
obo_addition_path='.../Concept-Recognition-as-Translation/Output_Folders/Concept_Norm_Files/'

##gold standard path
gold_standard_path='.../Concept-Recognition-as-Translation/craft-st-2019/gold-annotations/craft-ca/'

##output path
output_path='.../Concept-Recognition-as-Translation/Output_Folders/concept_system_output/'

##create the overview table for all ontologies
python3 create_overview_table.py -ontologies=$ontologies -training_annotation_path=$training_annotation_path -obo_addition_path=$obo_addition_path -gold_standard_path=$gold_standard_path -output_path=$output_path