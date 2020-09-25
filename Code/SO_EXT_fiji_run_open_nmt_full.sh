#!/usr/bin/env bash
source bin/activate



#ontologies='CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON'
#ontologies='CHEBI'
ont='SO_EXT'

excluded_files='11532192,17696610'

tokenized_file_path='/../Output_Folders/Tokenized_Files/'

concept_norm_files_path='../Output_Folders/Concept_Norm_Files'

full_files_path='full_files/'

##PREPROCESS ALL ONTOLOGY INFORMATION - already done in preprocess
#python3 concept_normalization_preprocess_full.py -tokenized_file_path=$tokenized_file_path -excluded_files=$excluded_files -ontologies=$ont -concept_norm_files_path=$concept_norm_files_path -full_files_path=$full_files_path





#data_files='/Users/mabo1182/negacy_project/Concept_Norm_Files'

#output_files='/Users/mabo1182/negacy_project/Concept_Norm_Files/CHEBI/seq_2_seq_output/'

src_file='_combo_src_file_char.txt'

tgt_file='_combo_tgt_concept_ids_char.txt'

src_val='_combo_src_file_char_val.txt'

tgt_val='_combo_tgt_concept_ids_char_val.txt'

seq_2_seq_output='seq_2_seq_output'

#ont='CHEBI'

echo "CURRENT ONTOLOGY: $ont"

#declare -a arr=('CHEBI' 'CL' 'GO_BP' 'GO_CC' 'GO_MF' 'MOP' 'NCBITaxon' 'PR' 'SO' 'UBERON')

#declare -a arr=('no_duplicates' 'random_ids' 'shuffled_ids')
declare -a arr=('full_files')

declare -a mod=('model-char_step_100000')

#char_file='_combo_src_file_char_val.txt'
#char_file='_combo_src_file_char.txt'

pred_files='predictions'

char_file_output='_pred.txt'
char_file_output_val='_val_pred.txt'


## now loop through the above array
for i in "${arr[@]}"
do
    echo $i


        ##process the model
        onmt_preprocess -train_src $concept_norm_files_path/$ont/$i/$ont$src_file -train_tgt $concept_norm_files_path/$ont/$i/$ont$tgt_file -valid_src $concept_norm_files_path/$ont/$i/$ont$src_val -valid_tgt $concept_norm_files_path/$ont/$i/$ont$tgt_val -save_data $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-char

        #train the model
        onmt_train -data $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-char -save_model $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-model-char -model_type='text' -encoder_type='rnn' -decoder_type='rnn' -rnn_type='LSTM' -save_checkpoint_steps=5000 -valid_steps=10000 -train_steps=100000 -early_stopping=10000 -optim='sgd' #--world_size 1 #--gpu_ranks 0

        ##runs the opennmt model
        for j in "${mod[@]}"
        do
            #training data predictions
            onmt_translate -model $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-$j.pt -src $concept_norm_files_path/$ont/$i/$ont$src_file -output $concept_norm_files_path/$ont/$i/$pred_files/$ont-$j$char_file_output -replace_unk #--gpu 0 #-verbose #$i/

            ##TODO: run report
            analysis_type=''
            python3 open_nmt_evaluation.py -ontologies=$ont -concept_norm_path=$concept_norm_files_path -truth_path=$i -predicted_path=$pred_files -analysis_type=$analysis_type



            #validation data predictions
            onmt_translate -model $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-$j.pt -src $concept_norm_files_path/$ont/$i/$ont$src_val -output $concept_norm_files_path/$ont/$i/$pred_files/$ont-$j$char_file_output_val -replace_unk #--gpu 0 #-verbose #$i/


            ##TODO: run report
            analysis_type='val'
            python3 open_nmt_evaluation.py -ontologies=$ont -concept_norm_path=$concept_norm_files_path -truth_path=$i -predicted_path=$pred_files -analysis_type=$analysis_type


        done

done


