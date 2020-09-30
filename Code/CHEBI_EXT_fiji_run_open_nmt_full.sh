#!/usr/bin/env bash
source bin/activate


##all ontologies
#ontologies='CHEBI,CL,GO_BP,GO_CC,GO_MF,MOP,NCBITaxon,PR,SO,UBERON'
##one ontology at a time for speed - parallelization
ont='CHEBI_EXT'

##list of excluded files from training
excluded_files='11532192,17696610'

##Path to the BIO- format tokenized files that were preprocessed
tokenized_file_path='/../Output_Folders/Tokenized_Files/'

##path to the concept norm file path
concept_norm_files_path='../Output_Folders/Concept_Norm_Files'
full_files_path='full_files/'


##preprocess all ontology information for concept normalization if not already done
python3 concept_normalization_preprocess_full.py -tokenized_file_path=$tokenized_file_path -excluded_files=$excluded_files -ontologies=$ont -concept_norm_files_path=$concept_norm_files_path -full_files_path=$full_files_path



##filename endings of all the different files for openNMT
#character source file with the text spans
src_file='_combo_src_file_char.txt'

#character target file with the concept IDs
tgt_file='_combo_tgt_concept_ids_char.txt'

#character source validation file with the text spans
src_val='_combo_src_file_char_val.txt'

#character target validation file with the concept IDs
tgt_val='_combo_tgt_concept_ids_char_val.txt'

#the output filename
seq_2_seq_output='seq_2_seq_output'

##output the name of the ontology for progress report
echo "CURRENT ONTOLOGY: $ont"



## now loop through the above array of filenames for the different experiments
declare -a arr=('full files' 'no_duplicates' 'random_ids' 'shuffled_ids' 'alphabetical')

##the model name for OpenNMT
declare -a mod=('model-char_step_100000')


##filenames for the output predictions
pred_files='predictions'
char_file_output='_pred.txt'
char_file_output_val='_val_pred.txt'


##Number of threads to use for GPUs
export OMP_NUM_THREADS=16



## now loop through the above array
for i in "${arr[@]}"
do
    echo $i


        ##process the model
        onmt_preprocess -train_src $concept_norm_files_path/$ont/$i/$ont$src_file -train_tgt $concept_norm_files_path/$ont/$i/$ont$tgt_file -valid_src $concept_norm_files_path/$ont/$i/$ont$src_val -valid_tgt $concept_norm_files_path/$ont/$i/$ont$tgt_val -save_data $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-char --num_threads=16

        #train the model
        onmt_train -data $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-char -save_model $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-model-char -model_type='text' -encoder_type='rnn' -decoder_type='rnn' -rnn_type='LSTM' -save_checkpoint_steps=5000 -valid_steps=10000 -train_steps=100000 -early_stopping=10000 -optim='sgd' #--world_size 1 #--gpu_ranks 0

        ##runs the opennmt model to translate/predict
        for j in "${mod[@]}"
        do
            #training data predictions
            onmt_translate -model $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-$j.pt -src $concept_norm_files_path/$ont/$i/$ont$src_file -output $concept_norm_files_path/$ont/$i/$pred_files/$ont-$j$char_file_output -replace_unk #--gpu 0 #-verbose #$i/

            #run an evaluation report on the training data predictions
            analysis_type=''
            python3 open_nmt_evaluation.py -ontologies=$ont -concept_norm_path=$concept_norm_files_path -truth_path=$i -predicted_path=$pred_files -analysis_type=$analysis_type



            #validation data predictions
            onmt_translate -model $concept_norm_files_path/$ont/$i/$seq_2_seq_output/$ont-$j.pt -src $concept_norm_files_path/$ont/$i/$ont$src_val -output $concept_norm_files_path/$ont/$i/$pred_files/$ont-$j$char_file_output_val -replace_unk #--gpu 0 #-verbose #$i/


            #run an evaluation report on the validation data predictions
            analysis_type='val'
            python3 open_nmt_evaluation.py -ontologies=$ont -concept_norm_path=$concept_norm_files_path -truth_path=$i -predicted_path=$pred_files -analysis_type=$analysis_type


        done

done

