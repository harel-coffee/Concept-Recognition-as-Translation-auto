import os
import re
import gzip
import argparse
import numpy as np
import nltk
# nltk.download()
import nltk.data
import sys
# import termcolor
# from termcolor import colored, cprint
from xml.etree import ElementTree as ET
import xml.etree.ElementTree as ET
from xml.dom import minidom
import multiprocessing as mp
import functools
import resource, sys
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import WordPunctTokenizer
import copy
from sklearn_crfsuite import CRF
from sklearn import model_selection
from sklearn_crfsuite.metrics import flat_classification_report
import eli5
from IPython.display import display
from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import time
# from sklearn.grid_search import RandomizedSearchCV
import joblib
import datetime
import argparse


def preprocess_data(tokenized_file_path, ontology, ontology_dict, concept_norm_files_path, evaluation_files):

    pmc_mention_id_index = 0  ##provide an id for it to use as its unique identifier - with pmcid

    for root, directories, filenames in os.walk(tokenized_file_path + ontology + '/'):
        for filename in sorted(filenames):

            ##per pmcid file - want to combine per ontology
            if filename.endswith('.txt') and ontology in filename and 'local' in filename and 'pred' not in filename and filename.split('_')[-1].replace('.txt','') in evaluation_files:
                print(filename)
                pmc_mention_id_index += 1 ##need to ensure we go up one always

                ##columns = ['PMCID', 'SENTENCE_NUM', 'SENTENCE_START', 'SENTENCE_END', 'WORD', 'POS_TAG', 'WORD_START', 'WORD_END', 'BIO_TAG', 'PMC_MENTION_ID', 'ONTOLOGY_CONCEPT_ID', 'ONTOLOGY_LABEL']

                pmc_tokenized_file_df = pd.read_csv(root+filename, sep='\t', header=0, quotechar='"', quoting=3)#, encoding='utf-8', engine='python')

                for index, row in pmc_tokenized_file_df.iterrows():
                    # print(index,row)


                    ##concepts: ONTOLOGY_DICT - pmc_mention_id -> [sentence_num, [word], [(word_indices)], span_model, [biotags]]

                    span_model = filename.replace('.txt', '')
                    sentence_num = row['SENTENCE_NUM']
                    word = str(row['WORD'])
                    word_start = row['WORD_START']
                    word_end = row['WORD_END']
                    bio_tag = row['BIO_TAG']

                    ##all Nones!
                    pmc_mention_id = row['PMC_MENTION_ID']
                    ontology_concept_id = row['ONTOLOGY_CONCEPT_ID']
                    ontology_label = row['ONTOLOGY_LABEL']
                    # all_nones = {pmc_mention_id, ontology_concept_id, ontology_label}

                    # print('word', word, bio_tag, word_start, word_end)

                    ##the dataframe reads the word 'null' as a NaN which is bad and so we change it
                    if word == 'nan':
                        word = 'null'
                        # print('NULL WORD!')
                        # print(row)
                        # raise Exception('ERROR IN DATAFRAME WITH A WEIRD VALUE!')


                    if {pmc_mention_id, ontology_concept_id, ontology_label} != {'None'}:

                        print(pmc_mention_id, ontology_concept_id, ontology_label)
                        raise Exception('ERROR WITH SPAN DETECTION NONES AT THE END!')
                    else:
                        ##TODO: issues here!
                        ##update pmc_mention_id to be an actual id: Ontology_yyyy_mm_dd_Instance_#####
                        str_date = datetime.date.today()

                        str_date = str_date.strftime("%Y-%m-%d")

                        pmc_mention_id = '%s_%s_%s_%s' %(ontology, str_date.replace('-', '_'), 'Instance', pmc_mention_id_index) #create a pmc_mention_id for future reference


                        ##beginning of a concept
                        if bio_tag == 'B':
                            if ontology_dict.get(pmc_mention_id):
                                pmc_mention_id_index += 1
                                pmc_mention_id = '%s_%s_%s_%s' % (ontology, str_date.replace('-', '_'), 'Instance', pmc_mention_id_index)
                                ontology_dict[pmc_mention_id] = [sentence_num, [word], [(word_start, word_end)],span_model]
                            else:
                                ontology_dict[pmc_mention_id] = [sentence_num, [word], [(word_start,word_end)], span_model]

                            ##set the value of pmc_mention_id also
                            pmc_tokenized_file_df.at[index, 'PMC_MENTION_ID'] = pmc_mention_id


                        ##continuation of a word - sometimes missing a B
                        elif bio_tag == 'I':
                            # print('word', word, word_start, word_end)
                            ##TODO: error because there is no B nearby - the ontology_dict doesn't have it!
                            if ontology_dict.get(pmc_mention_id):
                                ontology_dict[pmc_mention_id][1] += [word] #' %s' %word
                                ontology_dict[pmc_mention_id][2] += [(word_start,word_end)]
                            else: ##error because its missing a B to start
                                ontology_dict[pmc_mention_id] = [sentence_num, ['%s%s' %('...', word)], [(word_start, word_end)], span_model] #' %s' %word


                            ##set the value of pmc_mention_id also
                            pmc_tokenized_file_df.at[index, 'PMC_MENTION_ID'] = pmc_mention_id

                        ##discontinuity
                        elif bio_tag == 'O-':
                            if ontology_dict.get(pmc_mention_id): #and not ontology_dict[pmc_mention_id][1].endswith('...'):
                                # print('DISCONTINUITY! MADE IT HERE!')
                                ontology_dict[pmc_mention_id][1] += ['...'] #' ...'
                                ontology_dict[pmc_mention_id][2] += [(word_start, word_end)]
                            else: ##weird error where the concept is missing a B
                                ontology_dict[pmc_mention_id] = [sentence_num, ['...'], [(word_start, word_end)], span_model]

                        ##end of concept/ no concept
                        elif bio_tag == 'O':
                            if ontology_dict.get(pmc_mention_id):
                                # print('pmc_mention_id', pmc_mention_id)
                                # print('sentence num', sentence_num)
                                # print('updated concept:', ontology_dict[pmc_mention_id][1])
                                pmc_mention_id_index += 1

                            ##no concept at all
                            else:
                                pass

                        else: #'O-' continuously
                            raise Exception('ERROR WITH A WEIRD TAG OTHER THAN THE 4!')

                # for pmc_mention_id in ontology_dict:
                #     if '...' in ontology_dict[pmc_mention_id][1]:
                #         # print(ontology_dict[pmc_mention_id])
                #         pass

                ##output the new updated dataframe with pmc_mention_id labels
                pmc_tokenized_file_df.to_pickle('%s%s/%s_%s.pkl' %(concept_norm_files_path, ontology, filename.replace('.txt', ''), 'updated'))
                pmc_tokenized_file_df.to_csv('%s%s/%s_%s.tsv' %(concept_norm_files_path, ontology, filename.replace('.txt', ''), 'updated'), '\t')


    return ontology_dict





def output_all_files(concept_norm_files_path, ontology, ontology_dict, filename_combo_list, disc_error_output_file):


    ##set up files for outputs:
    combo_src_file = open('%s%s/%s_%s.txt' %(concept_norm_files_path, ontology, ontology, filename_combo_list[0]), 'w+')
    combo_src_file_char = open('%s%s/%s_%s_%s.txt' %(concept_norm_files_path, ontology, ontology, filename_combo_list[0], 'char'), 'w+')
    combo_link_file = open('%s%s/%s_%s.txt' % (concept_norm_files_path, ontology, ontology, filename_combo_list[1]), 'w+')


    # combo_link_mention_ids_file = open('%s%s/%s_%s.txt' %(concept_norm_files_path, ontology, ontology,  filename_combo_list[1]), 'w+')
    # combo_link_sent_nums_files = open('%s%s/%s_%s.txt' %(concept_norm_files_path, ontology, ontology, filename_combo_list[2]), 'w+')




    ##ONTOLOGY_DICT - pmc_mention_id -> [sentence_num, [word], [(word_indices)], span_model, [biotag]]
    disc_error_dict = {} #span_model -> count
    for pmc_mention_id in ontology_dict.keys():
        # print(ontology_dict[pmc_mention_id])
        sentence_num = ontology_dict[pmc_mention_id][0]
        word_list = ontology_dict[pmc_mention_id][1] #list of words
        word_indices_list = ontology_dict[pmc_mention_id][2]
        span_model = ontology_dict[pmc_mention_id][3]

        if len(word_list) != len(word_indices_list):
            print(len(word_list), len(word_indices_list))
            print(word_list)
            print(word_indices_list)
            raise Exception('ERROR WITH COLLECTING ALL WORDS AND INDICES!')

        ##put the concept together:
        #a concept based on O-
        if word_list == ['...']:
            # discontinuity_error_count += 1
            if disc_error_dict.get(span_model):
                disc_error_dict[span_model] += 1
            else:
                disc_error_dict[span_model] = 1
            ##skip this because no concept!
        else:
            updated_word = ''
            updated_word_indices_list = [] #[(start,end)]
            disc_sign = False
            for i, w in enumerate(word_list):
                ##I is first with no B
                if i == 0: #always take the first word to start
                    updated_word += '%s' %w
                    updated_word_indices_list += [word_indices_list[i]]
                    if w == '...':
                        disc_sign = True

                elif w == '...' and not disc_sign:
                    updated_word += ' %s' %w
                    updated_word_indices_list += [word_indices_list[i]]
                    disc_sign = True

                elif w != '...':
                    # if i == 1 and updated_word.endswith('...'):
                    #     updated_word += '%s' %w
                    # else:
                    updated_word += ' %s' %w

                    updated_word_indices_list += [word_indices_list[i]]
                    disc_sign = False
                else:
                    # print('GOT HERE!!')
                    pass

            # if len(updated_word.split(' ')) not in (len(updated_word_indices_list), len(updated_word_indices_list)-1):
            if len(updated_word.split(' ')) != len(updated_word_indices_list):
                raise Exception('ERROR WITH UPDATING THE WORD TO GET THE FULL CONCEPT WITH INDICES!')

            ##link files so we know where the final output goes
            combo_link_file.write('%s\t%s\t' %(pmc_mention_id, sentence_num))
            word_indices_output = ''
            for (s,e) in updated_word_indices_list:
                word_indices_output += '%s %s;' %(s,e)
            word_indices_output = word_indices_output[:len(word_indices_output)-1] ##get rid of the last ;
            combo_link_file.write('%s\t%s\t%s\n' %(word_indices_output, updated_word, span_model))

            # combo_link_mention_ids_file.write('%s\n' %pmc_mention_id)
            # combo_link_sent_nums_files.write('%s\n' %sentence_num)

            ##string files for input to the seq-to-seq algorithms for concept normalization
            combo_src_file.write('%s\n' %updated_word)

            ##character word
            char_word = ''
            # print([word])
            for c in updated_word:

                char_word += '%s ' %c

            char_word = char_word[:len(char_word)-1] ##cut off the last space
            combo_src_file_char.write('%s\n' %char_word)

    # print('DISCONTINUITY ERROR COUNT:', discontinuity_error_count)
    for sm in disc_error_dict.keys():
        disc_error_output_file.write('%s\t%s\n' % (sm, disc_error_dict[sm]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-ontologies', type=str, help='a list of ontologies to use delimited with ,')
    parser.add_argument('-results_span_detection_path', type=str,
                        help='the file path to the results of the span detection models')
    parser.add_argument('-concept_norm_files_path', type=str, help='the file path for the concept norm files')
    parser.add_argument('-evaluation_files', type=str, help='a list of the files to be evaluated delimited with ,')

    args = parser.parse_args()



    # ontologies = ['CHEBI', 'CL', 'GO_BP', 'GO_CC', 'GO_MF', 'MOP', 'NCBITaxon', 'PR', 'SO', 'UBERON']

    # results_span_detection_path = '/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/Evaluation_Files/Results_span_detection/'



    # concept_norm_files_path = '/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/Evaluation_Files/Concept_Norm_Files/'


    ontologies = args.ontologies.split(',')
    evaluation_files = args.evaluation_files.split(',')



    for ontology in ontologies:
        # if ontology == 'CHEBI':
        print('PROGRESS:', ontology)
        ontology_dict = {}
        disc_error_output_file = open('%s%s/%s_DISC_ERROR_SUMMARY.txt' %(args.concept_norm_files_path, ontology, ontology), 'w+')
        disc_error_output_file.write('%s\t%s\n' %('MODEL', 'NUM DISCONTINUITY ERRORS'))


        #ONTOLOGY_DICT - pmc_mention_id -> [sentence_num, word, [(word_indices)], span_model]
        ontology_dict = preprocess_data(args.results_span_detection_path, ontology, ontology_dict, args.concept_norm_files_path, evaluation_files)



        # od_indices = [1, 2, 3, -1, 0]


        filename_combo_list = ['combo_src_file', 'combo_link_file']

        ##TODO: make all of them lowercase and uniform! also duplicates!
        output_all_files(args.concept_norm_files_path, ontology, ontology_dict, filename_combo_list, disc_error_output_file)

    print('PROGRESS: FINISHED CONCEPT NORMALIZATION PROCESSING FOR ALL FILES!')