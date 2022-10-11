import os
import re
# import gzip
import argparse
import numpy as np

import pandas as pd
# from nltk.tokenize import TreebankWordTokenizer
# from nltk.tokenize import WhitespaceTokenizer
# from nltk.tokenize import WordPunctTokenizer
# import copy
# from sklearn_crfsuite import CRF
# import sklearn_crfsuite
from sklearn import model_selection
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile, f_classif
# import eli5
# from IPython.display import display
# from itertools import chain
#
# import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import sklearn_crfsuite
# from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import time
# from sklearn.grid_search import RandomizedSearchCV
import joblib
import pickle
import ast
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, multi_gpu_model
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Model, Input, Sequential, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Conv1D, concatenate, \
    SpatialDropout1D, GlobalMaxPooling1D, Lambda
from keras.layers.merge import add
##TODO: need keras_contrib for CRF!
# from keras_contrib.layers import CRF #interacts with sklearn CRF #https://medium.com/@kegui/how-to-install-keras-contrib-7b75334ab742
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K

import math
import h5py

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False

        ##columns: PMCID	SENTENCE_NUM	SENTENCE_START	SENTENCE_END	WORD	POS_TAG	WORD_START	WORD_END
        agg_func = lambda s: [(w, p) for w, p in zip(s["WORD"].values.tolist(), s["POS_TAG"].values.tolist())]
        agg_func_rest = lambda t: [(a,b,c,d,e,f) for a,b,c,d,e,f in zip(t["PMCID"].values.tolist(),t["SENTENCE_NUM"].values.tolist(), t["SENTENCE_START"].values.tolist(), t["SENTENCE_END"].values.tolist(), t["WORD_START"].values.tolist(), t["WORD_END"].values.tolist())]

        # reindex in terms of the order of the sentences
        correct_indices = self.data.SENTENCE_NUM.unique()  # sentence nums in order (list)

        ##feature information
        self.grouped = self.data.groupby("SENTENCE_NUM", sort=False).apply(agg_func)
        # self.grouped.reindex(correct_indices, level="SENTENCE_NUM")
        self.sentences = [s for s in self.grouped]



        ##the rest of the sentence information
        self.grouped_rest = self.data.groupby("SENTENCE_NUM", sort=False).apply(agg_func_rest)
        # self.grouped_rest.reindex(correct_indices, level="SENTENCE_NUM")
        self.sentence_info = [t for t in self.grouped_rest]

        # print(self.sentences[290:])
        # print(self.sentence_info[290:])



        # raise Exception('HOLD!')





def load_data(tokenized_file_path, filename, all_sentences, all_sentence_info, excluded_files, ontology):
    all_pmcid_list = []
    # valid_filename = False
    # for root, directories, filenames in os.walk(tokenized_file_path):
    #     for filename in sorted(filenames):
    ##find the correct tokenized files to take
    # if filename.endswith('.pkl') and (filename.replace('.pkl', '') in excluded_files or filename.replace('.nxml.gz.pkl', '') in excluded_files):
    #     valid_filename = True
    # elif 'covid' == ontology.lower() and filename.endswith('.pkl'):
    #     valid_filename = True
    # else:
    #     valid_filename = False
    #
    # ##take all the tokenized files
    # if valid_filename:
    # print(root)
    # print(filename)
    all_pmcid_list += [filename.replace('.pkl','')]

    ##columns = ['PMCID', 'SENTENCE_NUM', 'SENTENCE_START', 'SENTENCE_END', 'WORD', 'POS_TAG', 'WORD_START', 'WORD_END']

    pmc_tokenized_file = pd.read_pickle(tokenized_file_path+filename)
    # print(pmc_tokenized_file[0])

    getter = SentenceGetter(pmc_tokenized_file)
    # print(len(getter.sentences))
    # print(type(getter.sentences))
    all_sentences += getter.sentences
    all_sentence_info += getter.sentence_info
    # print(len(all_sentences))
    return all_sentences, all_sentence_info, all_pmcid_list

def regex_annotations(all_regex, sentence):
    ##input a sentence and all lexical cues and output if the sentence has some or not - only for negative sentences!


    emoji_info = []  # emoji start index
    neutral_emoji = u"\U0001F610"

    if neutral_emoji in sentence:
        # print(pmc_full_text.count(neutral_emoji))
        emoji_count = sentence.count(neutral_emoji)
        emoji_start = 0

        for i in range(emoji_count):
            emoji_info += [sentence.index(neutral_emoji, emoji_start)]
            emoji_start = emoji_info[-1] + 1




    # ##SAVE THE OCCURRENCES PER ONTOLOGY TERM
    # all_occurrence_dict = {} #(ontology_cue, ignorance_type) -> [regex, occurrences_list]

    # all_occurrence_ontology_cue = []
    # all_occurrence_ignorance_type = []
    # all_occurrence_regex_cue = []
    # all_occurrence_cue_occurrence_list = []

    for regex_cue in all_regex:

        ##find the start, end of the cue
        cue_occurrence_list = [(m.start(), m.end()) for m in re.finditer(regex_cue.replace('_', ' ').lower(),
                                                                         sentence.lower())]  ##all lowercase - a list of tuples: [(492, 501), (660, 669), (7499, 7508), (13690, 13699), (17158, 17167), (20029, 20038), (20219, 20228), (20279, 20288), (28148, 28157)]

        # print('OCCURRENCES: ', cue_occurrence_list)

        ##PREPROCESS THE CUE_OCCURRENCE_LIST
        updated_cue_occurrence_list = []

        for span_num in cue_occurrence_list:
            start = span_num[0]
            end = span_num[1]

            adding_count = 0
            ##check to make sure it has the correct information:
            if emoji_info:
                for e in emoji_info:
                    if start > e:
                        adding_count += 1
                    else:
                        break

            start += adding_count
            end += adding_count

            ##TODO: UPDATE HERE FOR PREPROCESSING!
            # if adding_count == 0:
            ##A) IS/OR/IF/HERE/HOW/EVEN WITHIN A WORD: GET RID OF IT
            # or 'is'==regex_cue or 'if'==regex_cue  or 'even' in regex_cue  or 'here' == regex_cue or 'how' == regex_cue or 'can' == regex_cue or 'weight' == regex_cue or 'issue' == regex_cue or 'view' == regex_cue
            if '}or' in regex_cue or '_or' in regex_cue or '}if' in regex_cue or 'here.' in regex_cue or regex_cue in ['is', 'if', 'even', 'here', 'how', 'can', 'weight', 'issue', 'view', 'call']:
                try:
                    ##sometimes this errors due to the start being the beginning
                    if sentence[start - 1 - adding_count].isalpha() or sentence[end - adding_count].isalpha():  # end index not included
                        # print(regex_cue, ' WITHIN ',pmc_full_text[start-5:end+5])
                        pass
                    else:
                        updated_cue_occurrence_list += [(start, end)]
                except IndexError:
                    #if an error then take the cue
                    updated_cue_occurrence_list += [(start, end)]
            else:
                updated_cue_occurrence_list += [(start, end)]


        ##found cues in the sentence - doesn't matter which cues
        if updated_cue_occurrence_list:
            # print('true negative example', sentence, regex_cue)
            return True

    #no cues found in sentence
    return False


def load_pmcid_sentence_data(ontology, pmcid_sentence_file_path, filename, excluded_files, all_regex, gold_standard, tokenized_file_path):
    # print(pmcid_sentence_file_path)
    # print(gold_standard, type(gold_standard))
    sentence_filename = filename.replace('.pkl', '_sentence_info.txt')

    # if ontology.upper() == 'IGNORANCE' or ontology.upper() == 'COVID':
    ##all files to return
    all_sents_ids = []
    all_sents = []
    all_pmcids = []
    all_sent_labels = []


    preprocess_sent_ids = []
    preprocess_sent = []
    preprocess_sent_labels = []
    # print(pmcid_sentence_file_path)
    # valid_filename = False
    # for root, directories, filenames in os.walk('%s' % pmcid_sentence_file_path):
    #     for filename in sorted(filenames):
            ##find the correct file
    if sentence_filename.endswith('sentence_info.txt') and filename.split('.nxml')[0] in excluded_files:
        valid_filename = True
        # print(filename)
        # print('got here')
        # print(gold_standard)
    elif sentence_filename.endswith('sentence_info.txt') and 'covid' == ontology.lower():
        valid_filename = True

    elif sentence_filename.endswith('sentence_info.txt'):
        # print(filename)
        valid_filename = True
        pmcid_sentence_file_path = tokenized_file_path.replace('Tokenized_Files/', pmcid_sentence_file_path)


    else:
        valid_filename = False

    #open the correct files to do stuff with
    if valid_filename:
        with open(pmcid_sentence_file_path + sentence_filename, 'r') as pmcid_sentence_file:
            next(pmcid_sentence_file)  # header: PMCID	SENTENCE_NUMBER	SENTENCE	SENTENCE_INDICES
            for line in pmcid_sentence_file:
                # print(gold_standard)
                if gold_standard:
                    # print('got here')
                    (pmcid, sentence_number, sentence_list, sentence_indices,
                     ontology_concepts_ids_list) = line.split('\t')

                    ontology_concepts_ids_list = ast.literal_eval(ontology_concepts_ids_list)
                    # print(ontology_concepts_ids_list)
                else:
                    (pmcid, sentence_number, sentence_list, sentence_indices) = line.split('\t')


                sentence = ast.literal_eval(sentence_list)[0]

                if all_regex:
                    true_negative_example = regex_annotations(all_regex, sentence) #true or false based on regex in it or not
                else:
                    true_negative_example = True #TODO: default that all are good!

                ##full set
                all_sents_ids += ['%s_%s_%s' % (pmcid, sentence_number, sentence_indices.strip('\n'))]
                all_pmcids += [pmcid]

                all_sents += [sentence]

                if gold_standard:

                    ##the ontology labels - 1 = ignorance, 0 = not
                    if len(ontology_concepts_ids_list) > 0:
                        all_sent_labels += [1]
                    else:
                        # print('got here!')
                        all_sent_labels += [0]

                ##preprocessed set:
                if true_negative_example:
                    # print('got here!')
                    preprocess_sent_ids += ['%s_%s_%s' % (pmcid, sentence_number, sentence_indices.strip('\n'))]

                    preprocess_sent += [sentence]

                    if gold_standard:
                        ##the ontology labels - 1 = ignorance, 0 = not
                        if len(ontology_concepts_ids_list) > 0:
                            preprocess_sent_labels += [1]
                        else:
                            # print('got here!')
                            preprocess_sent_labels += [0]


                else:
                    # print('no cues!!')
                    pass

    return all_sents_ids, all_sents, all_sent_labels, preprocess_sent_ids, preprocess_sent, preprocess_sent_labels, all_pmcids

    # else:
    #     raise Exception('ERROR: ONLY WORKING FOR THE IGNORANCE ONTOLOGY RIGHT NOW! NEED TO ADD MORE FUNCTIONALITY!')


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


def pred2label(pred, idx2biotag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2biotag[p_i].replace("ENDPAD", "O"))
        out.append(out_i)
    return out


def fake_loss(y_true, y_pred):
    return 0

class SentenceGetter_training(object):
    def __init__(self, data):
        self.n_sent_t = 1
        self.data_t = data
        self.empty_t = False
        agg_func_t = lambda s: [(w, p, t) for w, p, t in zip(s["WORD"].values.tolist(),
                                                           s["POS_TAG"].values.tolist(),
                                                           s["BIO_TAG"].values.tolist())]
        self.grouped_t = self.data_t.groupby("SENTENCE_NUM", sort=False).apply(agg_func_t)
        self.sentences_t = [s for s in self.grouped_t]


def load_data_training(tokenized_file_path, filename, ontology, all_ontology_sentences, excluded_files):
    # for root, directories, filenames in os.walk(tokenized_file_path + ontology + '/'):
    #     for filename in sorted(filenames):

            ##save 2 files to fully evaluate on later
            # if filename.endswith('.pkl') and 'full' not in filename and 'mention_id_dict' not in filename and (filename.replace('.pkl','') in excluded_files or filename.replace('.nxml.gz.pkl', '') in excluded_files):

    ##columns = ['PMCID', 'SENTENCE_NUM', 'SENTENCE_START', 'SENTENCE_END', 'WORD', 'POS_TAG', 'WORD_START', 'WORD_END', 'BIO_TAG', 'PMC_MENTION_ID', 'ONTOLOGY_CONCEPT_ID', 'ONTOLOGY_LABEL']
    # print(filename)

    pmc_tokenized_file = pd.read_pickle(tokenized_file_path+ontology+'/'+filename)

    getter = SentenceGetter_training(pmc_tokenized_file)
    # print(len(getter.sentences))
    # print(type(getter.sentences))
    all_ontology_sentences += getter.sentences_t
    # print(len(all_ontology_sentences))
    # else:
    #     pass
    return all_ontology_sentences

def LSTM_prediction_report(output_path, ontology, biotags, closer_biotags, filename):
    ##reads in the prediction file and outputs metrics

    ##metrics to report for prediction
    total_word_count = 0
    endpad_count = 0
    # exact_match = 0
    # true_mismatch = 0 #O instead of a B or I
    # closer_mismatch = 0 #B->I or I->B

    biotag_total_dict = {} #dict from: biotag_true -> [total_true_count, total_pred_count, exact_match, closer_mismatch, true_mismatch]

    ##initialize biotag_dict
    for biotag in biotags:
        biotag_total_dict[biotag] = [0,0,0,0,0]


    with open('%s%s/%s_eval_predictions.txt' % (output_path, ontology, filename.replace('.h5','')), 'r+') as pred_file:
        next(pred_file) #heading line: Word True Pred
        for line in pred_file:
            (word, true, pred) = line.replace('\n', '').split('\t')
            ##total word count
            if word:
                total_word_count += 1
                if word == 'ENDPAD':
                    endpad_count += 1
                else:
                    pass
            else:
                pass

            biotag_total_dict[true][0] += 1
            biotag_total_dict[pred][1] += 1

            if true == pred:
                ##exact match
                biotag_total_dict[true][2] += 1
            elif true in closer_biotags and pred in closer_biotags:
                ##closer match
                biotag_total_dict[true][3] += 1
            else:
                ##true mismatch
                biotag_total_dict[true][4] += 1


    with open('%s%s/%s_pred_report.txt' % (output_path, ontology, filename.replace('.h5', '')), 'w+') as pred_report_file:
        pred_report_file.write('%s\t%s\t%s\n' %('ONTOLOGY', 'TOTAL WORD COUNT', 'TOTAL ENDPAD COUNT'))
        pred_report_file.write('%s\t%s\t%s\n' %(ontology, total_word_count, endpad_count))

        ## biotag_total_dict = {} #dict from: biotag_true -> [total_true_count, total_pred_count, exact_match, closer_mismatch, true_mismatch]
        pred_report_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %('BIOTAG', 'TOTAL TRUE COUNT', '% TRUE COUNT', 'TOTAL PRED COUNT', '% PRED COUNT', 'NUM EXACT MATCH', '% EXACT MATCH', 'NUM CLOSER MATCH (B,I)', '% CLOSER MATCH', 'NUM TRUE MISMATCH', '% TRUE MISMATCH'))

        ##output the information per biotag
        for biotag in biotags:
            (total_true_count, total_pred_count, exact_match, closer_mismatch, true_mismatch) = biotag_total_dict[biotag]

            pred_report_file.write('%s\t%s\t%.4f\t%s\t%.4f\t%s\t%.4f\t%s\t%.4f\t%s\t%.4f\n' %(biotag,total_true_count, float(total_true_count)/float(total_word_count), total_pred_count, float(total_pred_count)/float(total_true_count), exact_match, float(exact_match)/float(total_true_count), closer_mismatch, float(closer_mismatch)/float(total_true_count), true_mismatch, float(true_mismatch)/float(total_true_count)))



def output_span_detection_results(all_pmcid_list, pmcid_starts_dict, output_path, filename, output_results, output_results_path):
    ##output the results in BIO format for concept normalization
    # ontology = filename.split('_')[0] ##error on GO_BP, GO_MF, GO_CC
    word_index = 0
    for pmcid in all_pmcid_list:
        # print('\tarticle:', pmcid)
        [pmcid_start_index, pmcid_end_index] = pmcid_starts_dict[pmcid]
        # print(pmcid_start_index, pmcid_end_index)

        # ##update the order so that it is in integer order of sentences not alphabetical order
        # updated_indices = {} #a dictionary of the correct order of sentence number: index -> information
        # max_sentence_num = 0
        # for o in output_results[pmcid_start_index:pmcid_end_index]:
        #     sentence_num = int(o[1].split('_')[-1])
        #     if updated_indices.get(sentence_num):
        #         updated_indices[sentence_num] += [o]
        #     else:
        #         updated_indices[sentence_num] = [o]
        #
        #     max_sentence_num = max(max_sentence_num, sentence_num)



        # if word_index == pmcid_starts_dict[pmcid]:
        #     continue
        # else:
        # print(output_results_path, pmcid)
        with open('%s_%s.txt' % (output_results_path, pmcid),'w+') as output_results_file:
            output_results_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
            'PMCID', 'SENTENCE_NUM', 'SENTENCE_START', 'SENTENCE_END', 'WORD', 'POS_TAG', 'WORD_START', 'WORD_END',
            'BIO_TAG', 'PMC_MENTION_ID', 'ONTOLOGY_CONCEPT_ID', 'ONTOLOGY_LABEL'))
            ##for each word in the sentence - output the information
            for o in output_results[pmcid_start_index:pmcid_end_index]:
            # for j in range(max_sentence_num+1):
            #     for o in updated_indices[j]:
                    # print(o)
                ##for column output the information
                for i, a in enumerate(o):
                    # print(a)
                    if i == len(o) - 1:
                        output_results_file.write('%s\n' % a)
                    else:
                        output_results_file.write('%s\t' % a)
                word_index += 1

                ##break here if the word_index is higher and we want to move to the next article
                if word_index == pmcid_end_index:
                    break
                else:
                    pass
    # raise Exception('HOLD!')


# def LSTM_collect_hyperparameters(ontology, save_models_path):
#     with open('%s%s/models/%s_LSTM_hyperparameterization.txt' %(save_models_path, ontology, ontology), 'r') as LSTM_hyperparamers_file:
#         ##header: Ontology	OPTIMIZER	LOSS	NEURONS	EPOCHS	BATCHES	BIO- MACRO F1 MEASURE	FULL MACRO F1 MEASURE	BIO- WEIGHTED F1 MEASURE	FULL WEIGHTED F1 MEASURE
#         for line in LSTM_hyperparamers_file:
#             if line.startswith(ontology):
#                 ontology, optimizer, loss, neurons, epochs, batch_size = line.split('\t')[:6]
#
#                 neurons = int(neurons)
#                 epochs = int(epochs)
#                 batch_size = int(batch_size)
#
#         return optimizer, loss, neurons, epochs, batch_size


def LSTM_collect_hyperparameters(ontology, save_models_path, algo_tuning):
    with open('%s%s/%s_%s_hyperparameterization.txt' %(save_models_path, ontology, ontology, algo_tuning), 'r') as LSTM_hyperparamers_file:
        ##header: Ontology	OPTIMIZER	LOSS	NEURONS	EPOCHS	BATCHES	BIO- MACRO F1 MEASURE	FULL MACRO F1 MEASURE	BIO- WEIGHTED F1 MEASURE	FULL WEIGHTED F1 MEASURE
        for line in LSTM_hyperparamers_file:
            if line.startswith(ontology):
                ontology, optimizer, loss, neurons, epochs, batch_size = line.split('\t')[:6]

                neurons = int(neurons)
                epochs = int(epochs)
                batch_size = int(batch_size)

        return optimizer, loss, neurons, epochs, batch_size


def ElmoEmbedding(x, elmo_model, batch_size, max_len):
    ##create a function that takes a sequence of strings and outputs the ELMo embedding (1024D)
    # elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    return elmo_model(inputs={
        "tokens": tf.squeeze(tf.cast(x, tf.string)),
        "sequence_len": tf.constant(batch_size * [max_len])
    },
        signature="tokens",
        as_dict=True)["elmo"]

def pred2label_LSTM_ELMO(pred, idx2biotag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            if idx2biotag[p_i] != 'ENDPAD':
                out_i.append(idx2biotag[p_i])
        out.append(out_i)
    return out


def run_models(tokenized_file_path, ontology, save_models_path, output_path, excluded_files, gold_standard, algos, pmcid_sentence_file_path,  all_lcs_path, gpu_count=1):
    # https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/
    # https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#evaluation

    ##only take sentences with a cue in it
    # create all_lcs_dict: lc -> [regex, ignorance_type]
    all_regex = []
    if all_lcs_path:
        with open('%s' % all_lcs_path, 'r') as all_lcs_file:
            next(all_lcs_file)
            for line in all_lcs_file:
                all_regex += [line.split('\t')[1]]

    if 'LSTM' in algos and gold_standard.lower() == 'true':
        local_eval_files_creport_LSTM = open('%s%s/%s_LSTM_local_eval_files_classification_report.txt' % (output_path, ontology, ontology),
                  'w+')
        local_eval_files_creport_LSTM.write('%s\t%s\t%s\t%s\t%s\n' %('PMCID', 'TYPE', 'PRECISION', 'RECALL', 'F1_MEASURE'))

    if 'LSTM_CRF' in algos and gold_standard.lower() == 'true':
        local_eval_files_creport_LSTM_CRF = open(
            '%s%s/%s_LSTM_CRF_local_eval_files_classification_report.txt' % (output_path, ontology, ontology),
            'w+')
        local_eval_files_creport_LSTM_CRF.write('%s\t%s\t%s\t%s\t%s\n' % ('PMCID', 'TYPE', 'PRECISION', 'RECALL', 'F1_MEASURE'))

    if 'CHAR_EMBEDDINGS' in algos and gold_standard.lower() == 'true':
        local_eval_files_creport_char_embeddings_LSTM = open(
            '%s%s/%s_char_embeddings_LSTM_local_eval_files_classification_report.txt' % (output_path, ontology, ontology),
            'w+')
        # local_eval_files_creport_char_embeddings_LSTM.write(
        #     '%s\t%s\t%s\t%s\t%s\n' % ('PMCID', 'TYPE', 'PRECISION', 'RECALL', 'F1_MEASURE'))


        #TODO: add these in!!

        # local_eval_files_creport_char_embeddings_LSTM_CRF = open(
        #     '%s%s/%s_char_embeddings_LSTM_CRF_local_eval_files_classification_report.txt' % (
        #     output_path, ontology, ontology),
        #     'w+')
        # local_eval_files_creport_char_embeddings_LSTM_CRF.write(
        #     '%s\t%s\t%s\t%s\t%s\n' % ('PMCID', 'TYPE', 'PRECISION', 'RECALL', 'F1_MEASURE'))

    if 'LSTM_ELMO' in algos and gold_standard.lower() == 'true':
        local_eval_files_creport_ELMO_LSTM = open(
            '%s%s/%s_LSTM_ELMO_local_eval_files_classification_report.txt' % (
            output_path, ontology, ontology),
            'w+')
        print('PROGRESS: classification report done!')



    ##gather all data for all algorithms
    for root, directories, filenames in os.walk(tokenized_file_path):
        for filename in sorted(filenames):



            if filename.endswith('.pkl') and (
                    filename.replace('.pkl', '') in excluded_files or filename.replace('.nxml.gz.pkl','') in excluded_files):
                valid_filename = True
            elif 'covid' == ontology.lower() and filename.endswith('.pkl'):
                valid_filename = True
            elif excluded_files[0].lower() == 'all' and filename.endswith('.pkl'):
                valid_filename = True
            else:
                valid_filename = False

            ##take all the tokenized files
            if valid_filename:
                print('PROGRESS: starting file: ', filename)
                pmcid = filename.replace('.pkl', '')

                # print('got here!')


                ##initialize all sentences
                all_sentences = []
                all_sentence_info = []

                ##load the data for all sentences in the evaluation data - loop in load_data
                all_sentences, all_sentence_info, all_pmcid_list = load_data(tokenized_file_path, filename, all_sentences, all_sentence_info, excluded_files, ontology)
                # print(all_sentences[0], all_sentence_info[0], all_pmcid_list[0])
                print('NUMBER OF SENTENCES TO EVALUATE ON:', len(all_sentences))


                if len(all_sentences) != len(all_sentence_info):
                    raise Exception('ERROR WITH GATHERING SENTENCE INFORMATION!')
                else:
                    pass




                ##get all the sentences with no gold standard from the pmcid_sentence_list
                if pmcid_sentence_file_path:

                    all_sents_ids, all_sents, all_sent_labels, preprocess_sent_ids, preprocess_sent, preprocess_sent_labels, all_pmcids = load_pmcid_sentence_data(ontology, pmcid_sentence_file_path, filename, excluded_files, all_regex, None, tokenized_file_path)
                    print('NUMBER OF REGEX SENTENCES TO EVALUATE ON:', len(preprocess_sent))
                    # print(all_sents[:10])
                    if len(all_sents) != len(all_sentences):
                        raise Exception('ERROR WITH LENGTHS OF SENTENCES MATCHING BETWEEN PMCIDS FILES AND BIO FILES!')


                ##Naive bayes preprocessed stuff to predict on
                if 'NAIVE_BAYES' in algos:
                    vectorizer_path = '%s%s/%s_%s_%s.pkl' %(save_models_path, ontology, ontology, 'NAIVE_BAYES', 'vectorizer')
                    vectorizer = pickle.load(open(vectorizer_path, 'rb'))

                    selector_path = '%s%s/%s_%s_%s.pkl' %(save_models_path, ontology, ontology, 'NAIVE_BAYES', 'selector')
                    selector = pickle.load(open(selector_path, 'rb'))

                    features_eval = vectorizer.transform(all_sents)
                    features_eval = selector.transform(features_eval).toarray()


                ##CRF preprocessed stuff:
                if 'CRF' in algos:

                    X_eval_crf = [sent2features(s) for s in all_sentences]

                ##LSTM preprocessed stuff:
                if 'LSTM' in algos:
                    ##LSTM preprocessed stuff:
                    #load in the word dict and biotag dict
                    with open('%s%s/%s_LSTM_word2idx.pkl' %(save_models_path, ontology, ontology), 'rb') as word2idx_output:
                        word2idx_LSTM = pickle.load(word2idx_output)

                    all_words_set_LSTM = list(word2idx_LSTM.keys()) ##all words used in

                    with open('%s%s/%s_LSTM_biotag2idx.pkl' % (save_models_path, ontology, ontology), 'rb') as biotag2idx_output:
                        biotag2idx_LSTM = pickle.load(biotag2idx_output)

                    all_biotags_set_LSTM = list(biotag2idx_LSTM.keys())

                    #load in the sentence length for padding
                    with open('%s%s/%s_LSTM_max_length.txt' %(save_models_path, ontology, ontology), 'r') as max_length_file:
                        for line in max_length_file:
                            if 'ontology' in line:
                                if line.split('\t')[1].strip('\n') != ontology:
                                    print('line:', line)
                                    print('ontology', ontology)
                                    print([line.split('\t')[1]])
                                    raise Exception('ERROR WITH ONTOLOGY MATCHING OF MAX LENGTH FILES!')
                                else:
                                    pass
                            elif 'sentence length' in line:
                                max_sentence_length_LSTM = int(line.split('\t')[1].strip('\n'))


                    #add in the scripts to preprocess the sentences using the stuff above
                    ##Pad the sentences and biotags by converting them to a sequence of numbers (word2idx, biotag2idx)
                    # X_eval_LSTM = [[word2idx[w[0]] for w in s] for s in all_sentences]
                    X_eval_LSTM = []
                    unseen_original_words_LSTM = []
                    for s in all_sentences:
                        sentence_vector = []
                        for w in s:
                            # print(w[0])
                            try:
                                sentence_vector += [word2idx_LSTM[w[0]]]
                                # print(type(word2idx[w[0]])) #integers
                            ##out of vocabulary/unseen words get a special tag
                            except KeyError:
                                # print(type(word2idx['OOV_UNSEEN'])) #integers
                                sentence_vector += [word2idx_LSTM['OOV_UNSEEN']]
                                unseen_original_words_LSTM += [w[0]]
                        # print(sentence_vector)
                        X_eval_LSTM += [sentence_vector]
                    # print(X_eval_LSTM)

                    X_eval_LSTM = pad_sequences(maxlen=max_sentence_length_LSTM, sequences=X_eval_LSTM, padding="post", value=word2idx_LSTM['ENDPAD']) ##add ENDPAD up to the max sentence length
                    # print(X[1])

                ##LSTM_CRF preprocessed stuff
                if 'LSTM_CRF' in algos:
                    with open('%s%s/%s_LSTM_CRF_word2idx.pkl' %(save_models_path, ontology, ontology), 'rb') as word2idx_output:
                        word2idx_LSTM_CRF = pickle.load(word2idx_output)

                    all_words_set_LSTM_CRF = list(word2idx_LSTM_CRF.keys()) ##all words used in - ADDED ONE FOR WORD2IDX
                    # print(len(all_words_set_LSTM_CRF))
                    # print(all_words_set_LSTM_CRF[0])
                    # print(all_words_set_LSTM_CRF[1])
                    # print(all_words_set_LSTM_CRF[21845]) ##ENDPAD
                    # print(all_words_set_LSTM_CRF[21846]) ##OOV_UNSEEN
                    # raise Exception('hold!')

                    with open('%s%s/%s_LSTM_CRF_biotag2idx.pkl' % (save_models_path, ontology, ontology), 'rb') as biotag2idx_output:
                        biotag2idx_LSTM_CRF = pickle.load(biotag2idx_output)

                    all_biotags_set_LSTM_CRF = list(biotag2idx_LSTM_CRF.keys())

                    #load in the sentence length for padding
                    with open('%s%s/%s_LSTM_CRF_max_length.txt' %(save_models_path, ontology, ontology), 'r') as max_length_file:
                        for line in max_length_file:
                            if 'ontology' in line:
                                if line.split('\t')[1].strip('\n') != ontology:
                                    print('line:', line)
                                    print('ontology', ontology)
                                    print([line.split('\t')[1]])
                                    raise Exception('ERROR WITH ONTOLOGY MATCHING OF MAX LENGTH FILES!')
                                else:
                                    pass
                            elif 'sentence length' in line:
                                max_sentence_length_LSTM_CRF = int(line.split('\t')[1].strip('\n'))

                    ##Pad the sentences and biotags by converting them to a sequence of numbers (word2idx, biotag2idx)
                    # X_eval_LSTM_CRF = [[word2idx_LSTM_CRF[w[0]] for w in s] for s in all_sentences]
                    # X_eval_LSTM_CRF = pad_sequences(maxlen=max_sentence_length_LSTM_CRF, sequences=X, padding="post",
                    #                   value=word2idx_LSTM_CRF['ENDPAD'])  # should be -1 or -2?

                    X_eval_LSTM_CRF = []
                    unseen_original_words_LSTM_CRF = []
                    for s in all_sentences:
                        sentence_vector = []
                        for w in s:
                            # print(w[0])
                            try:
                                sentence_vector += [word2idx_LSTM_CRF[w[0]]]
                                # print(type(word2idx[w[0]])) #integers
                            ##out of vocabulary/unseen words get a special tag
                            except KeyError:
                                # print(type(word2idx['OOV_UNSEEN'])) #integers
                                sentence_vector += [word2idx_LSTM_CRF['OOV_UNSEEN']]
                                unseen_original_words_LSTM_CRF += [w[0]]
                        # print(sentence_vector)
                        X_eval_LSTM_CRF += [sentence_vector]
                    # print(X_eval_LSTM_CRF)

                    X_eval_LSTM_CRF = pad_sequences(maxlen=max_sentence_length_LSTM_CRF, sequences=X_eval_LSTM_CRF, padding="post",
                                                value=word2idx_LSTM_CRF['ENDPAD'])  ##add ENDPAD up to the max sentence length
                    # print(X[1])

                ##CHAR EMBEDDINGS preprocessed stuff
                if 'CHAR_EMBEDDINGS' in algos:
                    with open('%s%s/%s_char_embeddings_word2idx.pkl' %(save_models_path, ontology, ontology), 'rb') as word2idx_output:
                        word2idx_char_embeddings = pickle.load(word2idx_output) #w+2, 0 = ENDPAD, 1 = OOV_UNSEEN

                    all_words_set_char_embeddings = list(word2idx_char_embeddings.keys()) ##all words used in - ADDED ONE FOR WORD2IDX

                    idx2word_char_embeddings = {value: key for key, value in word2idx_char_embeddings.items()}
                    # print('word2idx info')
                    # print(word2idx_char_embeddings['ENDPAD'])
                    # print(word2idx_char_embeddings['OOV_UNSEEN'])

                    # print(idx2word_char_embeddings[0])
                    # print(idx2word_char_embeddings[1])

                    with open('%s%s/%s_char_embeddings_biotag2idx.pkl' % (save_models_path, ontology, ontology), 'rb') as biotag2idx_output:
                        biotag2idx_char_embeddings = pickle.load(biotag2idx_output)

                    all_biotags_set_char_embeddings = list(biotag2idx_char_embeddings.keys())

                    with open('%s%s/%s_char_embeddings_char2idx.pkl' % (save_models_path, ontology, ontology), 'rb') as char2idx_output:
                        char2idx_char_embeddings = pickle.load(char2idx_output) #c+2, 0 = ENDPAD, 1 = OOV_UNSEEN

                    all_char_set_char_embeddings = list(char2idx_char_embeddings.keys())
                    # print('char2idx info')
                    # print(char2idx_char_embeddings['ENDPAD'])
                    # print(char2idx_char_embeddings['OOV_UNSEEN'])

                    # raise Exception('HOLD!')


                    #load in the sentence length for padding
                    with open('%s%s/%s_char_embeddings_max_length.txt' %(save_models_path, ontology, ontology), 'r') as max_length_file:
                        for line in max_length_file:
                            if 'ontology' in line:
                                if line.split('\t')[1].strip('\n') != ontology:
                                    print('line:', line)
                                    print('ontology', ontology)
                                    print([line.split('\t')[1]])
                                    raise Exception('ERROR WITH ONTOLOGY MATCHING OF MAX LENGTH FILES!')
                                else:
                                    pass
                            elif 'sentence length' in line:
                                max_sentence_length_char_embeddings = int(line.split('\t')[1].strip('\n'))
                                print('max_sentence length', max_sentence_length_char_embeddings)

                    #max character length based on longest word
                    max_char_length_char_embeddings = max([len(w) for w in all_words_set_char_embeddings if w not in ['OOV_UNSEEN', 'ENDPAD']])  # the maximum number of characters in all words
                    print('max_char_length', max_char_length_char_embeddings)

                    ##Pad the sentences and biotags by converting them to a sequence of numbers (word2idx, biotag2idx)
                    # X_word = [[word2idx[w[0]] for w in s] for s in all_ontology_sentences]
                    # X_word = pad_sequences(maxlen=max_sentence_length, sequences=X_word, value=word2idx["ENDPAD"],
                    #                        padding='post',
                    #                        truncating='post')
                    X_eval_word_char_embeddings = []
                    unseen_original_words_char_embeddings = []
                    # print('sentence 65')
                    # print(len(all_sentences[65]))
                    # print(all_sentences[65])

                    for s in all_sentences:
                        sentence_vector = []
                        for w in s:
                            # print(w[0])
                            try:
                                sentence_vector += [word2idx_char_embeddings[w[0]]]
                                # print(type(word2idx[w[0]])) #integers
                            ##out of vocabulary/unseen words get a special tag
                            except KeyError:
                                # print(type(word2idx['OOV_UNSEEN'])) #integers
                                # if 'OOV_UNEEN' in all_words_set_char_embeddings:
                                #     sentence_vector += [word2idx_char_embeddings['OOV_UNEEN']]
                                # else:
                                sentence_vector += [word2idx_char_embeddings['OOV_UNSEEN']]
                                # print('unknown word idx:', word2idx_char_embeddings['OOV_UNSEEN'], type(word2idx_char_embeddings['OOV_UNSEEN']))
                                # print('unknown word', w[0])
                                # raise Exception('be still')

                                unseen_original_words_char_embeddings += [w[0]]
                        # print(sentence_vector)
                        X_eval_word_char_embeddings += [sentence_vector]
                    # print(X_eval_LSTM_CRF)
                    # print(len(X_eval_word_char_embeddings[65]))
                    # print(X_eval_word_char_embeddings[65])
                    indices_unknown = [i for i, x in enumerate(X_eval_word_char_embeddings[65]) if x == 1]
                    # raise Exception('hold!')

                    X_eval_word_char_embeddings = pad_sequences(maxlen=max_sentence_length_char_embeddings, sequences=X_eval_word_char_embeddings, value=word2idx_char_embeddings['ENDPAD'], padding="post", truncating='post')  ##add ENDPAD up to the max sentence length

                    # print([len(x) for x in X_eval_word_char_embeddings])
                    # print(len(X_eval_word_char_embeddings[65]))
                    # print(X_eval_word_char_embeddings[65])

                    ##gather all the characters for each sentence! - need to include the unknowns and create char embeddings for the unknown words - TODO!!
                    X_eval_char_char_embeddings = []
                    unseen_word_count_char = 0
                    print('length of unseen words', len(unseen_original_words_char_embeddings)) #, unseen_original_words_char_embeddings[:10])
                    unseen_original_chars_char_embeddings = []
                    # print(max_sentence_length_char_embeddings, max_char_length_char_embeddings)
                    for t, sentence_vector in enumerate(X_eval_word_char_embeddings):
                        sent_seq = []
                        for i in range(max_sentence_length_char_embeddings):
                            current_word_idx = sentence_vector[i]  # [0]
                            # print(type(current_word_idx)) # <class 'numpy.int32'>
                            # raise Exception('be still')
                            # if current_word_idx.item() == 1:
                            #     raise Exception('hold please')
                            # print('current word idx', current_word_idx)
                            current_word = idx2word_char_embeddings[current_word_idx]
                            # if current_word != 'ENDPAD':
                                # print('current word', current_word, type(current_word))
                            # raise Exception('hold!')
                            if current_word == 'OOV_UNSEEN':
                                # print(unseen_word_count_char)
                                unknown_word = unseen_original_words_char_embeddings[unseen_word_count_char]
                                # print('unknown word', unknown_word)

                                unseen_word_count_char += 1

                            word_seq = []
                            for j in range(max_char_length_char_embeddings):
                                if current_word == 'OOV_UNSEEN':


                                    try:
                                        # print(unknown_word[j])
                                        if char2idx_char_embeddings.get(unknown_word[j]):
                                            word_seq.append(char2idx_char_embeddings.get(unknown_word[j]))
                                        else:
                                            word_seq.append(char2idx_char_embeddings['OOV_UNSEEN'])


                                    except IndexError:
                                        word_seq.append(char2idx_char_embeddings.get("ENDPAD"))

                                    # raise Exception('HOLD please')
                                elif current_word == 'ENDPAD':
                                    # print('endpad got here from word vector!')
                                    word_seq.append(char2idx_char_embeddings.get("ENDPAD"))

                                else:
                                    try:
                                        if char2idx_char_embeddings.get(current_word[j]):
                                            word_seq.append(char2idx_char_embeddings.get(current_word[j]))
                                        else:
                                            word_seq.append(char2idx_char_embeddings['OOV_UNSEEN'])
                                        # print('current word idx', current_word_idx)
                                        # print('current word', current_word)
                                        # print('current char', current_word[j])
                                        # print('character success!')
                                    except IndexError:
                                        word_seq.append(char2idx_char_embeddings.get("ENDPAD"))
                                        # print('endpad got here from character vector length!')
                                if None in word_seq:
                                    print('info for wordseq error')
                                    print(i)
                                    print(all_sentences[i])
                                    print(current_word_idx)
                                    print(current_word)
                                    print('unknown word', unknown_word)

                                    print(word_seq)
                                    raise Exception('ERROR WITH CONVERTING WORD TO CHAR EMBEDDINGS!')
                            sent_seq.append(word_seq)
                            # if current_word != 'ENDPAD':
                            #     print(sent_seq)
                        X_eval_char_char_embeddings.append(np.array(sent_seq))
                        # print([x.shape for x in X_eval_char_char_embeddings])
                        # for x in X_eval_char_char_embeddings:
                        #     for a in x:
                        #         print(len(a))
                        # raise Exception('HOLD!')
                    # print(len(X_eval_char_char_embeddings[65]))
                    # print(X_eval_char_char_embeddings[65])


                    # for i in indices_unknown:
                    #     print(i)
                    #     print(all_sentences[65][i])
                    #     print(X_eval_word_char_embeddings[65][i])
                    #     print(X_eval_char_char_embeddings[65][i])

                    # raise Exception('hold')

                    # for sentence in all_sentences:
                    #     sent_seq = []
                    #     for i in range(max_sentence_length_char_embeddings):
                    #         word_seq = []
                    #         for j in range(max_char_length_char_embeddings):
                    #             print('char embedding', sentence[i][0][j])
                    #             try:
                    #                 word_seq.append(char2idx_char_embeddings.get(sentence[i][0][j]))
                    #             except:
                    #                 word_seq.append(char2idx_char_embeddings.get("ENDPAD"))
                    #         sent_seq.append(word_seq)
                    #     X_eval_char_char_embeddings.append(np.array(sent_seq))


                if 'LSTM_ELMO' in algos:
                    # load in the sentence length for padding
                    with open('%s%s/%s_LSTM_ELMO_max_length.txt' % (
                    save_models_path, ontology, ontology), 'r') as max_length_file:
                        for line in max_length_file:
                            if 'ontology' in line:
                                if line.split('\t')[1].strip('\n') != ontology:
                                    print('line:', line)
                                    print('ontology', ontology)
                                    print([line.split('\t')[1]])
                                    raise Exception('ERROR WITH ONTOLOGY MATCHING OF MAX LENGTH FILES!')
                                else:
                                    pass
                            elif 'sentence length' in line:
                                max_sentence_length_LSTM_ELMO = int(line.split('\t')[1].strip('\n'))
                                print('max_sentence length', max_sentence_length_LSTM_ELMO)

                    with open('%s%s/%s_LSTM_ELMO_biotag2idx.pkl' % (save_models_path, ontology, ontology), 'rb') as biotag2idx_output:
                        biotag2idx_LSTM_ELMO = pickle.load(biotag2idx_output)

                    all_biotags_set_LSTM_ELMO = list(biotag2idx_LSTM_ELMO.keys())


                    ##CONVERT ALL SENTENCES TO STRINGS WITH PADDING
                    X_eval_LSTM_ELMO = [[w[0] for w in s] for s in all_sentences] #take the word from each sentence and add the padding based on the max sentence length!

                    new_X_LSTM_ELMO = []
                    for seq in X_eval_LSTM_ELMO:
                        new_seq = []
                        for i in range(max_sentence_length_LSTM_ELMO):
                            # new_seq.append(seq[i])
                            try:
                                new_seq.append(seq[i])
                            except IndexError:
                                new_seq.append("__PAD__")  # '__PAD__' = the string padding to the max_sentence_length
                        new_X_LSTM_ELMO.append(new_seq)

                    X_eval_LSTM_ELMO = new_X_LSTM_ELMO  # setup X to be the padded sequence

                    print('PROGRES: preprocessed all new stuff!')







                ##true answers from the original preprocessing since we are holding out sets
                if str(gold_standard).lower() == 'true':

                    if 'NAIVE_BAYES' in algos:
                        # print(gold_standard)
                        #(ontology, pmcid_sentence_file_path, filename, excluded_files, all_regex, gold_standard, tokenized_file_path)
                        all_sents_ids_gs, all_sents_gs, all_sent_labels_gs, preprocess_sent_ids_gs, preprocess_sent_gs, preprocess_sent_labels_gs, all_pmcids_gs = load_pmcid_sentence_data(ontology, pmcid_sentence_file_path.replace('/Evaluation_Files',''), filename, excluded_files, all_regex, gold_standard, tokenized_file_path)
                        # print('gold standard', all_sent_labels_gs[:10])



                    if 'CRF' in algos:
                        #CRF
                        training_sentences = []
                        # print(tokenized_file_path)
                        ##tokenized_file_path, ontology, all_ontology_sentences, excluded_files
                        training_sentences = load_data_training(tokenized_file_path.replace('/Evaluation_Files', ''), filename, ontology, training_sentences, excluded_files)

                        y_eval_crf = [sent2labels(s) for s in training_sentences] #true answers

                        if len(y_eval_crf) != len(X_eval_crf):
                            print(len(y_eval_crf))
                            print(len(X_eval_crf))
                            raise Exception('ERROR WITH CRF EVALUATION LENGTHS!')
                        # print(y_eval_crf)

                    if 'LSTM' in algos:
                        #LSTM
                        training_sentences = []
                        training_sentences = load_data_training(tokenized_file_path.replace('/Evaluation_Files', ''),filename, ontology, training_sentences, excluded_files)
                        y_eval_crf = [sent2labels(s) for s in training_sentences]  # true answers
                        y_eval_LSTM = [[biotag2idx_LSTM[l] for l in se] for se in y_eval_crf] #use the eval_crf labels as true answers to get the numbers we want
                        if len(y_eval_LSTM) != len(X_eval_LSTM):
                            print(len(y_eval_LSTM))
                            print(len(X_eval_LSTM))
                            raise Exception('ERROR WITH LSTM EVALUATION LENGTHS!')

                        y_eval_LSTM = pad_sequences(maxlen=max_sentence_length_LSTM, sequences=y_eval_LSTM, padding="post", value=biotag2idx_LSTM['O'])
                        # print(y[1])

                        ##changing the lables of y to categorical
                        y_eval_LSTM = [to_categorical(i, num_classes=len(all_biotags_set_LSTM)) for i in y_eval_LSTM]  # converts the class labels to binary matrix where the labels are the indices of the 1 in the matrix


                    if 'LSTM_CRF' in algos:
                        #LSTM_CRF
                        training_sentences = []
                        training_sentences = load_data_training(tokenized_file_path.replace('/Evaluation_Files', ''),filename, ontology, training_sentences, excluded_files)
                        y_eval_crf = [sent2labels(s) for s in training_sentences]  # true answers
                        y_eval_LSTM_CRF = [[biotag2idx_LSTM_CRF[l] for l in se] for se in y_eval_crf] #use the eval_crf labels as true answers to get the numbers we want
                        if len(y_eval_LSTM_CRF) != len(X_eval_LSTM_CRF):
                            print(len(y_eval_LSTM_CRF))
                            print(len(X_eval_LSTM_CRF))
                            raise Exception('ERROR WITH LSTM EVALUATION LENGTHS!')

                        y_eval_LSTM_CRF = pad_sequences(maxlen=max_sentence_length_LSTM_CRF, sequences=y_eval_LSTM_CRF, padding="post", value=biotag2idx_LSTM_CRF['O'])
                        # print(y[1])

                        ##changing the lables of y to categorical
                        y_eval_LSTM_CRF = [to_categorical(i, num_classes=len(all_biotags_set_LSTM_CRF)) for i in y_eval_LSTM_CRF]  # converts the class labels to binary matrix where the labels are the indices of the 1 in the matrix


                    if 'CHAR_EMBEDDINGS' in algos:
                        #CHAR EMBEDDINGS
                        training_sentences = []
                        training_sentences = load_data_training(tokenized_file_path.replace('/Evaluation_Files', ''),
                                                                filename, ontology, training_sentences, excluded_files)

                        y_eval_char_embeddings = [sent2labels(s) for s in training_sentences]  # true biotag answers
                        # print(len(y_eval_char_embeddings))


                    if 'LSTM_ELMO' in algos:
                        #LSTM_ELMO
                        training_sentences = []
                        training_sentences = load_data_training(tokenized_file_path.replace('/Evaluation_Files', ''),
                                                                filename, ontology, training_sentences, excluded_files)

                        y_eval_LSTM_ELMO = [sent2labels(s) for s in training_sentences]  # true biotag answers
                        # print(len(y_eval_char_embeddings))

                        print('PROGRES: loaded in gold standard!')
                        print('len of gold standard:', len(training_sentences))




                ##load the models for each ontology
                # for ontology in ontologies:
                print('PROGRESS:', ontology)

                for root, directories, filenames in os.walk('%s%s/' %(save_models_path, ontology)):
                    for filename in sorted(filenames):
                        # print(filename)
                        output_results = []  # PMCID	SENTENCE_NUM	SENTENCE_START	SENTENCE_END	WORD	POS_TAG	WORD_START	WORD_END	BIO_TAG	PMC_MENTION_ID	ONTOLOGY_CONCEPT_ID	ONTOLOGY_LABEL

                        ##Naive Bayes models
                        if 'NAIVE_BAYES' in algos and filename.endswith('model.pkl') and 'Naive_Bayes' in filename:
                            print('PROGRESS: running naive bayes models!')
                            NB_model = pickle.load(open(root+filename, 'rb'))
                            features_eval_pred = NB_model.predict(features_eval)
                            # print(features_eval_pred[:10])
                            # print(features_eval_pred.tolist()[:10])
                            # print(all_sent_labels_gs[:10])
                            # print(len(all_sent_labels_gs), len(features_eval_pred))
                            if len(all_sent_labels_gs) != len(features_eval_pred):
                                print(len(all_sent_labels_gs), len(features_eval_pred))
                                raise Exception('ERROR WITH PREDICTION AND GOLD STANDARD NUMBER OF SENTENCES!')


                            if str(gold_standard).lower() == 'true':
                                #TODO: output the prediction report
                                ##output the prediction information
                                ignorance_labels = ['not_ignorance', 'ignorance']
                                with open('%s%s/%s_pred_report.txt' % (output_path, ontology, filename.replace('.pkl', '')),
                                          'w+') as pred_report_file:
                                    pred_report_file.write('NAIVE BAYES FILE REPORT FOR DOCUMENTS: %s\n\n' % (excluded_files))
                                    pred_report_file.write('sentence prediction length: %s\n' % (len(features_eval_pred)))
                                    if str(gold_standard).lower() == 'true':
                                        pred_report_file.write(sklearn.metrics.classification_report(all_sent_labels_gs,features_eval_pred.tolist(), target_names=ignorance_labels, digits=3))


                            ##output the model in sentence format
                            for pmcid in all_pmcids:
                                with open('%s%s/%s_%s.txt' %(output_path, ontology, filename.replace('.pkl', ''), pmcid), 'w+') as nb_output:
                                    nb_output.write('%s\t%s\t%s\t%s\t%s\n' %('PMCID', 'SENTENCE_NUMBER', 'SENTENCE', 'SENTENCE_INDICES', 'IGNORANCE_CLASSIFICATION'))

                            for i, p in enumerate(features_eval_pred):
                                nb_sentence = all_sents[i]
                                nb_sentence_id = all_sents_ids[i] #(pmcid, sentence_number, sentence_indices) delimited with _
                                (nb_pmcid, nb_sentence_number, nb_sentence_indices) = nb_sentence_id.split('_')


                                with open('%s%s/%s_%s.txt' %(output_path, ontology, filename.replace('.pkl', ''), nb_pmcid), 'a+') as nb_output:
                                    if p == 0:
                                        nb_output.write('%s\t%s\t%s\t%s\t%s\n' % (nb_pmcid, nb_sentence_number, [nb_sentence], nb_sentence_indices, 'FALSE'))
                                    elif p == 1:
                                        nb_output.write('%s\t%s\t%s\t%s\t%s\n' % (nb_pmcid, nb_sentence_number, [nb_sentence], nb_sentence_indices, 'TRUE'))
                                    else:
                                        raise Exception('ERROR: THERE SHOULD NOT BE OTHER LABELS BESIDES 0 AND 1 FOR NAIVE BAYES IGNORNACE!')




                        ##CRF models!
                        elif 'CRF' in algos and filename.endswith('.joblib') and 'local' in filename and 'crf' in filename:
                            print('PROGRESS: running CRF models!')
                            # print('MODEL:', filename.replace('.joblib', ''))
                            loaded_model = joblib.load(root+filename)

                            ##labels
                            labels = list(loaded_model.classes_)
                            sorted_labels = sorted(
                                labels,
                                key=lambda name: (name[1:], name[0])
                            )

                            X_eval_crf_pred = loaded_model.predict(X_eval_crf)
                            # print(X_eval_crf)
                            binary_X_eval_crf_pred = [] #B, I, O-
                            binary_X_eval_crf_pred_regex = [] #B, I, O-
                            binary_X_eval_crf_pred_b = [] #only B
                            binary_X_eval_crf_pred_bi = [] #only B and I
                            for i, x in enumerate(X_eval_crf_pred):
                                ##X_eval_crf_pred
                                if len(set(x)) == 1 and list(set(x))[0] == 'O':
                                    binary_X_eval_crf_pred += [0]
                                    if all_sents_ids[i] in preprocess_sent_ids:
                                        # print('got here negative')
                                        binary_X_eval_crf_pred_regex += [0]
                                    else:
                                        pass
                                else:
                                    binary_X_eval_crf_pred += [1]
                                    if all_sents_ids[i] in preprocess_sent_ids:
                                        # print('got here postive')
                                        binary_X_eval_crf_pred_regex += [1]
                                    else:
                                        pass

                                if 'B' in set(x):
                                    binary_X_eval_crf_pred_b += [1]
                                else:
                                    binary_X_eval_crf_pred_b += [0]

                                if 'B' in set(x) or 'I' in set(x):
                                    binary_X_eval_crf_pred_bi += [1]
                                else:
                                    binary_X_eval_crf_pred_bi += [0]

                            # print(len(binary_X_eval_crf_pred_regex))
                            ##check the model is good - if the gold standard exists
                            if str(gold_standard).lower() == 'true':
                                # metrics.flat_f1_score(X_eval_crf_pred, y_eval_crf, average='weighted', labels=labels)
                                # print(len(X_eval_crf_pred))
                                # print(len(y_eval_crf))
                                ignorance_labels = ['not_ignorance', 'ignorance']
                                binary_y_eval_crf = []
                                binary_y_eval_crf_regex = []
                                binary_y_eval_crf_b = []
                                binary_y_eval_crf_bi = []
                                for j, y in enumerate(y_eval_crf):
                                    ##y_eval_crf
                                    if len(set(y)) == 1 and list(set(y))[0] == 'O':
                                        binary_y_eval_crf += [0]
                                        if all_sents_ids[j] in preprocess_sent_ids:
                                            # print('got here negative')
                                            binary_y_eval_crf_regex += [0]
                                        else:
                                            pass
                                    else:
                                        binary_y_eval_crf += [1]
                                        if all_sents_ids[j] in preprocess_sent_ids:
                                            # print('got here negative')
                                            binary_y_eval_crf_regex += [1]
                                        else:
                                            pass

                                    if 'B' in set(y):
                                        binary_y_eval_crf_b += [1]
                                    else:
                                        binary_y_eval_crf_b += [0]

                                    if 'B' in set(y) or 'I' in set(y):
                                        binary_y_eval_crf_bi += [1]
                                    else:
                                        binary_y_eval_crf_bi += [0]


                                #check that all lengths are still the same
                                if len(binary_X_eval_crf_pred) != len(X_eval_crf_pred) or len(binary_y_eval_crf) != len(y_eval_crf):
                                    raise Exception('ERROR WITH BINARY CRF EVALUATION START!')

                                if len(binary_X_eval_crf_pred_regex) != len(binary_y_eval_crf_regex):
                                    print(len(binary_X_eval_crf_pred_regex), len(binary_y_eval_crf_regex))
                                    raise Exception('ERROR WITH BINARY REGEX CRF EVALUATION LENGTHS!')





                                ##output the prediction information
                                with open('%s%s/%s_pred_report.txt' %(output_path, ontology, filename.replace('.joblib', '')), 'w+') as pred_report_file:


                                    if len(list(set(binary_y_eval_crf))) != len(ignorance_labels):
                                        print('FIXING THE IGNORANCE LABELS TO MATCH!')
                                        print(ignorance_labels)
                                        print(set(binary_X_eval_crf_pred))
                                        print(sorted(list(set(binary_X_eval_crf_pred))))
                                        updated_ignorance_labels = [ignorance_labels[l] for l in sorted(list(set(binary_X_eval_crf_pred)))]
                                        ignorance_labels = updated_ignorance_labels
                                    else:
                                        pass




                                    pred_report_file.write('CRF FILE REPORT FOR DOCUMENTS: %s\n\n' %(excluded_files))
                                    pred_report_file.write('sentence prediction length: %s\n' % (len(X_eval_crf_pred)))
                                    if str(gold_standard).lower() == 'true':
                                        pred_report_file.write(flat_classification_report(y_eval_crf, X_eval_crf_pred, labels=sorted_labels, digits=3))

                                        pred_report_file.write('\n\n')
                                        pred_report_file.write('%s\n' %('FULL BINARY PREDICTIONS WITH B,I,O-'))
                                        # print(set(binary_X_eval_crf_pred))
                                        # print(ignorance_labels)

                                        pred_report_file.write(sklearn.metrics.classification_report(binary_y_eval_crf, binary_X_eval_crf_pred, target_names=ignorance_labels, digits=3))

                                        pred_report_file.write('\n\n')
                                        pred_report_file.write('%s\n' % ('REGEX ONLY BINARY PREDICTIONS WITH B,I,O-'))
                                        pred_report_file.write(
                                            sklearn.metrics.classification_report(binary_y_eval_crf_regex, binary_X_eval_crf_pred_regex, target_names=ignorance_labels, digits=3))

                                        pred_report_file.write('\n\n')
                                        pred_report_file.write('%s\n' % ('FULL BINARY PREDICTIONS WITH B'))
                                        pred_report_file.write(
                                            sklearn.metrics.classification_report(binary_y_eval_crf_b,
                                                                                  binary_X_eval_crf_pred_b,
                                                                                  target_names=ignorance_labels, digits=3))

                                        pred_report_file.write('\n\n')
                                        pred_report_file.write('%s\n' % ('FULL BINARY PREDICTIONS WITH B, I'))
                                        pred_report_file.write(
                                            sklearn.metrics.classification_report(binary_y_eval_crf_bi,
                                                                                  binary_X_eval_crf_pred_bi,
                                                                                  target_names=ignorance_labels, digits=3))



                            ##output the model in sentence format
                            for pmcid in all_pmcids:
                                with open('%s%s/%s_%s_sentences.txt' % (output_path, ontology, filename.replace('.joblib', ''), pmcid),
                                          'w+') as crf_sent_output:
                                    crf_sent_output.write('%s\t%s\t%s\t%s\t%s\n' % (
                                    'PMCID', 'SENTENCE_NUMBER', 'SENTENCE', 'SENTENCE_INDICES', 'IGNORANCE_CLASSIFICATION'))

                            for i, b in enumerate(binary_X_eval_crf_pred):
                                b_sentence = all_sents[i]
                                b_sentence_id = all_sents_ids[i]  # (pmcid, sentence_number, sentence_indices) delimited with _
                                (b_pmcid, b_sentence_number, b_sentence_indices) = b_sentence_id.split('_')

                                with open('%s%s/%s_%s_sentences.txt' % (output_path, ontology, filename.replace('.joblib', ''), b_pmcid),
                                          'a+') as b_output:
                                    if b == 0:
                                        b_output.write('%s\t%s\t%s\t%s\t%s\n' % (
                                        b_pmcid, b_sentence_number, [b_sentence], b_sentence_indices, 'FALSE'))
                                    elif b == 1:
                                        b_output.write('%s\t%s\t%s\t%s\t%s\n' % (
                                        b_pmcid, b_sentence_number, [b_sentence], b_sentence_indices, 'TRUE'))
                                    else:
                                        raise Exception(
                                            'ERROR: THERE SHOULD NOT BE OTHER LABELS BESIDES 0 AND 1 FOR CRF SENTENCE IGNORANCE CLASSIFICATION!')


                            ##output the model in BIO format for concept normalizaiton
                            pmcid_starts_dict = {} #pmcid -> index
                            for i,s in enumerate(all_sentences): #index, value
                                if len(s) != len(X_eval_crf[i]):
                                    raise Exception('ERROR WITH PREDICTION ON EVAL SET!')
                                else:
                                    for j in range(len(s)):
                                        ##gather the end of the article to know when to move to the next article
                                        pmcid_starts_dict[all_sentence_info[i][j][0]] = [None, len(output_results)] #last one will stay
                                        # if pmcid_starts_dict.get(all_sentence_info[i][j][0]):
                                        #     pass
                                        #
                                        # else:
                                        #     pmcid_starts_dict[all_sentence_info[i][j][0]] = len(output_results)

                                        ##output results by word
                                        output_results += [(all_sentence_info[i][j][0], all_sentence_info[i][j][1], all_sentence_info[i][j][2], all_sentence_info[i][j][3], s[j][0], s[j][1], all_sentence_info[i][j][4], all_sentence_info[i][j][5], X_eval_crf_pred[i][j], None, None, None)]

                            ##add the starts to the pmcid_starts_dict
                            for i, pmcid in enumerate(all_pmcid_list):
                                if i == 0:
                                    pmcid_starts_dict[pmcid][0] = 0
                                else:
                                    pmcid_starts_dict[pmcid][0] = pmcid_starts_dict[all_pmcid_list[i-1]][1]+1

                            # print(pmcid_starts_dict)



                            # ##output the results in BIO format for concept normalization
                            output_results_path = '%s%s/%s' % (output_path, ontology, filename.replace('.joblib', ''))
                            output_span_detection_results(all_pmcid_list, pmcid_starts_dict, output_path, filename, output_results, output_results_path)




                        ##LSTM models!
                        elif 'LSTM' in algos and filename.endswith('.h5') and 'local' in filename and filename.startswith('%s_%s' %(ontology, 'LSTM_model')):
                            print('PROGRESS: running LSTM models!')
                            # print(root+filename)
                            LSTM_model = load_model(root+filename)

                            ##output without ENDPAD
                            X_eval_LSTM_no_endpad = [] #words of the sentence
                            X_eval_LSTM_pred = [] #biotag predictions


                            ##predictions on new documents
                            #prediction output
                            # with open('%s%s/%s_eval_predictions.txt' % (output_path, ontology, filename.replace('.h5','')), 'w+') as pred_file:

                            if str(gold_standard).lower() == 'true': #3 columns in the output
                                unseen_word_count = 0
                                # pred_file.write('%s\t%s\t%s\n' % ("Word", "True", "Pred"))
                                y_eval_LSTM_biotags = []  # biotag

                                #predict per sentence in X_eval_LSTM
                                for i in range(len(X_eval_LSTM)):
                                    ##collect the sentence information for the final output:
                                    word_sentence_vector = []
                                    biotag_sentence_vector = []
                                    y_eval_biotag_sentence_vector = []

                                    p = LSTM_model.predict(np.array([X_eval_LSTM[i]]))
                                    p = np.argmax(p, axis=-1)

                                    # print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
                                    # print('test labels:', type(y_te[i]), y_te[i])

                                    for w, tru_bin, pred in zip(X_eval_LSTM[i], y_eval_LSTM[i], p[0]):
                                        ##tru is a binary numpy array - need to find the index of the 1 and that is the biotag2idx label
                                        tru = np.where(tru_bin == 1.)
                                        if len(tru[0]) != 1:
                                            raise Exception('ERROR WITH BINARY LABELS (CATEGORICAL) FOR BIOTAG LABELS!')
                                        else:
                                            tru = tru[0][0]

                                        # print(w, type(w))
                                        # print(tru, type(tru))
                                        # print(pred, type(pred))
                                        # print("{:15}: {} {}".format(all_words_set[w], all_biotags_set[w], all_biotags_set[pred]))

                                        ##print out the original word not OOV_UNSEEN
                                        if all_words_set_LSTM[w] == 'OOV_UNSEEN':
                                            unseen_word = unseen_original_words_LSTM[unseen_word_count]
                                            unseen_word_count += 1
                                            # pred_file.write('%s\t%s\t%s\n' % (unseen_word, all_biotags_set[int(tru)], all_biotags_set[pred]))
                                            if all_words_set_LSTM[w] != 'ENDPAD':
                                                word_sentence_vector += [unseen_word]
                                                biotag_sentence_vector += [all_biotags_set_LSTM[pred]]
                                                y_eval_biotag_sentence_vector += [all_biotags_set_LSTM[tru]]
                                            else:
                                                pass
                                        else:
                                            # pred_file.write('%s\t%s\t%s\n' % (all_words_set[w], all_biotags_set[int(tru)], all_biotags_set[pred]))
                                            if all_words_set_LSTM[w] != 'ENDPAD':
                                                word_sentence_vector += [all_words_set_LSTM[w]]
                                                biotag_sentence_vector += [all_biotags_set_LSTM[pred]]
                                                y_eval_biotag_sentence_vector += [all_biotags_set_LSTM[tru]]
                                            else:
                                                pass

                                    X_eval_LSTM_no_endpad += [word_sentence_vector]  # words of the sentence
                                    X_eval_LSTM_pred += [biotag_sentence_vector]  # biotag predictions
                                    y_eval_LSTM_biotags += [y_eval_biotag_sentence_vector] #biotag true

                            else: #2 columns in the output
                                unseen_word_count_LSTM = 0
                                # pred_file.write('%s\t%s\n' % ("Word", "Pred"))
                                for i in range(len(X_eval_LSTM)):
                                    ##collect the sentence information for the final output:
                                    word_sentence_vector = []
                                    biotag_sentence_vector = []

                                    p = LSTM_model.predict(np.array([X_eval_LSTM[i]]))
                                    p = np.argmax(p, axis=-1)

                                    # print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
                                    # print('test labels:', type(y_te[i]), y_te[i])
                                    for w, pred in zip(X_eval_LSTM[i], p[0]):
                                        ##print out the original word not OOV_UNSEEN
                                        if all_words_set_LSTM[w] == 'OOV_UNSEEN':
                                            unseen_word = unseen_original_words_LSTM[unseen_word_count_LSTM]
                                            unseen_word_count_LSTM += 1
                                            # pred_file.write('%s\t%s\n' % (unseen_word, all_biotags_set[pred]))

                                            if all_words_set_LSTM[w] != 'ENDPAD':
                                                word_sentence_vector += [unseen_word]
                                                biotag_sentence_vector += [all_biotags_set_LSTM[pred]]
                                            else:
                                                pass
                                        else:
                                            # pred_file.write('%s\t%s\n' % (all_words_set[w], all_biotags_set[pred]))
                                            if all_words_set_LSTM[w] != 'ENDPAD':
                                                word_sentence_vector += [all_words_set_LSTM[w]]
                                                biotag_sentence_vector += [all_biotags_set_LSTM[pred]]
                                            else:
                                                pass

                                    X_eval_LSTM_no_endpad += [word_sentence_vector]  # words of the sentence
                                    X_eval_LSTM_pred += [biotag_sentence_vector]  # biotag predictions

                            #prediction report if gold standard information
                            if str(gold_standard).lower() == 'true':
                                biotags = ['B', 'I', 'O', 'O-']
                                closer_biotags = ['B', 'I']
                                # LSTM_prediction_report(output_path, ontology, biotags, closer_biotags, filename) #outputs the prediction report given that we have all the gold standard information
                                # print(X_eval_LSTM_pred[0])
                                # print(len(X_eval_LSTM_pred))
                                # print(y_eval_LSTM[0])
                                # print(y_eval_LSTM_biotags[0])
                                # print(len(y_eval_LSTM_biotags), len(y_eval_LSTM))
                                # raise Exception('HOLD!')

                                ##update the labels so that O- becomes D for discontinuity to get it in the classification report
                                # print(y_eval_LSTM_biotags[0])
                                # print(X_eval_LSTM_pred[0])

                                updated_test_labels = ['D' if l == 'O-' else l for s in y_eval_LSTM_biotags for l in s]
                                updated_pred_labels = ['D' if l == 'O-' else l for s in X_eval_LSTM_pred for l in s]

                                ##full with O as W for words not included!
                                updated_test_labels_full = ['W' if l == 'O' else l for s in updated_test_labels for l in s]
                                updated_pred_labels_full = ['W' if l == 'O' else l for s in updated_pred_labels for l in s]

                                # print(updated_pred_labels_full.count('B'), updated_test_labels_full.count('B'))

                                # print(sklearn.metrics.classification_report(updated_test_labels, updated_pred_labels))
                                # print(sklearn.metrics.classification_report(updated_test_labels_full, updated_pred_labels_full))

                                # # output the classification report - B, I, O- (D), O (W)
                                # print(updated_test_labels[0])
                                # print(updated_pred_labels[0])
                                #
                                #'PMCID', 'TYPE', 'PRECISION', 'RECALL', 'F1_MEASURE'
                                ###TODO: WEIRDNESS WITH CLASSIFICATION REPORT!!! i don't think it is correct
                                local_eval_files_creport_LSTM.write('%s\t%s\t%.2f\t%.2f\t%.2f\n' % (pmcid, 'MACRO-AVG',sklearn.metrics.precision_score(updated_test_labels, updated_pred_labels, average='macro'), sklearn.metrics.recall_score(updated_test_labels_full, updated_pred_labels, average='macro'), sklearn.metrics.f1_score(updated_test_labels, updated_pred_labels, average='macro')))

                                local_eval_files_creport_LSTM.write('%s\t%s\t%.2f\t%.2f\t%.2f\n' % (pmcid, 'WEIGHTED', sklearn.metrics.precision_score(updated_test_labels_full, updated_pred_labels_full, average='weighted'), sklearn.metrics.recall_score(updated_test_labels_full, updated_pred_labels_full, average='weighted'), sklearn.metrics.f1_score(updated_test_labels_full, updated_pred_labels_full, average='weighted')))




                            else:
                                pass #no prediction report


                            #TODO: tokenized files in BIO format for conept normalization
                            ##output the model in BIO format for concept normalizaiton:
                            # output_results = []  # PMCID	SENTENCE_NUM	SENTENCE_START	SENTENCE_END	WORD	POS_TAG	WORD_START	WORD_END	BIO_TAG	PMC_MENTION_ID	ONTOLOGY_CONCEPT_ID	ONTOLOGY_LABEL


                            ##need to get rid of the ENDPAD
                            pmcid_starts_dict = {}  # pmcid -> index
                            for i, s in enumerate(all_sentences):  # index, value
                                if len(s) != len(X_eval_LSTM_no_endpad[i]):
                                    print(len(s))
                                    print(len(X_eval_LSTM_no_endpad[i]))
                                    print(len(X_eval_LSTM_pred[i]))
                                    raise Exception('ERROR WITH PREDICTION ON EVAL SET!')
                                else:
                                    for j in range(len(s)):
                                        ##gather the end of the article to know when to move to the next article
                                        pmcid_starts_dict[all_sentence_info[i][j][0]] = [None, len(output_results)]  # last one will stay
                                        # if pmcid_starts_dict.get(all_sentence_info[i][j][0]):
                                        #     pass
                                        #
                                        # else:
                                        #     pmcid_starts_dict[all_sentence_info[i][j][0]] = len(output_results)

                                        ##output results by word
                                        output_results += [(all_sentence_info[i][j][0], all_sentence_info[i][j][1],
                                                            all_sentence_info[i][j][2], all_sentence_info[i][j][3], s[j][0],
                                                            s[j][1], all_sentence_info[i][j][4], all_sentence_info[i][j][5],
                                                            X_eval_LSTM_pred[i][j], None, None, None)]

                            ##add the starts to the pmcid_starts_dict
                            for i, pmcid in enumerate(all_pmcid_list):
                                if i == 0:
                                    pmcid_starts_dict[pmcid][0] = 0
                                else:
                                    pmcid_starts_dict[pmcid][0] = pmcid_starts_dict[all_pmcid_list[i - 1]][1] + 1

                            ##full output in bio format
                            output_results_path = '%s%s/%s' % (output_path, ontology, filename.replace('.h5', ''))
                            output_span_detection_results(all_pmcid_list, pmcid_starts_dict, output_path, filename, output_results, output_results_path)




                        ##LSTM_CRF models!
                        elif 'LSTM_CRF' in algos and filename.endswith('.h5') and 'local' in filename and filename.startswith('%s_%s' %(ontology, 'LSTM_CRF_model')):
                            print('PROGRESS: running LSTM_CRF models!')
                            # print(root+filename)
                            crf = CRF(len(all_biotags_set_LSTM_CRF))
                            ##LOAD THE LSTM CRF MODEL: https://github.com/keras-team/keras-contrib/issues/125
                            LSTM_CRF_model = load_model(root + filename, custom_objects={'CRF': CRF, 'crf_loss': crf.loss_function, 'crf_viterbi_accuracy': crf.viterbi_acc})

                            ##output without ENDPAD
                            X_eval_LSTM_CRF_no_endpad = []  # words of the sentence
                            X_eval_LSTM_CRF_pred = []  # biotag predictions

                            ##predictions on new documents
                            # prediction output
                            # with open('%s%s/%s_eval_predictions.txt' % (output_path, ontology, filename.replace('.h5','')), 'w+') as pred_file:

                            if str(gold_standard).lower() == 'true':  # 3 columns in the output
                                unseen_word_count_LSTM_CRF = 0
                                # pred_file.write('%s\t%s\t%s\n' % ("Word", "True", "Pred"))
                                y_eval_LSTM_CRF_biotags = []  # biotag

                                # predict per sentence in X_eval_LSTM_CRF
                                for i in range(len(X_eval_LSTM_CRF)):
                                    ##collect the sentence information for the final output:
                                    word_sentence_vector = []
                                    biotag_sentence_vector = []
                                    y_eval_biotag_sentence_vector = []

                                    p = LSTM_CRF_model.predict(np.array([X_eval_LSTM_CRF[i]]))
                                    p = np.argmax(p, axis=-1)

                                    # print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
                                    # print('test labels:', type(y_te[i]), y_te[i])
                                    # print('LSTM_CRF INFO')
                                    # print(X_eval_LSTM_CRF[i])
                                    # print(len(X_eval_LSTM_CRF[i]))
                                    # print(p)
                                    # print(len(y_eval_LSTM_CRF[i]))


                                    for w, tru_bin, pred in zip(X_eval_LSTM_CRF[i], y_eval_LSTM_CRF[i], p[0]):
                                        ##tru is a binary numpy array - need to find the index of the 1 and that is the biotag2idx label

                                        tru = np.where(tru_bin == 1.)
                                        if len(tru[0]) != 1:
                                            raise Exception('ERROR WITH BINARY LABELS (CATEGORICAL) FOR BIOTAG LABELS!')
                                        else:
                                            tru = tru[0][0]

                                        # print(w, type(w))
                                        # print(tru, type(tru))
                                        # print(pred, type(pred))
                                        # print("{:15}: {} {}".format(all_words_set[w], all_biotags_set[w], all_biotags_set[pred]))
                                        # print(w, tru, pred)
                                        # print(len(all_words_set_LSTM_CRF))
                                        ##print out the original word not OOV_UNSEEN - WE ADDED ONE FOR THE WORD2IDX SO WE HAVE TO SUBTRACT 1 FROM W (W-1) TO GET THE RIGHT WORD
                                        if all_words_set_LSTM_CRF[w-1] == 'OOV_UNSEEN':
                                            unseen_word = unseen_original_words_LSTM_CRF[unseen_word_count_LSTM_CRF]
                                            unseen_word_count_LSTM_CRF += 1
                                            # pred_file.write('%s\t%s\t%s\n' % (unseen_word, all_biotags_set[int(tru)], all_biotags_set[pred]))
                                            if all_words_set_LSTM_CRF[w-1] != 'ENDPAD':
                                                word_sentence_vector += [unseen_word]
                                                biotag_sentence_vector += [all_biotags_set_LSTM_CRF[pred]]
                                                y_eval_biotag_sentence_vector += [all_biotags_set_LSTM_CRF[tru]]
                                            else:
                                                pass
                                        else:
                                            # pred_file.write('%s\t%s\t%s\n' % (all_words_set[w], all_biotags_set[int(tru)], all_biotags_set[pred]))
                                            if all_words_set_LSTM_CRF[w-1] != 'ENDPAD':
                                                word_sentence_vector += [all_words_set_LSTM_CRF[w-1]]
                                                biotag_sentence_vector += [all_biotags_set_LSTM_CRF[pred]]
                                                y_eval_biotag_sentence_vector += [all_biotags_set_LSTM_CRF[tru]]
                                            else:
                                                pass

                                    X_eval_LSTM_CRF_no_endpad += [word_sentence_vector]  # words of the sentence
                                    X_eval_LSTM_CRF_pred += [biotag_sentence_vector]  # biotag predictions
                                    y_eval_LSTM_CRF_biotags += [y_eval_biotag_sentence_vector]  # biotag true

                            else:  # 2 columns in the output
                                unseen_word_count_LSTM_CRF = 0
                                # pred_file.write('%s\t%s\n' % ("Word", "Pred"))
                                for i in range(len(X_eval_LSTM_CRF)):
                                    ##collect the sentence information for the final output:
                                    word_sentence_vector = []
                                    biotag_sentence_vector = []

                                    p = LSTM_CRF_model.predict(np.array([X_eval_LSTM_CRF[i]]))
                                    p = np.argmax(p, axis=-1)

                                    # print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
                                    # print('test labels:', type(y_te[i]), y_te[i])
                                    for w, pred in zip(X_eval_LSTM_CRF[i], p[0]):
                                        ##print out the original word not OOV_UNSEEN
                                        if all_words_set_LSTM_CRF[w-1] == 'OOV_UNSEEN':
                                            unseen_word = unseen_original_words_LSTM_CRF[unseen_word_count_LSTM_CRF]
                                            unseen_word_count_LSTM_CRF += 1
                                            # pred_file.write('%s\t%s\n' % (unseen_word, all_biotags_set[pred]))

                                            if all_words_set_LSTM_CRF[w-1] != 'ENDPAD':
                                                word_sentence_vector += [unseen_word]
                                                biotag_sentence_vector += [all_biotags_set_LSTM_CRF[pred]]
                                            else:
                                                pass
                                        else:
                                            # pred_file.write('%s\t%s\n' % (all_words_set[w], all_biotags_set[pred]))
                                            if all_words_set_LSTM_CRF[w-1] != 'ENDPAD':
                                                word_sentence_vector += [all_words_set_LSTM_CRF[w-1]]
                                                biotag_sentence_vector += [all_biotags_set_LSTM_CRF[pred]]
                                            else:
                                                pass

                                    X_eval_LSTM_CRF_no_endpad += [word_sentence_vector]  # words of the sentence
                                    X_eval_LSTM_CRF_pred += [biotag_sentence_vector]  # biotag predictions

                            # prediction report if gold standard information
                            if str(gold_standard).lower() == 'true':
                                biotags = ['B', 'I', 'O', 'O-']
                                closer_biotags = ['B', 'I']
                                # LSTM_prediction_report(output_path, ontology, biotags, closer_biotags, filename) #outputs the prediction report given that we have all the gold standard information
                                # print(X_eval_LSTM_CRF_pred[0])
                                # print(len(X_eval_LSTM_CRF_pred))
                                # print(y_eval_LSTM_CRF[0])
                                # print(y_eval_LSTM_CRF_biotags[0])
                                # print(len(y_eval_LSTM_CRF_biotags), len(y_eval_LSTM_CRF))
                                # raise Exception('HOLD!')

                                ##update the labels so that O- becomes D for discontinuity to get it in the classification report
                                # print(y_eval_LSTM_CRF_biotags[0])
                                # print(X_eval_LSTM_CRF_pred[0])

                                updated_test_labels = ['D' if l == 'O-' else l for s in y_eval_LSTM_CRF_biotags for l in s]
                                updated_pred_labels = ['D' if l == 'O-' else l for s in X_eval_LSTM_CRF_pred for l in s]

                                ##full with O as W for words not included!
                                updated_test_labels_full = ['W' if l == 'O' else l for s in updated_test_labels for l in
                                                            s]
                                updated_pred_labels_full = ['W' if l == 'O' else l for s in updated_pred_labels for l in
                                                            s]

                                # print(updated_pred_labels_full.count('B'), updated_test_labels_full.count('B'))

                                # print(sklearn.metrics.classification_report(updated_test_labels, updated_pred_labels))
                                # print(sklearn.metrics.classification_report(updated_test_labels_full, updated_pred_labels_full))

                                # # output the classification report - B, I, O- (D), O (W)
                                # print(updated_test_labels[0])
                                # print(updated_pred_labels[0])
                                #
                                # 'PMCID', 'TYPE', 'PRECISION', 'RECALL', 'F1_MEASURE'
                                ###TODO: WEIRDNESS WITH CLASSIFICATION REPORT!!! i don't think it is correct
                                local_eval_files_creport_LSTM_CRF.write('%s\t%s\t%.2f\t%.2f\t%.2f\n' % (pmcid, 'MACRO-AVG',
                                                                                                    sklearn.metrics.precision_score(
                                                                                                        updated_test_labels,
                                                                                                        updated_pred_labels,
                                                                                                        average='macro'),
                                                                                                    sklearn.metrics.recall_score(
                                                                                                        updated_test_labels_full,
                                                                                                        updated_pred_labels,
                                                                                                        average='macro'),
                                                                                                    sklearn.metrics.f1_score(
                                                                                                        updated_test_labels,
                                                                                                        updated_pred_labels,
                                                                                                        average='macro')))

                                local_eval_files_creport_LSTM_CRF.write('%s\t%s\t%.2f\t%.2f\t%.2f\n' % (pmcid, 'WEIGHTED',
                                                                                                    sklearn.metrics.precision_score(
                                                                                                        updated_test_labels_full,
                                                                                                        updated_pred_labels_full,
                                                                                                        average='weighted'),
                                                                                                    sklearn.metrics.recall_score(
                                                                                                        updated_test_labels_full,
                                                                                                        updated_pred_labels_full,
                                                                                                        average='weighted'),
                                                                                                    sklearn.metrics.f1_score(
                                                                                                        updated_test_labels_full,
                                                                                                        updated_pred_labels_full,
                                                                                                        average='weighted')))




                            else:
                                pass  # no prediction report

                            # TODO: tokenized files in BIO format for conept normalization
                            ##output the model in BIO format for concept normalizaiton:
                            # output_results = []  # PMCID	SENTENCE_NUM	SENTENCE_START	SENTENCE_END	WORD	POS_TAG	WORD_START	WORD_END	BIO_TAG	PMC_MENTION_ID	ONTOLOGY_CONCEPT_ID	ONTOLOGY_LABEL

                            ##need to get rid of the ENDPAD
                            pmcid_starts_dict = {}  # pmcid -> index
                            for i, s in enumerate(all_sentences):  # index, value
                                if len(s) != len(X_eval_LSTM_CRF_no_endpad[i]):
                                    print(len(s))
                                    print(len(X_eval_LSTM_CRF_no_endpad[i]))
                                    print(len(X_eval_LSTM_CRF_pred[i]))
                                    raise Exception('ERROR WITH PREDICTION ON EVAL SET!')
                                else:
                                    for j in range(len(s)):
                                        ##gather the end of the article to know when to move to the next article
                                        pmcid_starts_dict[all_sentence_info[i][j][0]] = [None, len(
                                            output_results)]  # last one will stay
                                        # if pmcid_starts_dict.get(all_sentence_info[i][j][0]):
                                        #     pass
                                        #
                                        # else:
                                        #     pmcid_starts_dict[all_sentence_info[i][j][0]] = len(output_results)

                                        ##output results by word
                                        output_results += [(all_sentence_info[i][j][0], all_sentence_info[i][j][1],
                                                            all_sentence_info[i][j][2], all_sentence_info[i][j][3],
                                                            s[j][0],
                                                            s[j][1], all_sentence_info[i][j][4],
                                                            all_sentence_info[i][j][5],
                                                            X_eval_LSTM_CRF_pred[i][j], None, None, None)]

                            ##add the starts to the pmcid_starts_dict
                            for i, pmcid in enumerate(all_pmcid_list):
                                if i == 0:
                                    pmcid_starts_dict[pmcid][0] = 0
                                else:
                                    pmcid_starts_dict[pmcid][0] = pmcid_starts_dict[all_pmcid_list[i - 1]][1] + 1

                            ##full output in bio format
                            output_results_path = '%s%s/%s' % (output_path, ontology, filename.replace('.h5', ''))
                            output_span_detection_results(all_pmcid_list, pmcid_starts_dict, output_path, filename,
                                                          output_results, output_results_path)


                        ##CHAR EMBEDDING MODELS! - ONLY LSTM RIGHT NOW - TODO: ADD LSTM-CRF POTENTIALLY
                        elif 'CHAR_EMBEDDINGS' in algos and filename.endswith('.h5') and 'local' in filename and filename.startswith('%s_%s' %(ontology, 'char_embeddings')):
                            #load the model
                            print('model name', filename)
                            char_embeddings_model = load_model(root + filename)
                            print(len(X_eval_word_char_embeddings), len(X_eval_char_char_embeddings))
                            print(np.array(X_eval_char_char_embeddings).reshape((len(X_eval_char_char_embeddings), max_sentence_length_char_embeddings, max_char_length_char_embeddings)).shape)
                            print(X_eval_word_char_embeddings.shape)
                            print(X_eval_word_char_embeddings[0][0])
                            # print(X_eval_word_char_embeddings[0], X_eval_char_char_embeddings[0])

                            # #run the model

                            # char_embeddings_model.summary()
                            ##calculate the prediction separately for each sentence
                            # get the prediction labels as biotags per sentence per word

                            idx2biotag_char_embeddings = {i: w for w, i in biotag2idx_char_embeddings.items()}
                            # X_eval_char_embeddings_pred = []
                            # for s in range(len(X_eval_word_char_embeddings)):
                            #
                            #     try:
                            #         s_pred = char_embeddings_model.predict([[X_eval_word_char_embeddings[s]], np.array([X_eval_char_char_embeddings[s]]).reshape(1, max_sentence_length_char_embeddings, max_char_length_char_embeddings)])
                            #         # print(s_pred)
                            #
                            #         p = np.argmax(s_pred, axis=-1)
                            #         # print(p[0]) #the biotag idx for all the words - len is 410
                            #         sentence_pred = []
                            #         for b in p[0]:
                            #             sentence_pred += [idx2biotag_char_embeddings[b]]
                            #
                            #         # print(sentence_pred)
                            #         if 'B' in sentence_pred or 'I' in sentence_pred:
                            #             print('woot!')
                            #         X_eval_char_embeddings_pred += [sentence_pred]
                            #
                            #     except:
                            #         ##fixed!
                            #         print(s)
                            #         print(X_eval_word_char_embeddings[s])
                            #         print(X_eval_char_char_embeddings[s])
                            #         print(np.array([X_eval_char_char_embeddings[s]]).reshape(1,max_sentence_length_char_embeddings,max_char_length_char_embeddings))
                            #         raise Exception('ERROR WITH SENTENCE/WORD/CHAR EMBEDDING!')

                                # raise Exception('hold!')


                            ##DO IT ALL AT ONCE INSTEAD OF PER SENTENCE!
                            X_eval_char_embeddings_pred = char_embeddings_model.predict([X_eval_word_char_embeddings, np.array(X_eval_char_char_embeddings).reshape((len(X_eval_char_char_embeddings), max_sentence_length_char_embeddings, max_char_length_char_embeddings))])

                            #get the prediction labels as biotags per sentence per word
                            idx2biotag_char_embeddings = {i: w for w, i in biotag2idx_char_embeddings.items()}

                            pred_labels_char_embeddings = pred2label(X_eval_char_embeddings_pred, idx2biotag_char_embeddings) #prediction labels on the new documents!
                            # print(pred_labels_char_embeddings[0])
                            # raise Exception('HOLD')

                            #output the predictions!
                            unseen_word_count = 0
                            ##output without ENDPAD
                            X_eval_char_embeddings_no_endpad = []  # words of the sentence
                            X_eval_char_embeddings_biotags_no_endpad = [] #biotags without endpad

                            for i in range(len(X_eval_word_char_embeddings)):
                                ##collect the sentence information for the final output:
                                word_sentence_vector = []
                                biotag_sentence_vector = []

                                for j, w in enumerate(X_eval_word_char_embeddings[i]):
                                    # print(w, idx2word_char_embeddings[w])
                                    ##print out the original word not OOV_UNSEEN
                                    if idx2word_char_embeddings[w] == 'OOV_UNSEEN':
                                        unseen_word = unseen_original_words_char_embeddings[unseen_word_count]
                                        unseen_word_count += 1
                                        word_sentence_vector += [unseen_word] #the word

                                        biotag_sentence_vector += [pred_labels_char_embeddings[i][j]] #the biotag


                                    else:
                                        # pred_file.write('%s\t%s\t%s\n' % (all_words_set[w], all_biotags_set[int(tru)], all_biotags_set[pred]))
                                        if idx2word_char_embeddings[w] != 'ENDPAD':
                                            word_sentence_vector += [idx2word_char_embeddings[w]]
                                            biotag_sentence_vector += [pred_labels_char_embeddings[i][j]]  # the biotag

                                        else:
                                            pass

                                X_eval_char_embeddings_no_endpad += [word_sentence_vector]  # words of the sentence
                                X_eval_char_embeddings_biotags_no_endpad += [biotag_sentence_vector] #biotags!
                                # if 'B' in biotag_sentence_vector or 'I' in biotag_sentence_vector:
                                #     print('woot!!!')
                                # print(len(X_eval_char_embeddings_biotags_no_endpad[0]))


                            if gold_standard.lower() == 'true':


                                # print('prediction info')
                                # print(len(y_eval_char_embeddings), len(X_eval_char_embeddings_biotags_no_endpad))
                                # print('num B for true', sum([t.count('B') for t in y_eval_char_embeddings]))
                                # print('num B for pred', sum([p.count('B') for p in X_eval_char_embeddings_biotags_no_endpad]))
                                #
                                # print('num I for true', sum([t.count('I') for t in y_eval_char_embeddings]))
                                # print('num I for pred', sum([p.count('I') for p in X_eval_char_embeddings_biotags_no_endpad]))
                                #
                                # print('num O- for true', sum([t.count('O-') for t in y_eval_char_embeddings]))
                                # print('num O- for pred', sum([p.count('O-') for p in X_eval_char_embeddings_biotags_no_endpad]))
                                #
                                # print('num O for true', sum([t.count('O') for t in y_eval_char_embeddings]))
                                # print('num O for pred', sum([p.count('O') for p in X_eval_char_embeddings_biotags_no_endpad]))
                                #
                                # print('classification report scores!')
                                # print(precision_score(y_eval_char_embeddings, X_eval_char_embeddings_biotags_no_endpad))
                                # print(recall_score(y_eval_char_embeddings, X_eval_char_embeddings_biotags_no_endpad))
                                # print(f1_score(y_eval_char_embeddings, X_eval_char_embeddings_biotags_no_endpad))
                                #
                                # print(classification_report(y_eval_char_embeddings, X_eval_char_embeddings_biotags_no_endpad))

                                ##update the labels so that O- becomes D for discontinuity to get it in the classification report
                                # print(y_eval_LSTM_biotags[0])
                                # print(X_eval_LSTM_pred[0])

                                updated_test_labels = ['D' if l == 'O-' else l for s in y_eval_char_embeddings for l in s]
                                updated_pred_labels = ['D' if l == 'O-' else l for s in X_eval_char_embeddings_biotags_no_endpad for l in s]

                                ##full with O as W for words not included!
                                updated_test_labels_full = ['W' if l == 'O' else l for s in updated_test_labels for l in
                                                            s]
                                updated_pred_labels_full = ['W' if l == 'O' else l for s in updated_pred_labels for l in
                                                            s]

                                # print(updated_pred_labels_full.count('B'), updated_test_labels_full.count('B'))

                                # print(sklearn.metrics.classification_report(updated_test_labels, updated_pred_labels))
                                # print(sklearn.metrics.classification_report(updated_test_labels_full, updated_pred_labels_full))

                                # # output the classification report - B, I, O- (D), O (W)
                                # print(updated_test_labels[0])
                                # print(updated_pred_labels[0])
                                #
                                # 'PMCID', 'TYPE', 'PRECISION', 'RECALL', 'F1_MEASURE'
                                ###TODO: WEIRDNESS WITH CLASSIFICATION REPORT!!! i don't think it is correct
                                local_eval_files_creport_char_embeddings_LSTM.write(
                                    '%s\t%s\t%.2f\t%.2f\t%.2f\n' % (pmcid, 'MACRO-AVG',
                                                                    sklearn.metrics.precision_score(
                                                                        updated_test_labels,
                                                                        updated_pred_labels,
                                                                        average='macro'),
                                                                    sklearn.metrics.recall_score(
                                                                        updated_test_labels,
                                                                        updated_pred_labels,
                                                                        average='macro'),
                                                                    sklearn.metrics.f1_score(
                                                                        updated_test_labels,
                                                                        updated_pred_labels,
                                                                        average='macro')))

                                local_eval_files_creport_char_embeddings_LSTM.write(
                                    '%s\t%s\t%.2f\t%.2f\t%.2f\n' % (pmcid, 'WEIGHTED',
                                                                    sklearn.metrics.precision_score(
                                                                        updated_test_labels_full,
                                                                        updated_pred_labels_full,
                                                                        average='weighted'),
                                                                    sklearn.metrics.recall_score(
                                                                        updated_test_labels_full,
                                                                        updated_pred_labels_full,
                                                                        average='weighted'),
                                                                    sklearn.metrics.f1_score(
                                                                        updated_test_labels_full,
                                                                        updated_pred_labels_full,
                                                                        average='weighted')))






                            # TODO: tokenized files in BIO format for conept normalization
                            ##output the model in BIO format for concept normalizaiton:
                            # output_results = []  # PMCID	SENTENCE_NUM	SENTENCE_START	SENTENCE_END	WORD	POS_TAG	WORD_START	WORD_END	BIO_TAG	PMC_MENTION_ID	ONTOLOGY_CONCEPT_ID	ONTOLOGY_LABEL

                            ##need to get rid of the ENDPAD
                            pmcid_starts_dict = {}  # pmcid -> index
                            for i, s in enumerate(all_sentences):  # index, value
                                if len(s) != len(X_eval_char_embeddings_no_endpad[i]):
                                    print(len(s))
                                    print(len(X_eval_char_embeddings_no_endpad[i]))
                                    print(len(pred_labels_char_embeddings[i]))
                                    raise Exception('ERROR WITH PREDICTION ON EVAL SET!')
                                else:
                                    for j in range(len(s)):
                                        ##gather the end of the article to know when to move to the next article
                                        pmcid_starts_dict[all_sentence_info[i][j][0]] = [None, len(output_results)]  # last one will stay
                                        # if pmcid_starts_dict.get(all_sentence_info[i][j][0]):
                                        #     pass
                                        #
                                        # else:
                                        #     pmcid_starts_dict[all_sentence_info[i][j][0]] = len(output_results)

                                        ##output results by word
                                        output_results += [(all_sentence_info[i][j][0], all_sentence_info[i][j][1],
                                                            all_sentence_info[i][j][2], all_sentence_info[i][j][3],
                                                            s[j][0],
                                                            s[j][1], all_sentence_info[i][j][4],
                                                            all_sentence_info[i][j][5],
                                                            X_eval_char_embeddings_biotags_no_endpad[i][j], None, None, None)]

                            ##add the starts to the pmcid_starts_dict
                            for i, pmcid in enumerate(all_pmcid_list):
                                if i == 0:
                                    pmcid_starts_dict[pmcid][0] = 0
                                else:
                                    pmcid_starts_dict[pmcid][0] = pmcid_starts_dict[all_pmcid_list[i - 1]][1] + 1

                            ##full output in bio format
                            output_results_path = '%s%s/%s' % (output_path, ontology, filename.replace('.h5', ''))
                            output_span_detection_results(all_pmcid_list, pmcid_starts_dict, output_path, filename,
                                                          output_results, output_results_path)



                        ##LSTM ELMO MODELS!
                        elif 'LSTM_ELMO' in algos and filename.endswith('.h5') and 'local' in filename and filename.startswith('%s_%s' %(ontology, 'LSTM_ELMO')):
                            # LSTM_ELMO_model = load_model(root + filename)

                            #gather LSTM hyperparamters - need batch_size to be able to split up the input prediction data so that
                            # optimizer, loss, neurons, epochs, batch_size = LSTM_collect_hyperparameters(ontology,save_models_path)

                            print('PROGRES: loading in model!')
                            #info for LSTM-ELMO since not enough memory
                            # batch_size = 18
                            # neurons = 512
                            # epochs = 5

                            # gather LSTM hyperparameters per ontology
                            optimizer, loss, neurons, epochs, batch_size = LSTM_collect_hyperparameters(ontology,
                                                                                                        args.save_models_path,
                                                                                                        'LSTM_ELMO')

                            with tf.Session() as session:
                                K.set_session(session)
                                session.run(tf.global_variables_initializer())
                                session.run(tf.tables_initializer())

                                elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False) ##TODO: trainable=True in original span detection
                                input_text = Input(shape=(max_sentence_length_LSTM_ELMO,), dtype=tf.string,
                                                   name="Input_layer")  # input layer
                                # ElmoEmbedding parameters: elmo_model, x, batch_size, max_len
                                # arguments={'elmo_model':elmo_model, 'batch_size':batch_size, 'max_len':max_sentence_length}

                                embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024),
                                                   arguments={'elmo_model': elmo_model, 'batch_size': batch_size,
                                                              'max_len': max_sentence_length_LSTM_ELMO}, name="Elmo_Embedding")(
                                    input_text)
                                x = Bidirectional(LSTM(units=neurons, return_sequences=True,
                                                       recurrent_dropout=0.2, dropout=0.2))(embedding)
                                x_rnn = Bidirectional(LSTM(units=neurons, return_sequences=True,
                                                           recurrent_dropout=0.2, dropout=0.2))(
                                    x)  ##units = 512 in original (TODO)
                                x = add([x, x_rnn])  # residual connection to the first biLSTM
                                out = TimeDistributed(Dense(len(all_biotags_set_LSTM_ELMO), activation="softmax"))(x)

                                LSTM_ELMO_model = Model(input_text, out)

                                if gpu_count > 1:
                                    LSTM_ELMO_model = multi_gpu_model(LSTM_ELMO_model, gpus=gpu_count)

                                LSTM_ELMO_model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
                                              metrics=["accuracy"])  # has adam in the tutorial

                                LSTM_ELMO_model.load_weights(root + filename) #load the weights

                                print('PROGRES: loaded model successfully!')



                                split_value_te = math.floor(len(X_eval_LSTM_ELMO) / batch_size) #lower number

                                X_eval_LSTM_ELMO_pred = [] #LSTM ELMO prediction from model
                                # ID to biotag
                                idx2biotag_LSTM_ELMO = {i: w for w, i in biotag2idx_LSTM_ELMO.items()}

                                print('PROGRES: predicting on new items!')
                                ##ADD 1 TO BE ABLE TO GET ALL OF IT
                                for i in range(split_value_te+1):
                                    # pred_label_j = []  # sentence level
                                    if i == split_value_te:
                                        # print('final prediction!')

                                        ##pad the length of the sentences to be the same as the batch size!
                                        if len(X_eval_LSTM_ELMO[i * batch_size:]) != batch_size:
                                            padded_sentences = []
                                            for t in range(batch_size-len(X_eval_LSTM_ELMO[i * batch_size:])):
                                                new_pad_sentence = ["__PAD__" for m in range(max_sentence_length_LSTM_ELMO)]
                                                padded_sentences += [new_pad_sentence]

                                        else:
                                            pass


                                        # print(np.array(X_eval_LSTM_ELMO[i * batch_size:]+padded_sentences).shape)
                                        ##grab it all until the end of the values
                                        print('len of padded sentences:', len(padded_sentences))
                                        p = LSTM_ELMO_model.predict(np.array(X_eval_LSTM_ELMO[i * batch_size:] + padded_sentences))[0]  ##TODO: probably will error due to index isssues if not divisible by batch_size (take until the end)
                                        # print('successful final prediction!')
                                        p = np.argmax(p, axis=-1)

                                        for j in range(i * batch_size, len(X_eval_LSTM_ELMO)):
                                            pred_label_j = []  # sentence level
                                            for w, pred in zip(X_eval_LSTM_ELMO[j], p):
                                                ##tru is a binary numpy array - need to find the index of the 1 and that is the biotag2idx label
                                                if w != '__PAD__':  # not endpad
                                                    # pred_file.write('%s\t%s\t%s\n' % (w, biotags[pred], biotags[pred]))
                                                    pred_label_j += [all_biotags_set_LSTM_ELMO[pred]]

                                            X_eval_LSTM_ELMO_pred += [pred_label_j]  # sentence level information biotags without endpad!!!

                                    else:
                                        p = LSTM_ELMO_model.predict(np.array(X_eval_LSTM_ELMO[i * batch_size:i * batch_size + batch_size]))[0]  ##TODO: probably will error due to index isssues if not divisible by batch_size (take until the end)
                                        # print('successful predictions!')
                                        p = np.argmax(p, axis=-1)

                                        for j in range(i * batch_size, i * batch_size + batch_size):
                                            pred_label_j = []
                                            for w, pred in zip(X_eval_LSTM_ELMO[j], p):
                                                ##tru is a binary numpy array - need to find the index of the 1 and that is the biotag2idx label
                                                if w != '__PAD__':  # not endpad
                                                    # pred_file.write('%s\t%s\t%s\n' % (w, biotags[pred], biotags[pred]))

                                                    pred_label_j += [all_biotags_set_LSTM_ELMO[pred]]
                                            # print('sentence_length', pred_label_j)

                                            X_eval_LSTM_ELMO_pred += [pred_label_j]  # sentence level information biotags without endpad!!!


                                if len(X_eval_LSTM_ELMO) != len(X_eval_LSTM_ELMO_pred):

                                    print(len(X_eval_LSTM_ELMO_pred))
                                    print(X_eval_LSTM_ELMO_pred[:1])
                                    raise Exception('ERROR WITH CAPTURING ALL PREDICTIONS!')

                            if gold_standard.lower() == 'true':


                                for t, y in enumerate(y_eval_LSTM_ELMO):
                                    # print('filename', filename)
                                    # print('training sentence:', training_sentences[t])
                                    # print('eval sentence:', all_sentences[t])

                                    if len(y) != len(X_eval_LSTM_ELMO_pred[t]):

                                        print('sentence', t)
                                        print(all_sentences[t])
                                        print('training sentence', training_sentences[t])
                                        print('truth', len(y))
                                        print('pred', len(X_eval_LSTM_ELMO_pred[t]))

                                        raise Exception('ERROR WITH LENGTH OF BIOTAG SEQUENCES!')

                                print('PROGRES: classification reports if gold standard!')
                                ##update the labels so that O- becomes D for discontinuity to get it in the classification report


                                updated_test_labels = ['D' if l == 'O-' else l for s in y_eval_LSTM_ELMO for l in
                                                       s]
                                updated_pred_labels = ['D' if l == 'O-' else l for s in
                                                       X_eval_LSTM_ELMO_pred for l in s]

                                ##full with O as W for words not included!
                                updated_test_labels_full = ['W' if l == 'O' else l for s in updated_test_labels for l in
                                                            s]
                                updated_pred_labels_full = ['W' if l == 'O' else l for s in updated_pred_labels for l in
                                                            s]

                                # print(updated_pred_labels_full.count('B'), updated_test_labels_full.count('B'))

                                # print(sklearn.metrics.classification_report(updated_test_labels, updated_pred_labels))
                                # print(sklearn.metrics.classification_report(updated_test_labels_full, updated_pred_labels_full))

                                # # output the classification report - B, I, O- (D), O (W)
                                # print(updated_test_labels[0])
                                # print(updated_pred_labels[0])
                                #
                                # 'PMCID', 'TYPE', 'PRECISION', 'RECALL', 'F1_MEASURE'
                                ###TODO: WEIRDNESS WITH CLASSIFICATION REPORT!!! i don't think it is correct
                                local_eval_files_creport_ELMO_LSTM.write(
                                    '%s\t%s\t%.2f\t%.2f\t%.2f\n' % (pmcid, 'MACRO-AVG',
                                                                    sklearn.metrics.precision_score(
                                                                        updated_test_labels,
                                                                        updated_pred_labels,
                                                                        average='macro'),
                                                                    sklearn.metrics.recall_score(
                                                                        updated_test_labels,
                                                                        updated_pred_labels,
                                                                        average='macro'),
                                                                    sklearn.metrics.f1_score(
                                                                        updated_test_labels,
                                                                        updated_pred_labels,
                                                                        average='macro')))

                                local_eval_files_creport_ELMO_LSTM.write(
                                    '%s\t%s\t%.2f\t%.2f\t%.2f\n' % (pmcid, 'WEIGHTED',
                                                                    sklearn.metrics.precision_score(
                                                                        updated_test_labels_full,
                                                                        updated_pred_labels_full,
                                                                        average='weighted'),
                                                                    sklearn.metrics.recall_score(
                                                                        updated_test_labels_full,
                                                                        updated_pred_labels_full,
                                                                        average='weighted'),
                                                                    sklearn.metrics.f1_score(
                                                                        updated_test_labels_full,
                                                                        updated_pred_labels_full,
                                                                        average='weighted')))



                            # TODO: tokenized files in BIO format for conept normalization
                            ##output the model in BIO format for concept normalizaiton:
                            # output_results = []  # PMCID	SENTENCE_NUM	SENTENCE_START	SENTENCE_END	WORD	POS_TAG	WORD_START	WORD_END	BIO_TAG	PMC_MENTION_ID	ONTOLOGY_CONCEPT_ID	ONTOLOGY_LABEL

                            ##need to get rid of the ENDPAD - TODO to fix!!!!!
                            pmcid_starts_dict = {}  # pmcid -> index
                            for i, s in enumerate(all_sentences):  # index, value
                                if len(s) != len(X_eval_LSTM_ELMO_pred[i]):
                                    print(len(s))
                                    print(len(X_eval_LSTM_ELMO_pred[i]))
                                    # print(len(X_eval_LSTM_ELMO_pred[i]))
                                    raise Exception('ERROR WITH PREDICTION ON EVAL SET!')
                                else:
                                    for j in range(len(s)):
                                        ##gather the end of the article to know when to move to the next article
                                        pmcid_starts_dict[all_sentence_info[i][j][0]] = [None, len(
                                            output_results)]  # last one will stay
                                        # if pmcid_starts_dict.get(all_sentence_info[i][j][0]):
                                        #     pass
                                        #
                                        # else:
                                        #     pmcid_starts_dict[all_sentence_info[i][j][0]] = len(output_results)

                                        ##output results by word
                                        output_results += [(all_sentence_info[i][j][0], all_sentence_info[i][j][1],
                                                            all_sentence_info[i][j][2], all_sentence_info[i][j][3],
                                                            s[j][0],
                                                            s[j][1], all_sentence_info[i][j][4],
                                                            all_sentence_info[i][j][5],
                                                            X_eval_LSTM_ELMO_pred[i][j], None,
                                                            None, None)]

                            ##add the starts to the pmcid_starts_dict
                            for i, pmcid in enumerate(all_pmcid_list):
                                if i == 0:
                                    pmcid_starts_dict[pmcid][0] = 0
                                else:
                                    pmcid_starts_dict[pmcid][0] = pmcid_starts_dict[all_pmcid_list[i - 1]][1] + 1

                            ##full output in bio format
                            output_results_path = '%s%s/%s' % (output_path, ontology, filename.replace('.h5', ''))
                            output_span_detection_results(all_pmcid_list, pmcid_starts_dict, output_path, filename,
                                                          output_results, output_results_path)




def biobert_model(tokenized_file_path, ontologies, save_models_path, output_path, excluded_files, gold_standard, algos, pmcid_sentence_file_path,  all_lcs_path, gpu_count=1):
    ##only take sentences with a cue in it
    # create all_lcs_dict: lc -> [regex, ignorance_type]
    all_regex = []
    if all_lcs_path:
        with open('%s' % all_lcs_path, 'r') as all_lcs_file:
            next(all_lcs_file)
            for line in all_lcs_file:
                all_regex += [line.split('\t')[1]]



    ##gather all data for all algorithms
    ##initialize all sentences
    all_sentences = []
    all_sentence_info = []

    if gold_standard.lower() == 'true':
        training_sentences_dict = {}
        true_labels_dict = {}
        for o in ontologies:
            training_sentences_dict[o] = [] #the sentences initialized
            true_labels_dict[o] = []



    for root, directories, filenames in os.walk(tokenized_file_path):
        for filename in sorted(filenames):
            if filename.endswith('.pkl') and (
                    filename.replace('.pkl', '') in excluded_files or filename.replace('.nxml.gz.pkl', '') in excluded_files):
                valid_filename = True
            elif excluded_files[0].lower() == 'all' and filename.endswith('.pkl'):
                valid_filename = True

            else:
                valid_filename = False


            if valid_filename:


                ##load the data for all sentences in the evaluation data - loop in load_data
                all_sentences, all_sentence_info, all_pmcid_list = load_data(tokenized_file_path, filename,
                                                                             all_sentences, all_sentence_info,
                                                                             excluded_files, ontologies)
                # print(all_sentences[0], all_sentence_info[0], all_pmcid_list[0])
                print('NUMBER OF SENTENCES TO EVALUATE ON:', len(all_sentences))

                if len(all_sentences) != len(all_sentence_info):
                    raise Exception('ERROR WITH GATHERING SENTENCE INFORMATION!')
                else:
                    pass


                if gold_standard.lower() == 'true':
                    for ontology in ontologies:
                        # print(tokenized_file_path)
                        ##tokenized_file_path, ontology, all_ontology_sentences, excluded_files
                        ont_training_sentences = load_data_training(tokenized_file_path.replace('/Evaluation_Files', ''),
                                                                filename,
                                                                ontology, training_sentences_dict[ontology], excluded_files)
                        training_sentences_dict[ontology] = ont_training_sentences

                print('PROGRESS: finished file:', filename)


    if gold_standard.lower() == 'true':
        ##get all the labels for each ontology
        for ontology in ontologies:
            true_labels_dict[ontology] = [sent2labels(s) for s in training_sentences_dict[ontology]]  # true answers




            #check that we have all the sentences
            if len(true_labels_dict[ontology]) != len(all_sentences):
                print(len(true_labels_dict[ontology]), len(all_sentences))
                raise Exception('ERROR WITH BIOBERT LENGTHS FOR GOLD STANDARD!')

            else:
                pass


            ##check that the sentences are the same also
            for i, s in enumerate(all_sentences):
                if len(s) != len(true_labels_dict[ontology][i]):
                    print(i, s)
                    print(len(s), len(true_labels_dict[ontology][i]))
                    raise Exception('ERROR WITH BIOBERT NUMBER OF WORDS IN EACH SENTENCE')
                else:
                    pass

                for j, w in enumerate(s):
                    training_word_info = training_sentences_dict[ontology][i][j]

                    if w[0] != training_word_info[0] or w[1] != training_word_info[1]:
                        print(w[0], training_word_info[0])
                        print(w[1], training_word_info[0])
                        print(s)
                        print(training_sentences_dict[ontology][i])
                        raise Exception('ERROR WITH BIOBERT TRAINING SENTENCES AND ALL SENTENCE WORD MATCH!')



    ##output all the words once for biobert in the tokenized files!
    X_eval_biobert = [[w[0] for w in s] for s in all_sentences]
    # if 'EXT' in ontologies[0]:
    #     output_file = 'test_EXT'
    # else:
    #     output_file = 'test'
    output_file = 'test'
    with open('%s%s/%s.tsv' % (tokenized_file_path, 'BIOBERT', output_file), 'w+') as biobert_test: #, open('%s%s/devel.tsv' % (tokenized_file_path, 'BIOBERT'), 'w+') as biobert_devel:
        for i, b_sentence in enumerate(X_eval_biobert):
            for j, b_word in enumerate(b_sentence):
                ##O as a placeholder for predictions because we need it but not correct
                biobert_test.write('%s\t%s\n' % (b_word, 'O'))
                # biobert_devel.write('%s\n' %b_word)

            biobert_test.write('\n')
            # biobert_devel.write('\n')

    if gold_standard.lower() == 'true':
        for ontology in ontologies:
            with open('%s%s/%s_test.tsv' % (tokenized_file_path, 'BIOBERT', ontology), 'w+') as ont_biobert_test:
                # use the true labels from the training information per ontology
                for i, b_sentence in enumerate(X_eval_biobert):
                    for j, b_word in enumerate(b_sentence):
                        ont_biobert_test.write('%s\t%s\n' % (b_word, true_labels_dict[ontology][i][j]))

                    ont_biobert_test.write('\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-ontologies', type=str, help='a list of ontologies to use delimited with ,')
    parser.add_argument('-excluded_files', type=str, help='a list of excluded files delimited with ,')
    parser.add_argument('-tokenized_file_path', type=str, help='the file path for the tokenized files')
    parser.add_argument('-save_models_path', type=str, help='the file path for saving the span detection models')

    parser.add_argument('-output_path', type=str, help='the file path to the results of the span detection models')



    parser.add_argument('-algos', type=str,
                        help='a list of algorithms to evaluate with models delimited with , and all uppercase')

    #OPTIONAL = DEFAULT IS NONE
    parser.add_argument('--gold_standard', type=str, help='True if gold standard available else false', default=None)
    parser.add_argument('--pmcid_sentence_files_path', type=str,
                        help='the file path to the pmicd sentence files for the ontologies', default=None)
    parser.add_argument('--all_lcs_path', type=str, help='the file path to the lexical cues for the ignorance ontology', default=None)

    args = parser.parse_args()

    # ontologies = ['CHEBI', 'CL', 'GO_BP', 'GO_CC', 'GO_MF', 'MOP', 'NCBITaxon', 'PR', 'SO', 'UBERON']

    biotags = ['B', 'I', 'O', 'O-']
    closer_biotags = ['B', 'I']

    # excluded_files = ['11532192', '17696610']

    # tokenized_file_path = '/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/Evaluation_Files/Tokenized_Files/'
    #
    # save_models_path = '/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/PythonScripts/'
    #
    # output_path = '/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/Evaluation_Files/Results_span_detection/'

    ontologies = args.ontologies.split(',')
    excluded_files = args.excluded_files.split(',')
    algos = args.algos.split(',')


    if 'BIOBERT' in algos:
        biobert_model(args.tokenized_file_path, ontologies, args.save_models_path, args.output_path, excluded_files, args.gold_standard, algos, args.pmcid_sentence_files_path, args.all_lcs_path)

        print('PROGRESS: FINISHED PREPROCESSING BIOBERT FILES!')
    else:
        print('PROGRESS: STARTED RUNNING SPAN DETECTION MODELS PER ONTOLOGY!')
        for ontology in ontologies:
            # if ontology == 'CHEBI':
            start_time = time.time()
            run_models(args.tokenized_file_path, ontology, args.save_models_path, args.output_path, excluded_files, args.gold_standard, algos, args.pmcid_sentence_files_path, args.all_lcs_path)
            print('FINAL TIME IN SECONDS:', time.time()-start_time)

        print('PROGRESS: FINISHED RUNNING SPAN DETECTION MODELS!')