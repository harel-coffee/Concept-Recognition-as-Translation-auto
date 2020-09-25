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
from sklearn.metrics import make_scorer
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.naive_bayes import GaussianNB
# from sklearn.feature_selection import SelectPercentile, f_classif
# import eli5
# from IPython.display import display
# from itertools import chain
#
# import nltk
import sklearn
import scipy.stats
# from sklearn.metrics import make_scorer, classification_report
import sklearn.metrics
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
# import multiprocessing
# from functools import partial
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, multi_gpu_model
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Model, Input, Sequential, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Conv1D, concatenate, SpatialDropout1D, GlobalMaxPooling1D, Lambda, Input
from keras.layers.merge import add
from keras_contrib.layers import CRF #interacts with sklearn CRF
# from keras_contrib.utils import save_load_utils
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
# import tensorflow.compat.v1 as tf2
# tf2.enable_eager_execution()


import math
# import h5py


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["WORD"].values.tolist(),
                                                           s["POS_TAG"].values.tolist(),
                                                           s["BIO_TAG"].values.tolist())]
        self.grouped = self.data.groupby("SENTENCE_NUM", sort=False).apply(agg_func)
        self.sentences = [s for s in self.grouped]


def load_data(tokenized_file_path, ontology, all_ontology_sentences, excluded_files):
    for root, directories, filenames in os.walk('%s%s/' % (tokenized_file_path, ontology)):
        for filename in sorted(filenames):

            ##save 2 files to fully evaluate on later
            if filename.endswith(
                    '.pkl') and 'full' not in filename and 'mention_id_dict' not in filename and filename.replace(
                    '.pkl', '') not in excluded_files:

                ##columns = ['PMCID', 'SENTENCE_NUM', 'SENTENCE_START', 'SENTENCE_END', 'WORD', 'POS_TAG', 'WORD_START', 'WORD_END', 'BIO_TAG', 'PMC_MENTION_ID', 'ONTOLOGY_CONCEPT_ID', 'ONTOLOGY_LABEL']

                pmc_tokenized_file = pd.read_pickle(root + filename)

                getter = SentenceGetter(pmc_tokenized_file)
                # print(len(getter.sentences))
                # print(type(getter.sentences))
                all_ontology_sentences += getter.sentences
                # print(len(all_ontology_sentences))
            else:
                pass
    return all_ontology_sentences


def load_pmcid_sentence_data(ontology, pmcid_sentence_file_path, excluded_files, all_regex):
    if ontology.upper() == 'IGNORANCE':
        ##all files to return
        all_sents_ids = []
        all_sents = []
        all_sent_labels = []

        preprocess_sent_ids = []
        preprocess_sent = []
        preprocess_sent_labels = []

        for root, directories, filenames in os.walk('%s' % pmcid_sentence_file_path):
            for filename in sorted(filenames):
                if filename.endswith('sentence_info.txt') and filename.split('.nxml')[0] not in excluded_files:
                    with open(root + filename) as pmcid_sentence_file:
                        next(
                            pmcid_sentence_file)  # header: PMCID	SENTENCE_NUMBER	SENTENCE	SENTENCE_INDICES	ONTOLOGY_CONCEPT_IDS_LIST
                        for line in pmcid_sentence_file:
                            (pmcid, sentence_number, sentence_list, sentence_indices,
                            ontology_concepts_ids_list) = line.split('\t')

                            ontology_concepts_ids_list = ast.literal_eval(ontology_concepts_ids_list)

                            sentence = ast.literal_eval(sentence_list)[0]

                            true_negative_example = regex_annotations(all_regex, sentence) #true or false based on regex in it or not

                            ##full set
                            all_sents_ids += ['%s_%s' % (pmcid, sentence_number)]

                            all_sents += [sentence]



                            ##the ontology labels - 1 = ignorance, 0 = not
                            if len(ontology_concepts_ids_list) > 0:
                                all_sent_labels += [1]
                            else:
                                # print('got here!')
                                all_sent_labels += [0]


                            ##preprocessed set:
                            if true_negative_example:
                                # print('got here!')
                                preprocess_sent_ids += ['%s_%s' % (pmcid, sentence_number)]

                                preprocess_sent += [sentence]


                                ##the ontology labels - 1 = ignorance, 0 = not
                                if len(ontology_concepts_ids_list) > 0:
                                    preprocess_sent_labels += [1]
                                else:
                                    # print('got here!')
                                    preprocess_sent_labels += [0]
                            else:
                                # print('no cues!!')
                                pass

        return all_sents_ids, all_sents, all_sent_labels, preprocess_sent_ids, preprocess_sent, preprocess_sent_labels

    else:
        raise Exception('ERROR: ONLY WORKING FOR THE IGNORANCE ONTOLOGY RIGHT NOW! NEED TO ADD MORE FUNCTIONALITY!')


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
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
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
            if '}or' in regex_cue or '_or' in regex_cue or '}if' in regex_cue or 'here.' in regex_cue or regex_cue in [
                'is', 'if', 'even', 'here', 'how', 'can', 'weight', 'issue', 'view', 'call']:

                if sentence[start - 1 - adding_count].isalpha() or sentence[end - adding_count].isalpha():  # end index not included
                    # print(regex_cue, ' WITHIN ',pmc_full_text[start-5:end+5])
                    pass
                else:
                    updated_cue_occurrence_list += [(start, end)]
            else:
                updated_cue_occurrence_list += [(start, end)]


        ##found cues in the sentence - doesn't matter which cues
        if updated_cue_occurrence_list:
            # print('true negative example', sentence, regex_cue)
            return True

    #no cues found in sentence
    return False

#https://towardsdatascience.com/training-a-naive-bayes-model-to-identify-the-author-of-an-email-or-document-17dc85fa630a
def classical_ML(ontology, pmcid_sentence_file_path, tokenized_file_path, save_models_path, excluded_files, corpus, all_lcs_path):
    print('PROGRESS: current ontology is', ontology)

    ##only take sentences with a cue in it
    # create all_lcs_dict: lc -> [regex, ignorance_type]
    all_regex = []
    with open('%s' % all_lcs_path, 'r') as all_lcs_file:
        next(all_lcs_file)
        for line in all_lcs_file:
            all_regex += [line.split('\t')[1]]

    ##initialize all sentences:
    all_sents_ids, all_sents, all_sent_labels, preprocess_sent_ids, preprocess_sent, preprocess_sent_labels = load_pmcid_sentence_data(ontology, pmcid_sentence_file_path,excluded_files, all_regex)
    # print(len(all_ontology_sentences))
    print('PROGRESS: gathered all sentences')
    print('all sents', len(all_sents))
    print('processed_sents', len(preprocess_sent))





    ##split set into train and test
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(all_sents, all_sent_labels, test_size=0.10, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(preprocess_sent, preprocess_sent_labels, test_size=0.10, random_state=10)


    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise Exception('ERROR WITH SENTENCE FEATURES AND LABEL LENGTHS!')

    if len(X_train_full) != len(y_train_full) or len(X_test_full) != len(y_test_full):
        raise Exception('ERROR WITH FULL SENTENCE FEATURES AND LABEL LENGTHS!!')

    print('PROGRESS: TRAINING!')
    # print(len(y_train), len(X_train))

    ##TFIDF vectorize everything!
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')  # - 0.5 before
    # features_train = vectorizer.fit_transform(X_train)  # .toarray() #we want the model to learn the vocabulary and the document frequencies by the train set, and then transform the train features into a terms-document matrix.
    # print(features_train[:10])
    features_train = vectorizer.fit_transform(X_train_full)

    # features_test = vectorizer.transform(X_test)  # .toarray() # For the test set we just want to use the learned document frequencies (idf’s) and vocabulary to transform it into a term-document matrix.
    features_test = vectorizer.transform(X_test_full)

    vocabulary = vectorizer.get_feature_names()
    # print(type(vocabulary))
    # print('vocab/feature length', len(vocabulary))

    # save the vocabulary vectorizer to be able to use going forward with transform!
    with open('%s%s/%s_%s.pkl' % (save_models_path, ontology, ontology, 'Naive_Bayes_vectorizer'),
              'wb') as vocab_tfidf_vectorizer:
        pickle.dump(vectorizer, vocab_tfidf_vectorizer)

    selector = SelectPercentile(f_classif, percentile=10)
    # selector.fit(features_train, y_train)
    selector.fit(features_train, y_train_full)

    features_train = selector.transform(features_train).toarray()
    features_test = selector.transform(features_test).toarray()

    ##save the selector:
    with open('%s%s/%s_%s.pkl' % (save_models_path, ontology, ontology, 'Naive_Bayes_selector'),
              'wb') as vocab_tfidf_selector:
        pickle.dump(selector, vocab_tfidf_selector)

    with open('%s%s/%s_%s.txt' % (save_models_path, ontology, ontology, 'Naive_Bayes_features'),
              'w') as nb_features_file:
        smaller_feature_set = list(selector.get_support())
        nb_features_file.write('%s\t%s\n' % ('WORD', 'BOOLEAN SMALLER FEATURE SET'))
        for i, w in enumerate(vocabulary):
            nb_features_file.write('%s\t%s\n' % (w, smaller_feature_set[i]))

    # print(selector.get_support())
    # print(len(selector.get_support()), len(vocabulary))

    ##train the naive bayes guassian classifier
    with open('%s%s/%s_%s.txt' % (save_models_path, ontology, ontology, 'Naive_Bayes_output'),
              'w+') as model_output_file:
        NBM_classifier = GaussianNB()

        # model = NBM_classifier.fit(features_train, y_train)
        model = NBM_classifier.fit(features_train, y_train_full)

        # print('MODEL TRAINED:', model)
        model_output_file.write('%s\t%s\n' % ('model trained:', model))
        # model_output_file.write('%s\t%s\n' % ('training size:', len(X_train)))
        # model_output_file.write('%s\t%s\n' % ('testing size:', len(X_test)))
        model_output_file.write('%s\t%s\n' % ('training size:', len(X_train_full)))
        model_output_file.write('%s\t%s\n' % ('testing size:', len(X_test_full)))
        model_output_file.write('%s\t%s\n' % ('full vocab/feature size:', len(vocabulary)))
        model_output_file.write('%s\t%s\n' % ('feature selection vocab size:', features_train.shape[1]))

        # All predictions
        ## convert the predictions into the labels
        class_labels = ['not_ignorance', 'ignorance']

        model_output_file.write('\n%s\n' % ('Class counts'))
        for i in range(len(set(all_sent_labels))):
            model_output_file.write('\t%s\t%s\n' % (class_labels[i], all_sent_labels.count(i)))

        model_output_file.write('\n')

        ##train predictions
        train_predictions = NBM_classifier.predict(features_train)
        # train_predictions_score = NBM_classifier.score(features_train, y_train)
        train_predictions_score = NBM_classifier.score(features_train, y_train_full)

        train_predictions_list = list(train_predictions)
        # print(classification_report(y_train, train_predictions_list, target_names=class_labels))
        model_output_file.write('%s\t%s\n' % ('train predictions of length:', len(train_predictions_list)))
        model_output_file.write('%s\t%s\n' % ('train prediction score:', train_predictions_score))
        # model_output_file.write(
            # '%s\n' % (classification_report(y_train, train_predictions_list, target_names=class_labels)))
        model_output_file.write(
            '%s\n' % (sklearn.metrics.classification_report(y_train_full, train_predictions_list, target_names=class_labels)))

        model_output_file.write('\n')

        # test predictions
        print('PROGRSS: TESTING!')
        test_predictions = NBM_classifier.predict(features_test)

        # test_predictions_score = NBM_classifier.score(features_test, y_test)
        test_predictions_score = NBM_classifier.score(features_test, y_test_full)

        # print(test_predictions_score)

        # print(list(test_predictions)[:10])
        # print(y_test[:10])
        # print(len(y_test))
        # print('y_test')
        # for i, y in enumerate(y_test):
        #     if y==0:
        #         print(i)
        # print(len(list(test_predictions)))
        test_predictions_list = list(test_predictions)

        # if len(y_test) != len(test_predictions_list) :
        #     raise Exception('ERROR WITH TEST PREDICTION LENGTH!')

        if len(y_test_full) != len(test_predictions_list) :
            raise Exception('ERROR WITH TEST PREDICTION LENGTH!')

        # print(classification_report(y_test, test_predictions_list, target_names=class_labels))
        model_output_file.write('%s\t%s\n' % ('test predictions of length:', len(test_predictions_list)))
        model_output_file.write('%s\t%s\n' % ('test prediction score:', test_predictions_score))
        # model_output_file.write(
            # '%s\n' % (classification_report(y_test, test_predictions_list, target_names=class_labels)))
        model_output_file.write(
            '%s\n' % (sklearn.metrics.classification_report(y_test_full, test_predictions_list, target_names=class_labels)))


        ##save the model to pkl file
        pickle.dump(model, open('%s%s/%s_%s.pkl' % (save_models_path, ontology, ontology, 'Naive_Bayes_model'),'wb'))


def crf_collect_hyperparameters(ontology, save_models_path, hyperparameter_dict):
    with open('%s%s/%s_crf_full_model_hyperparameterization.txt' % (save_models_path, ontology, ontology),
              'r+') as ont_hyperparams_file:
        for line in ont_hyperparams_file:
            if 'best params:' in line:
                params_dict = ast.literal_eval(
                    line.split('\t')[-1])  ##grab the dictionary of the best parameters for c1 and c2 for each ontology
                # print(params_dict)
                # print(type(params_dict['c1']))

                if hyperparameter_dict.get(ontology):
                    raise Exception('ERROR WITH DUPLICATE ONTOLOGY CALLS!')
                else:
                    hyperparameter_dict[ontology] = [params_dict['c1'], params_dict['c2']]

    return hyperparameter_dict


def train_crf_per_ontology(tokenized_file_path, save_models_path, excluded_files, hyperparameter_dict, ontology,
                           biotags, corpus):
    # https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/
    # https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#evaluation

    print('PROGRESS: current ontology is', ontology)

    ##initialize all sentences:
    all_ontology_sentences = []

    all_ontology_sentences = load_data(tokenized_file_path, ontology, all_ontology_sentences, excluded_files)
    # print(len(all_ontology_sentences))

    if corpus.upper() == 'CRAFT':
        corpus_cutoff = 19000
    elif corpus.upper() == 'IGNORANCE':
        corpus_cutoff = int(len(all_ontology_sentences) * 0.9)  # 10% as the validation test set
    else:
        raise Exception('ERROR: WRONG CORPUS INPUTTED')

    X_train = [sent2features(s) for s in all_ontology_sentences[:corpus_cutoff]]
    y_train = [sent2labels(s) for s in all_ontology_sentences[:corpus_cutoff]]



    # print(y_train[0])
    # # raise Exception('HOLD!')
    y_train_binary = []
    for y in y_train:
        if len(set(y)) == 1 and list(set(y))[0] == 'O':
            y_train_binary += [0]
        else:
            y_train_binary += [1]



    X_test = [sent2features(s) for s in all_ontology_sentences[corpus_cutoff:]]
    y_test = [sent2labels(s) for s in all_ontology_sentences[corpus_cutoff:]]

    y_test_binary = []
    for y in y_test:
        if len(set(y)) == 1 and list(set(y))[0] == 'O':
            y_test_binary += [0]
        else:
            y_test_binary += [1]

    if len(X_train) != len(y_train) or len(X_test) != len(y_test) or len(y_test) != len(y_test_binary) or len(y_train) != len(y_train_binary):
        raise Exception('ERROR WITH SENTENCE FEATURES AND LABEL LENGTHS!')

    print('TRAINING!')
    print(len(y_train), len(X_train))

    # print(X_train[0])
    # print(y_train[0])

    ##grab the correct hyperparameters!
    if hyperparameter_dict:
        (ont_c1, ont_c2) = hyperparameter_dict[ontology]

        ##Run a CRF - hyperparameter search: 'c1': scipy.stats.expon(scale=0.5),'c2': scipy.stats.expon(scale=0.05),
        ##partial model without the test set
        crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                  c1=ont_c1,
                  c2=ont_c2,
                  max_iterations=100,
                  all_possible_transitions=True)

        pred = model_selection.cross_val_predict(estimator=crf, X=X_train, y=y_train, cv=5)  ##5 fold cross validation

        ##Evaluate the model
        report = flat_classification_report(y_pred=pred, y_true=y_train)

        # print(report)
        crf.fit(X_train, y_train)


        ##full model (X_train + X_test)
        crf2 = sklearn_crfsuite.CRF(algorithm='lbfgs',
                   c1=ont_c1,
                   c2=ont_c2,
                   max_iterations=100,
                   all_possible_transitions=False)

        pred2 = model_selection.cross_val_predict(estimator=crf2, X=X_train + X_test, y=y_train + y_test,
                                                  cv=5)  ##5 fold cross validation
        crf2.fit(X_train + X_test, y_train + y_test)

        ##Evaluate the model
        report2 = flat_classification_report(y_pred=pred2, y_true=y_train + y_test)
        y_pred_full = crf2.predict(X_train+X_test)
        y_pred_full_binary = []
        for y in y_pred_full:
            if len(set(y)) == 1 and list(set(y))[0] == 'O':
                y_pred_full_binary += [0]
            else:
                y_pred_full_binary += [1]

        print('TESTING!')
        print(len(X_test), len(y_test))
        ##labels
        labels = list(crf.classes_)
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        ignorance_labels = ['not_ignorance', 'ignorance']

        ##FULL model: X_train+X_test
        y_pred2 = crf2.predict(X_test)
        y_pred2_binary = []
        for y in y_pred2:
            if len(set(y)) == 1 and list(set(y))[0] == 'O':
                y_pred2_binary += [0]
            else:
                y_pred2_binary += [1]

        # print(y_pred2[0])
        # raise Exception('HOLD!')
        metrics.flat_f1_score(y_test, y_pred2,
                              average='weighted', labels=labels)

        with open('%s%s/%s_%s.txt' % (
        save_models_path, ontology, ontology, 'crf_model_full_5cv_pred_report_local'), 'w+') as report_file2:
            report_file2.write('CRF FILE\n')
            report_file2.write('sentence training length: %s\n' % len(X_train + X_test))
            report_file2.write('sentence testing length: %s\n' % (len(X_test)))
            report_file2.write(report2)

            report_file2.write(('\n\n'))
            report_file2.write('%s\n' % ('FULL BINARY PREDICTIONS WITH B,I,O- ()'))
            report_file2.write(sklearn.metrics.classification_report(y_train_binary+y_test_binary, y_pred_full_binary,
                                                                     target_names=ignorance_labels, digits=3))

            report_file2.write('\n\n')
            report_file2.write('TESTING INFORMATION (CHEATING BECAUSE IN TRAINING DATA)\n')
            report_file2.write(metrics.flat_classification_report(y_test, y_pred2, labels=sorted_labels, digits=3))

            report_file2.write('\n\n')
            report_file2.write('%s\n' % ('TESTING BINARY PREDICTIONS WITH B,I,O- (CHEATING BECAUSE IN TRAINING DATA)'))
            report_file2.write(sklearn.metrics.classification_report(y_test_binary, y_pred2_binary,
                                                                         target_names=ignorance_labels, digits=3))
        # print(report2)

        # display(eli5.show_weights(crf, top=30)) #todo: errors!

        ##MOST model: X_train only
        y_pred = crf.predict(X_test)
        y_pred_binary = []
        for y in y_pred:
            if len(set(y)) == 1 and list(set(y))[0] == 'O':
                y_pred_binary += [0]
            else:
                y_pred_binary += [1]


        metrics.flat_f1_score(y_test, y_pred,
                              average='weighted', labels=labels)

        # print(metrics.flat_classification_report(
        #     y_test, y_pred, labels=sorted_labels, digits=3
        # ))

        with open('%s%s/%s_%s.txt' % (
        save_models_path, ontology, ontology, 'crf_model_most_5cv_pred_report_local'), 'w+') as report_file:
            report_file.write('CRF FILE\n')
            report_file.write('sentence training length: %s\n' % (len(X_train)))
            report_file.write('sentence testing length: %s\n' % (len(X_test)))

            report_file.write(report)
            report_file.write('\n\n')
            report_file.write('TESTING INFORMATION\n')
            report_file.write(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))

            report_file.write('\n\n')
            report_file.write('%s\n' % ('FULL BINARY PREDICTIONS WITH B,I,O-'))
            report_file.write(sklearn.metrics.classification_report(y_test_binary, y_pred_binary,
                                                                     target_names=ignorance_labels, digits=3))

        ##save the model for future use

        with open('%s%s/%s_crf_model_most_5cv_local.joblib' % (save_models_path, ontology, ontology),
                  'wb') as crf_model_output:
            pickle.dump(crf, crf_model_output)
        with open('%s%s/%s_crf_model_full_5cv_local.joblib' % (save_models_path, ontology, ontology,),
                  'wb') as crf2_model_output:
            pickle.dump(crf2, crf2_model_output)


    # #hyperparameter optimization - regularization penalties
    else:
        print('PROGRESS: STARTING HYPERPARAMETER OPTIMIZATION!')
        start_time = time.time()

        # define fixed parameters and parameters to search
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )

        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        ##labels - TODO: added
        labels = biotags

        # use the same metric for evaluation
        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted', labels=labels)

        # search
        rs = model_selection.RandomizedSearchCV(crf, params_space,
                                                cv=3,
                                                verbose=1,
                                                n_jobs=-1,
                                                n_iter=50,
                                                scoring=f1_scorer)

        rs.fit(X_train, y_train)

        # crf = rs.best_estimator_
        with open('%s%s/%s_%s.txt' %(save_models_path, ontology, ontology, 'EXT_crf_full_model_hyperparameterization'), 'w+') as crf_hyperparams_file:
            crf_hyperparams_file.write('%s\t%s\n' %('CURRENT ONTOLOGY', ontology))
            crf_hyperparams_file.write('%s\t%s\n' %('best params:', rs.best_params_))
            crf_hyperparams_file.write('%s\t%s\n' %('best CV score:', rs.best_score_))
            crf_hyperparams_file.write('%s\n' % ('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000)))
            crf_hyperparams_file.write('%s\t%s\n' %('FINAL TIME:', time.time() - start_time))


        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        '''
        results
        [Parallel(n_jobs=-1)]: Done 150 out of 150 | elapsed: 109.2min finished
        best params: {'c1': 0.1888333663259127, 'c2': 0.0031476606121170524}
        best CV score: 0.9947707778711368
        model size: 0.23M

        '''

        # _x = [s.parameters['c1'] for s in rs.grid_scores_]
        # _y = [s.parameters['c2'] for s in rs.grid_scores_]
        # _c = [s.mean_validation_score for s in rs.grid_scores_]
        #
        # fig = plt.figure()
        # fig.set_size_inches(12, 12)
        # ax = plt.gca()
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # ax.set_xlabel('C1')
        # ax.set_ylabel('C2')
        # ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
        #     min(_c), max(_c)
        # ))
        #
        # ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0, 0, 0])
        # fig.save('%s%s/%s/%s_%s.png' % (save_models_path, ontology, 'models', ontology, 'hyperparameter_search_graph'))
        #
        # print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

        print('FINAL TIME:', time.time() - start_time)



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





def fit_lstm(current_model, gpu_count, X_tr, y_tr, X_te, y_te, optimizer, loss, metrics_list, neurons, epochs, batch_size):
    with tf.device("/cpu:0"):
        # fit an LSTM network to training data
        model = Sequential()
        # model.add(LSTM(neurons, batch_input_shape=(batch_size, X_tr.shape[1], X_tr.shape[2]), stateful=True))
        # model.add(Dense(1))
        model.add(current_model)
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)


    model.compile(loss=loss, optimizer=optimizer, metrics=metrics_list, )  # 'mean_squared_error',
    # for i in range(nb_epoch):
    model.fit(X_tr, np.array(y_tr), epochs=epochs, batch_size=batch_size, validation_data=[X_te, np.array(y_te)],
              verbose=1, shuffle=False)
    model.reset_states()
    return model

def LSTM_experiment(ontology, gpu_count, current_model, repeats, X_tr, X_te, y_tr, y_te, tuning_file_output, tuning_file_output_creport, tuning_file_output_creport_full, all_biotags_set, biotag2idx, all_words_set, optimizer, loss, metrics_list, neurons, epochs, batch_size, save_models_path, save_model):
    # run a repeated experiment

    for r in range(repeats):
        # fit the model

        ##write out the parameters for the model #%('REPEAT NUM', 'OPTIMIZER', 'LOSS', 'METRICS LIST', 'NEURONS', 'EPOCHS', 'BATCH SIZE'))
        tuning_file_output.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %(r, optimizer, loss, metrics_list, neurons, epochs, batch_size))
        tuning_file_output_creport.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %(r, optimizer, loss, metrics_list, neurons, epochs, batch_size))

        tuning_file_output_creport_full.write(
            '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (r, optimizer, loss, metrics_list, neurons, epochs, batch_size))

        ##fit the model!
        lstm_model = fit_lstm(current_model, gpu_count, X_tr, y_tr, X_te, y_te, optimizer, loss, metrics_list, neurons, epochs, batch_size)

        ## TODO: save the model:
        if save_model:
            lstm_model.save('%s%s/%s_LSTM_model_local.h5' % (save_models_path, ontology, ontology))

        else:
            pass


        train_scores = lstm_model.evaluate(X_tr, np.array(y_tr), batch_size, epochs) ##training scores
        test_scores = lstm_model.evaluate(X_te, np.array(y_te), batch_size, epochs) ##test scores - also validation
        ##output the scores
        tuning_file_output.write('\t\t%s\t%s' %('training scores:', train_scores))
        tuning_file_output.write('\t\t%s\t%s' % ('testing scores:', test_scores))

        tuning_file_output_creport.write('\t\t%s\t%s\n' % ('training scores:', train_scores))
        tuning_file_output_creport.write('\t\t%s\t%s\n' % ('testing scores:', test_scores))

        tuning_file_output_creport_full.write('\t\t%s\t%s\n' % ('training scores:', train_scores))
        tuning_file_output_creport_full.write('\t\t%s\t%s\n' % ('testing scores:', test_scores))


        ##output the classificaiton report:

        test_pred = lstm_model.predict(X_te, verbose=1)
        # convert the indices to labels again!
        idx2biotag = {i: w for w, i in biotag2idx.items()}
        pred_labels = pred2label(test_pred, idx2biotag)
        test_labels = pred2label(y_te, idx2biotag)

        ##update the labels so that O- becomes D for discontinuity to get it in the classification report
        updated_test_labels = ['D' if l == 'O-' else l for s in test_labels for l in s]
        updated_pred_labels = ['D' if l == 'O-' else l for s in pred_labels for l in s]

        ##full with O as W for words not included!
        updated_test_labels_full = ['W' if l == 'O' else l for s in updated_test_labels for l in s]
        updated_pred_labels_full = ['W' if l == 'O' else l for s in updated_pred_labels for l in s]



        #output the classification report - B, I, O- (D), O (W)
        tuning_file_output_creport.write('%s\n\n' %(sklearn.metrics.classification_report(updated_test_labels, updated_pred_labels)))
        tuning_file_output_creport_full.write('%s\n\n' % (sklearn.metrics.classification_report(updated_test_labels_full, updated_pred_labels_full)))




    return 'completed!'




def initialize_train_LSTM_per_ontology(tokenized_file_path, save_models_path, excluded_files):
    print('PROGRESS: current ontology is', ontology)
    ##ALL SET UP FOR EACH MODEL

    ##initialize all sentences:
    all_ontology_sentences = []

    all_ontology_sentences = load_data(tokenized_file_path, ontology, all_ontology_sentences, excluded_files)
    print('number of sentences:', len(all_ontology_sentences))

    ##histogram of sentence lengths
    plt.style.use("ggplot")
    plt.hist([len(s) for s in all_ontology_sentences], bins=50)
    plt.savefig('%s%s/%s_LSTM_sentence_histogram.png' % (save_models_path, ontology, ontology))
    # plt.show()
    max_sentence_length = max([len(s) for s in all_ontology_sentences])
    # print('max sentence length:', max_sentence_length)
    with open('%s%s/%s_LSTM_max_length.txt' % (save_models_path, ontology, ontology),
              'w+') as max_length_file:
        max_length_file.write('%s\t%s\n' % ('ontology:', ontology))
        max_length_file.write('%s\t%s\n' % ('max sentence length:', max_sentence_length))

    ##collect all words and BIO_TAGS
    # words - save to use for prediction
    all_words_set = list(set([w[0] for s in all_ontology_sentences for w in s]))
    all_words_set.append(
        'ENDPAD')  ##added for padding purposes to get to the max length of the sentence since input needs to be the same length
    all_words_set.append('OOV_UNSEEN')  ##added for unknown/unseen words

    print('all unique words:', len(all_words_set))
    word2idx = {w: i for i, w in enumerate(all_words_set)}  # words to indices
    with open('%s%s/%s_LSTM_word2idx.pkl' % (save_models_path, ontology, ontology),
              'wb') as word2idx_output:
        pickle.dump(word2idx, word2idx_output)

    # biotags - save to use for prediction
    all_biotags_set = list(set([w[2] for s in all_ontology_sentences for w in s]))
    print('all unique tags:', len(all_biotags_set), all_biotags_set)
    biotag2idx = {t: i for i, t in enumerate(all_biotags_set)}  # bio_tags to indices
    with open('%s%s/%s_LSTM_biotag2idx.pkl' % (save_models_path, ontology, ontology),
              'wb') as biotag2idx_output:
        pickle.dump(biotag2idx, biotag2idx_output)

    ##Pad the sentences and biotags by converting them to a sequence of numbers (word2idx, biotag2idx)
    X = [[word2idx[w[0]] for w in s] for s in all_ontology_sentences]
    X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post",
                      value=word2idx['ENDPAD'])  # TODO: should be -1 or -2?
    # print(X[1])

    y = [[biotag2idx[w[2]] for w in s] for s in all_ontology_sentences]
    y = pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=biotag2idx["O"])
    # print(y[1])

    ##changing the lables of y to categorical
    y = [to_categorical(i, num_classes=len(all_biotags_set)) for i in
         y]  # converts the class labels to binary matrix where the labels are the indices of the 1 in the matrix
    # print('Y LABEL INFORMATION')
    # print(y[1])
    # print(type(y), type(y[1]))
    # print(biotag2idx, all_biotags_set)

    ##split into train and test set - 10% held out test set
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)  # 10 percent held out test size
    # print(X_te[0])
    print(len(X_tr), len(X_te), len(y_tr), len(y_te))

    return max_sentence_length, all_words_set, all_biotags_set, biotag2idx, X_tr, X_te, y_tr, y_te

def train_LSTM_per_ontology(tokenized_file_path, gpu_count, save_models_path, excluded_files, optimizer_list, loss_list, all_metrics_list, epochs_list, neurons_list, ontology, max_sentence_length, all_words_set, all_biotags_set, biotag2idx, X_tr, X_te, y_tr, y_te, batch_size, save_model):



    ##TODO: fit an LSTM by tuning! https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/


    ##schema for tuning LSTM: batches, epochs, neurons, optimizers, loss, metrics
    #only 1 repeat if we are saving the model
    if save_model:
        repeats = 1
    #otherwise take as many as you want
    else:
        repeats = 1

    # #output files: #output after each batch size is run!
    # tuning_file_output = open('%s%s/%s/%s_LSTM_tuning_info.txt' %(save_models_path, ontology, 'models', ontology,), 'w+')
    # tuning_file_output.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %('REPEAT NUM', 'OPTIMIZER', 'LOSS', 'METRICS LIST', 'NEURONS', 'EPOCHS', 'BATCH SIZE')) ##TODO: add more stuff here to print out - evaluation on validation set


    #moved batch_size out instead of the inner most for loop
    # for batch_size in batch_size_list:
    for optimizer in optimizer_list:
        print('STARTED BATCH SIZE AND OPTIMIZER: %s, %s' %(batch_size, optimizer))



        # output files: #output after each batch size is run! - 60 current situation
        with open('%s%s/%s_LSTM_tuning_info_%s_%s_part2.txt' % (save_models_path, ontology, ontology, optimizer, batch_size), 'w+') as tuning_file_output_batch_opt, open('%s%s/%s_LSTM_tuning_info_%s_%s_%s_part2.txt' % (save_models_path, ontology, ontology, optimizer, batch_size, 'classification_report'), 'w+') as tuning_file_output_batch_opt_creport, open('%s%s/%s_LSTM_tuning_info_%s_%s_%s_part2.txt' % (save_models_path, ontology, ontology, optimizer, batch_size, 'classification_report_full'), 'w+') as tuning_file_output_batch_opt_creport_full:
            tuning_file_output_batch_opt.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('REPEAT NUM', 'OPTIMIZER', 'LOSS', 'METRICS LIST', 'NEURONS', 'EPOCHS', 'BATCH SIZE'))  ##TODO: add more stuff here to print out - evaluation on validation set

            tuning_file_output_batch_opt_creport.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('REPEAT NUM', 'OPTIMIZER', 'LOSS', 'METRICS LIST', 'NEURONS', 'EPOCHS', 'BATCH SIZE'))

            tuning_file_output_batch_opt_creport_full.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
            'REPEAT NUM', 'OPTIMIZER', 'LOSS', 'METRICS LIST', 'NEURONS', 'EPOCHS', 'BATCH SIZE'))


            for loss in loss_list:
                metrics_list = [m for m in all_metrics_list if m != loss]
                for neurons in neurons_list:
                    for epochs in epochs_list:
                        ##fit an LSTM to the data using a functional API of keras - following the tutorial
                        input = Input(shape=(max_sentence_length,))
                        model = Embedding(input_dim=len(all_words_set), output_dim=50, input_length=max_sentence_length)(input)
                        model = Dropout(0.1)(model)
                        model = Bidirectional(LSTM(units=neurons, return_sequences=True, recurrent_dropout=0.1))(model) ##neurons set here as units!
                        out = TimeDistributed(Dense(len(all_biotags_set), activation="softmax"))( model)  # softmax output layer
                        model = Model(input, out)

                        ##returning the completion of the LSTM experiment
                        print('CURRENT PARAMETER SETTINGS:', batch_size, epochs, neurons)
                        completed = LSTM_experiment(ontology, gpu_count, model, repeats, X_tr, X_te, y_tr, y_te, tuning_file_output_batch_opt, tuning_file_output_batch_opt_creport, tuning_file_output_batch_opt_creport_full, all_biotags_set, biotag2idx, all_words_set, optimizer, loss, metrics_list, neurons, epochs, batch_size, save_models_path, save_model)
                        print('COMPLETED ALL REPEATS: %s, %s, %s, %s, %s' %(batch_size, optimizer, loss, neurons, epochs))





##run these models that we put in given the parameters!
def train_LSTM_CRF_per_ontology(tokenized_file_path, save_models_path, excluded_files, batch_size, optimizer, epochs, neurons, ontology, hyperparameter_dict):
    ##SAME START AS AN LSTM WITH THE DATA FORMATTING AND STUFF - TAKEN FROM train_LSTM_per_ontology
    print('PROGRESS: current ontology is', ontology)
    ##ALL SET UP FOR EACH MODEL

    ##initialize all sentences:
    all_ontology_sentences = []

    all_ontology_sentences = load_data(tokenized_file_path, ontology, all_ontology_sentences, excluded_files)
    print('number of sentences:', len(all_ontology_sentences))

    ##histogram of sentence lengths
    plt.style.use("ggplot")
    plt.hist([len(s) for s in all_ontology_sentences], bins=50)
    plt.savefig('%s%s/%s_LSTM_CRF_sentence_histogram.png' % (save_models_path, ontology, ontology))
    # plt.show()
    max_sentence_length = max([len(s) for s in all_ontology_sentences])
    # print('max sentence length:', max_sentence_length)
    with open('%s%s/%s_LSTM_CRF_max_length.txt' % (save_models_path, ontology, ontology),
              'w+') as max_length_file:
        max_length_file.write('%s\t%s\n' % ('ontology:', ontology))
        max_length_file.write('%s\t%s\n' % ('max sentence length:', max_sentence_length))

    ##collect all words and BIO_TAGS
    # words - save to use for prediction
    all_words_set = list(set([w[0] for s in all_ontology_sentences for w in s]))
    all_words_set.append(
        'ENDPAD')  ##added for padding purposes to get to the max length of the sentence since input needs to be the same length
    all_words_set.append('OOV_UNSEEN')  ##added for unknown/unseen words

    print('all unique words:', len(all_words_set))
    word2idx = {w: i + 1 for i, w in enumerate(all_words_set)}  # words to indices
    with open('%s%s/%s_LSTM_CRF_word2idx.pkl' % (save_models_path, ontology, ontology),
              'wb') as word2idx_output:
        pickle.dump(word2idx, word2idx_output)

    # biotags - save to use for prediction
    all_biotags_set = list(set([w[2] for s in all_ontology_sentences for w in s]))
    print('all unique tags:', len(all_biotags_set), all_biotags_set)
    biotag2idx = {t: i for i, t in enumerate(all_biotags_set)}  # bio_tags to indices
    with open('%s%s/%s_LSTM_CRF_biotag2idx.pkl' % (save_models_path, ontology, ontology),
              'wb') as biotag2idx_output:
        pickle.dump(biotag2idx, biotag2idx_output)

    ##Pad the sentences and biotags by converting them to a sequence of numbers (word2idx, biotag2idx)
    X = [[word2idx[w[0]] for w in s] for s in all_ontology_sentences]
    X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post",
                      value=word2idx['ENDPAD'])  # TODO: should be -1 or -2?
    # print(X[1])

    y = [[biotag2idx[w[2]] for w in s] for s in all_ontology_sentences]
    y = pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=biotag2idx["O"])
    # print(y[1])

    ##changing the lables of y to categorical
    y = [to_categorical(i, num_classes=len(all_biotags_set)) for i in
         y]  # converts the class labels to binary matrix where the labels are the indices of the 1 in the matrix
    # print('Y LABEL INFORMATION')
    # print(y[1])
    # print(type(y), type(y[1]))
    # print(biotag2idx, all_biotags_set)

    ##split into train and test set - 10% held out test set
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)  # 10 percent held out test size
    # print(X_te[0])

    ##RUN MODELS = take the best parameters from the previous LSTM training and crf training
    input = Input(shape=(max_sentence_length,))
    model = Embedding(input_dim=len(all_words_set) + 1, output_dim=100,
                      input_length=max_sentence_length)(input)  # 20-dim embedding - , mask_zero=True - error with different versions in keras
    model = Bidirectional(LSTM(units=neurons, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer

    (ont_c1, ont_c2) = hyperparameter_dict[ontology]
    crf = CRF(len(all_biotags_set))  # CRF layer - keras implementation, , c1=ont_c1, c2=ont_c2
    out = crf(model)  # output

    model = Model(input, out)

    ##TODO: use the models tuned from above for parameter settings!
    ##we use the crf loss in this case to add this as the last layer on top of the LSTM
    # print('epochs', epochs, type(epochs))
    # print('batch_size', batch_size, type(batch_size))
    # print('neurons', neurons, type(neurons))
    model.compile(optimizer=optimizer, loss=crf.loss_function, metrics=[crf.accuracy])

    history = model.fit(X_tr, np.array(y_tr), batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1, use_multiprocessing=True)

    #save the model - TODO!!! https://github.com/keras-team/keras-contrib/issues/125
    model.save('%s%s/%s_LSTM_CRF_model_local.h5' % (save_models_path, ontology, ontology))

    hist = pd.DataFrame(history.history)
    print(hist.keys()) #Index(['val_loss', 'val_crf_viterbi_accuracy', 'loss', 'crf_viterbi_accuracy'], dtype='object')
    ##plot the model in terms of accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 12))
    plt.plot(hist['crf_viterbi_accuracy'])
    plt.plot(hist['val_crf_viterbi_accuracy'])
    # plt.show()
    plt.savefig(
        '%s%s/%s_%s_%s_LSTM_CRF_accuracy.png' % (save_models_path, ontology, ontology, batch_size, optimizer))

    ##test the model on the X_te and compare test_pred with y_te
    test_pred = model.predict(X_te, verbose=1)
    idx2biotag = {i: w for w, i in biotag2idx.items()}
    pred_labels = pred2label(test_pred, idx2biotag)
    test_labels = pred2label(y_te, idx2biotag)
    # print(pred_labels[0])
    # print(test_labels[0])

    ##predictions on evaluation set: output file to explore
    # with open('%s%s/models/%s_%s_%s_LSTM_CRF_training_set_pred.txt' % (
    # save_models_path, ontology, ontology, batch_size, optimizer), 'w+') as pred_file:
    #     pred_file.write('%s\t%s\t%s\n' % ("Word", "True", "Pred"))
    for i in range(len(X_te)):
        p = model.predict(np.array([X_te[i]]))
        p = np.argmax(p, axis=-1)

        # print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
        # print('test labels:', type(y_te[i]), y_te[i])
        for w, tru_bin, pred in zip(X_te[i], y_te[i], p[0]):
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
            # pred_file.write('%s\t%s\t%s\n' % (all_words_set[w], all_biotags_set[int(tru)], all_biotags_set[pred]))

    ##output the classification report:
    with open('%s%s/%s_%s_%s_LSTM_CRF_classification_report.txt' % (
    save_models_path, ontology, ontology, batch_size, optimizer), 'w+') as pred_report_file:
        pred_report_file.write('%s\t%s\t%s\n' % (ontology, batch_size, optimizer))

        pred_report_file.write('%s\n' % (classification_report(test_labels, pred_labels)))

        ##update the labels so that O- becomes D for discontinuity to get it in the classification report
        updated_test_labels = ['D' if l == 'O-' else l for s in test_labels for l in s]
        updated_pred_labels = ['D' if l == 'O-' else l for s in pred_labels for l in s]

        ##full with O as W for words not included!
        updated_test_labels_full = ['W' if l == 'O' else l for s in updated_test_labels for l in s]
        updated_pred_labels_full = ['W' if l == 'O' else l for s in updated_pred_labels for l in s]

        pred_report_file.write('\n%s\n' %('FLATTENED CLASSIFICATION REPORTS FOR BIO- TAGS'))
        # output the classification report - B, I, O- (D), O (W)
        pred_report_file.write(
            '%s\n\n' % (sklearn.metrics.classification_report(updated_test_labels, updated_pred_labels)))

        pred_report_file.write('\n%s\n' % ('FLATTENED CLASSIFICATION REPORTS FOR FULL TAGS (D = O- AND W = 0)'))
        pred_report_file.write(
            '%s\n\n' % (sklearn.metrics.classification_report(updated_test_labels_full, updated_pred_labels_full)))


##run these models that we put take the LSTM parameters for!
def train_char_embeddings_per_ontology(tokenized_file_path, save_models_path, excluded_files, batch_size, optimizer,loss, epochs, neurons, ontology):
    ##SAME START AS AN LSTM WITH THE DATA FORMATTING AND STUFF - TAKEN FROM train_LSTM_per_ontology
    print('PROGRESS: current ontology is', ontology)
    ##ALL SET UP FOR EACH MODEL

    ##initialize all sentences:
    all_ontology_sentences = []

    all_ontology_sentences = load_data(tokenized_file_path, ontology, all_ontology_sentences, excluded_files)
    print('number of sentences:', len(all_ontology_sentences))

    ##histogram of sentence lengths
    plt.style.use("ggplot")
    plt.hist([len(s) for s in all_ontology_sentences], bins=50)
    plt.savefig('%s%s/%s_char_embeddings_sentence_histogram.png' % (save_models_path, ontology, ontology))
    # plt.show()
    max_sentence_length = max([len(s) for s in all_ontology_sentences])
    # print('max sentence length:', max_sentence_length)
    with open('%s%s/%s_char_embeddings_max_length.txt' % (save_models_path, ontology, ontology),
              'w+') as max_length_file:
        max_length_file.write('%s\t%s\n' % ('ontology:', ontology))
        max_length_file.write('%s\t%s\n' % ('max sentence length:', max_sentence_length))

    ##collect all words and BIO_TAGS and reverse dictionaries
    # words - save to use for prediction
    all_words_set = list(set([w[0] for s in all_ontology_sentences for w in s]))
    # all_words_set.append(
    #     'ENDPAD')  ##added for padding purposes to get to the max length of the sentence since input needs to be the same length
    # all_words_set.append('OOV_UNSEEN')  ##added for unknown/unseen words

    print('all unique words:', len(all_words_set))  # ENDPAD AND OOV_UNSEEN NOT INCLUDED!
    max_char_length = max([len(w) for w in all_words_set])  # the maximum number of characters in all words
    # print(max_char_length)

    word2idx = {w: i + 2 for i, w in enumerate(all_words_set)}  # words to indices
    word2idx['ENDPAD'] = 0
    word2idx['OOV_UNSEEN'] = 1

    idx2word = {i: w for w, i in word2idx.items()}  # index to the word itself (backwards of word2idx)

    with open('%s%s/%s_char_embeddings_word2idx.pkl' % (save_models_path, ontology, ontology),
              'wb') as word2idx_output:
        pickle.dump(word2idx, word2idx_output)

    # biotags - save to use for prediction
    all_biotags_set = list(set([w[2] for s in all_ontology_sentences for w in s]))
    print('all unique tags:', len(all_biotags_set), all_biotags_set)
    biotag2idx = {t: i + 1 for i, t in enumerate(all_biotags_set)}  # bio_tags to indices
    biotag2idx['ENDPAD'] = 0
    with open('%s%s/%s_char_embeddings_biotag2idx.pkl' % (save_models_path, ontology, ontology),
              'wb') as biotag2idx_output:
        pickle.dump(biotag2idx, biotag2idx_output)

    idx2biotag = {i: w for w, i in biotag2idx.items()}  # index to the biotag reverse!

    ##Map the sentences to the sequences of numbers - utilizing the mask_zero parameter in embedding layer to ignore endpad
    X_word = [[word2idx[w[0]] for w in s] for s in all_ontology_sentences]
    X_word = pad_sequences(maxlen=max_sentence_length, sequences=X_word, value=word2idx["ENDPAD"], padding='post',
                           truncating='post')

    ##set up the character embeddings by taking all the characters
    all_chars_set = set([w_i for w in all_words_set for w_i in w])
    print('LENGTH OF ALL CHARCTERS:', len(all_chars_set))
    char2idx = {c: i + 2 for i, c in enumerate(all_chars_set)}
    char2idx["OOV_UNSEEN"] = 1
    char2idx["ENDPAD"] = 0

    with open('%s%s/%s_char_embeddings_char2idx.pkl' % (save_models_path, ontology, ontology),'wb') as char2idx_output:
        pickle.dump(char2idx, char2idx_output)

    ##gather all the characters for each sentence!
    X_char = []
    for sentence in all_ontology_sentences:
        sent_seq = []
        for i in range(max_sentence_length):
            word_seq = []
            for j in range(max_char_length):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("ENDPAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))

    ##all the labels we also need to pad and convert to the embedding
    y = [[biotag2idx[w[2]] for w in s] for s in all_ontology_sentences]
    y = pad_sequences(maxlen=max_sentence_length, sequences=y, value=biotag2idx["ENDPAD"], padding='post',
                      truncating='post')

    ##SPLIT INTO TRAIN AND TEST SET with the char embeddings underneath
    X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2020)
    X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=2020)

    print('X info:', len(X_word_te), X_word_te[0], len(X_char_te), X_char_te[0])
    # raise Exception('HOLD!')

    ##CHARACTER EMBEDDING MODEL! - need to wrap parts applied to characters in a TimeDistributed layer to apply the same layers to all character sequences
    # input and embedding for words - #mask the zero to get rid of the endpad
    word_in = Input(shape=(max_sentence_length,))
    emb_word = Embedding(input_dim=len(all_words_set) + 2, output_dim=20,
                         input_length=max_sentence_length, mask_zero=True)(word_in)

    # word_embeddings = emb_word.get_weights()[0]
    # word2embeddings = {w: word_embeddings[idx] for w, idx in word2idx.items()}
    # with open('%s%s/%s/%s_char_embeddings_word2embeddings.pkl' % (save_models_path, ontology, 'models', ontology),'wb') as word2embeddings_output:
    #     pickle.dump(word2embeddings, word2embeddings_output)
    # print('WORD EMBEDDING INFO')
    # for w in word2embeddings.keys()[:5]:
    #     print(w, word2embeddings[w])



    # print('word embedding:', emb_word.shape, emb_word[0])
    # input and embeddings for characters - using the timedistributed to ensure the same layers to all characters
    char_in = Input(shape=(max_sentence_length, max_char_length,))
    emb_char = TimeDistributed(Embedding(input_dim=len(all_chars_set) + 2, output_dim=10,
                                         input_length=max_char_length, mask_zero=True))(char_in)
    # character LSTM to get word encodings by characters - using the timedistributed to ensure the same layers to all characters - 20 neurons here
    char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                    recurrent_dropout=0.5))(emb_char)

    # char_encodings = char_enc.get_weights()[0]
    # char2embeddings = {c: char_encodings[idx] for c, idx in char2idx.items()}
    # with open('%s%s/%s/%s_char_embeddings_char2embeddings.pkl' % (save_models_path, ontology, 'models', ontology),'wb') as char2embeddings_output:
    #     pickle.dump(char2embeddings, char2embeddings_output)
    #
    # print('CHAR EMBEDDING INFO')
    # for c in char2embeddings.keys()[:5]:
    #     print(c, char2embeddings[c])

    # print('char embedding:', char_enc.shape, char_enc[0])
    # main LSTM to run
    x = concatenate([emb_word, char_enc])
    x = SpatialDropout1D(0.3)(x)
    # print('CHAR and WORD EMBEDDING:', x.shape, x[0])

    main_lstm = Bidirectional(LSTM(units=neurons, return_sequences=True,
                                   recurrent_dropout=0.6))(x)
    out = TimeDistributed(Dense(len(all_biotags_set) + 1, activation="softmax"))(main_lstm)

    LSTM_model = Model([word_in, char_in], out)



    ##TRAIN THE MODEL LIKE USUAL: TODO: get the hyperparameters for each ontology from the LSTM

    LSTM_model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"])
    # fit the model - TODO: this may have an issue with the np.array.reshape()

    ##TODO: LSTM-CRF: not workign right now
    # LSTM_CRF_model = TimeDistributed(Dense(neurons, activation="relu"))(main_lstm)  # a dense layer as suggested by neuralNer
    # crf = CRF(all_biotags_set)  # CRF layer
    # LSTM_CRF_out = crf(LSTM_CRF_model)  # output - TODO: errors!
    #
    # LSTM_CRF_model = Model([word_in, char_in], LSTM_CRF_out)
    #
    # ##TODO: use the models tuned from above for parameter settings!
    # LSTM_CRF_model.compile(optimizer=optimizer, loss=crf.loss_function, metrics=[crf.accuracy])

    model_dict = {'LSTM': LSTM_model} #, 'LSTM_CRF': LSTM_CRF_model}

    for model_name in model_dict.keys():
        model = model_dict[model_name]

        history = model.fit([X_word_tr,np.array(X_char_tr).reshape((len(X_char_tr), max_sentence_length, max_char_length))], np.array(y_tr).reshape((len(y_tr), max_sentence_length, 1)), batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

        ##save model!!
        model.save('%s%s/%s_char_embeddings_%s_model_local.h5' % (save_models_path, ontology, ontology, model_name))

        hist = pd.DataFrame(history.history)
        ##plot the model in terms of accuracy
        plt.style.use("ggplot")
        plt.figure(figsize=(12, 12))
        plt.plot(hist["acc"])
        plt.plot(hist["val_acc"])
        # plt.show()
        plt.savefig('%s%s/%s_%s_%s_char_embeddings_%s_accuracy.png' % (
        save_models_path, ontology, ontology, batch_size, optimizer, model_name))

        ##test the model on the X_te and compare test_pred with y_te
        y_pred = model.predict([X_word_te, np.array(X_char_te).reshape((len(X_char_te), max_sentence_length, max_char_length))])
        # idx2biotag = {i: w for w, i in biotag2idx.items()}
        pred_labels = pred2label(y_pred, idx2biotag)

        # test_labels = pred2label(y_te, idx2biotag)
        ##Grab the true labels!
        test_labels = []
        for i in range(len(y_te)):
            # each sentence
            biotag_sentence_vector = []
            for w, t in zip(X_word_te[i], y_te[i]):
                if w != 0: #ENDPAD
                    biotag_sentence_vector += [idx2biotag[t]]
            test_labels += [biotag_sentence_vector]



        print('prediction info')
        print(len(test_labels),len(pred_labels))
        print('num B for test', sum([t.count('B') for t in test_labels]))
        print('num B for pred', sum([p.count('B') for p in pred_labels]))

        print('num I for test', sum([t.count('I') for t in test_labels]))
        print('num I for pred', sum([p.count('I') for p in pred_labels]))

        print('num O- for test', sum([t.count('O-') for t in test_labels]))
        print('num O- for pred', sum([p.count('O-') for p in pred_labels]))

        print('num O for test', sum([t.count('O') for t in test_labels]))
        print('num O for pred', sum([p.count('O') for p in pred_labels]))
        ##predictions on evaluation set: output file to explore
        # with open('%s%s/models/%s_%s_%s_char_embeddings_%s_training_set_pred.txt' % (
        #         save_models_path, ontology, ontology, batch_size, optimizer, model_name), 'w+') as pred_file:
        #     pred_file.write('%s\t%s\t%s\n' % ("Word", "True", "Pred"))
        #     for i in range(len(y_pred)):
        #         p = np.argmax(y_pred[i], axis=-1)
        #
        #         # print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
        #         # print('test labels:', type(y_te[i]), y_te[i])
        #         for w, tru_bin, pred in zip(X_word_te[i], y_te[i], p):
        #             ##tru is a binary numpy array - need to find the index of the 1 and that is the biotag2idx label
        #             if w != 0:  # not endpad
        #                 pred_file.write('%s\t%s\t%s\n' % (idx2word[w], idx2biotag[tru_bin], idx2biotag[pred]))

        ##output the classification report:
        with open('%s%s/%s_%s_%s_char_embeddings_training_%s_set_pred_report.txt' % (
                save_models_path, ontology, ontology, batch_size, optimizer, model_name), 'w+') as pred_report_file:
            pred_report_file.write('%s\t%s\t%s\n' % (ontology, batch_size, optimizer))
            print('classification report scores!')
            print(precision_score(test_labels, pred_labels))
            print(recall_score(test_labels, pred_labels))
            print(f1_score(test_labels, pred_labels))
            pred_report_file.write('%s\n' % (classification_report(test_labels, pred_labels)))
            #TODO: add a prediction report that includes the O- tag converted to B-ID (inside discontinuity) - keeping the shape the same


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
            if idx2biotag[p_i] != '__PAD__':
                out_i.append(idx2biotag[p_i])
        out.append(out_i)
    return out


##TODO: run these models!
##https://insights.insofe.com/index.php/2020/02/11/how-to-use-elmo-embedding-in-bidirectional-lstm-model-architecture/
#https://www.depends-on-the-definition.com/named-entity-recognition-with-residual-lstm-and-elmo/
def train_LSTM_ELMO_per_ontology(tokenized_file_path, save_models_path, excluded_files, batch_size, optimizer, loss, epochs, neurons, ontology, gpu_count):
    ##SAME START AS AN LSTM WITH THE DATA FORMATTING AND STUFF - TAKEN FROM train_LSTM_per_ontology
    print('PROGRESS: current ontology is', ontology)
    ##ALL SET UP FOR EACH MODEL

    ##initialize all sentences:
    all_ontology_sentences = []

    all_ontology_sentences = load_data(tokenized_file_path, ontology, all_ontology_sentences, excluded_files)
    print('number of sentences:', len(all_ontology_sentences))

    ##histogram of sentence lengths
    plt.style.use("ggplot")
    plt.hist([len(s) for s in all_ontology_sentences], bins=50)
    plt.savefig('%s%s/%s_LSTM_ELMO_sentence_histogram.png' % (save_models_path, ontology, ontology))
    # plt.show()
    max_sentence_length = max([len(s) for s in all_ontology_sentences])
    print('max sentence length:', max_sentence_length)

    with open('%s%s/%s_LSTM_ELMO_max_length.txt' % (save_models_path, ontology, ontology),
              'w+') as max_length_file:
        max_length_file.write('%s\t%s\n' % ('ontology:', ontology))
        max_length_file.write('%s\t%s\n' % ('max sentence length:', max_sentence_length))

    ##collect all words and BIO_TAGS and reverse dictionaries
    # words - save to use for prediction
    all_words_set = list(set([w[0] for s in all_ontology_sentences for w in s]))
    all_words_set.append('ENDPAD')  ##added for padding purposes to get to the max length of the sentence since input needs to be the same length
    # all_words_set.append('OOV_UNSEEN')  ##added for unknown/unseen words

    print('all unique words:', len(all_words_set))

    ##no need because we are using strings for the ELMO model
    # word2idx = {w: i + 2 for i, w in enumerate(all_words_set)}
    # word2idx["OOV_UNSEEN"] = 1 #UNK
    # word2idx["ENDPAD"] = 0 #PAD
    # with open('%s%s/%s/%s_LSTM_ELMo_word2idx.pkl' % (save_models_path, ontology, 'models', ontology),
    #           'wb') as word2idx_output:
    #     pickle.dump(word2idx, word2idx_output)

    # biotags - save to use for prediction
    all_biotags_set = list(set([w[2] for s in all_ontology_sentences for w in s]))
    print('all unique tags:', len(all_biotags_set), all_biotags_set)
    biotag2idx = {t: i for i, t in enumerate(all_biotags_set)}  # bio_tags to indices



    with open('%s%s/%s_LSTM_ELMO_biotag2idx.pkl' % (save_models_path, ontology, ontology),
              'wb') as biotag2idx_output:
        pickle.dump(biotag2idx, biotag2idx_output)


    #word ID to biotag
    idx2biotag = {i: w for w, i in biotag2idx.items()}



    ##ELMo Embeddings - need the strings as input - pad them with strings to the desired length of the sentence


    X = [[w[0] for w in s] for s in all_ontology_sentences]
    new_X = []
    for seq in X:
        new_seq = []
        for i in range(max_sentence_length):
            # new_seq.append(seq[i])
            try:
                new_seq.append(seq[i])
            except IndexError:
                new_seq.append("__PAD__") #'__PAD__' = the string padding to the max_sentence_length
        new_X.append(new_seq)

    X = new_X #setup X to be the padded sequence

    ##map the biotag labels to an integer and pad the sequences
    y = [[biotag2idx[w[2]] for w in s] for s in all_ontology_sentences]

    y = pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=biotag2idx["O"])

    ##split into train and test set:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=2020)

    ##the ELMo residual LSTM Model
    # batch_size = batch_size  # TODO: 32 in the tutorial

    print('set up the LSTM-ELMO model through tensorflow hub')

    # initialize the tensorflow session and download the pretrained model which may take some time!

    # with tf2.Session() as sess:
    #     tf2.keras.backend.set_session(sess)
    #     # elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    #     sess.run(tf2.global_variables_initializer())
    #     sess.run(tf2.tables_initializer())

    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())


        ##TODO: errors with memory so smaller batch_size, neurons, and epochs
        # batch_size = 18
        # neurons = 512
        # epochs=5

        elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        input_text = Input(shape=(max_sentence_length,), dtype=tf.string, name="Input_layer")  # input layer
        # ElmoEmbedding parameters: elmo_model, x, batch_size, max_len
        # arguments={'elmo_model':elmo_model, 'batch_size':batch_size, 'max_len':max_sentence_length}

        embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024),
                           arguments={'elmo_model': elmo_model, 'batch_size': batch_size,
                                      'max_len': max_sentence_length}, name="Elmo_Embedding")(input_text)
        x = Bidirectional(LSTM(units=neurons, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(embedding)
        x_rnn = Bidirectional(LSTM(units=neurons, return_sequences=True,
                                   recurrent_dropout=0.2, dropout=0.2))(x) ##units = 512 in original (TODO)
        x = add([x, x_rnn])  # residual connection to the first biLSTM
        out = TimeDistributed(Dense(len(all_biotags_set), activation="softmax"))(x)


        model = Model(input_text, out)

        if gpu_count > 1:
            model = multi_gpu_model(model, gpus=gpu_count)

        model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])  # has adam in the tutorial

        # TODO: split train and test - need good sizes so that it doesn't break tensorflow later - divisible by batch size!
        split_value_tr = math.floor(len(X_tr) / batch_size)
        split_value_te = math.floor(len(X_te) / batch_size)
        print(split_value_tr, type(split_value_tr), split_value_te, type(split_value_te))

        X_tr, X_val = X_tr[:split_value_tr * batch_size], X_tr[-split_value_te * batch_size:]
        y_tr, y_val = y_tr[:split_value_tr * batch_size], y_tr[-split_value_te * batch_size:]
        y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
        y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
        # y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
        # y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

        print('split test and train data successfully!')

        # TODO: need to fit the model on a GPU!
        # model_elmo = elmo_BiDirectional_model.fit(X_train, y_train, epochs=100, batch_size=128)
        # train_prediction = elmo_BiDirectional_model.predict(X_train)
        print('fitting the model')
        # print('training', np.array(X_tr).shape, print(X_tr[0]))
        # print('testing', np.array(X_val).shape, print(X_te[0]))
        # history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),
        #                     batch_size=batch_size, epochs=epochs, verbose=1)

        print('training')
        print(y_tr.shape)
        # print(y_tr[0])
        print('evaluation')
        print(y_val.shape)
        # print(y_val[0])

        history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),
                            batch_size=batch_size, epochs=epochs, verbose=1)

        # history = model.fit(np.array(X_tr), y_tr.reshape(y_tr.shape(0), y_tr), validation_data=(np.array(X_te), y_te),
        #                     batch_size=batch_size, epochs=epochs, verbose=1)

        # output the curve
        # hist = pd.DataFrame(history.history)
        # plt.figure(figsize=(12, 12))
        # plt.plot(hist["acc"])
        # plt.plot(hist["val_acc"])
        # plt.title("Learning curves")
        # plt.legend()
        # # plt.show()
        # plt.savefig(
        #     '%s%s/models/%s_%s_%s_LSTM_ELMO_accuracy.png' % (save_models_path, ontology, ontology, batch_size, optimizer))

        ##save the model weights!!!
        # try:
        #     model.save_weights('./%s%s/%s/%s_LSTM_ELMO_model_weights_local.h5' % (save_models_path, ontology, 'models', ontology))
        #     print('saved model weights successfully PART 1!')
        #
        # except:
        #     print('TRYING TO SAVE MODEL ANOTHER WAY!')

        # try:
        model.save_weights('./%s_LSTM_ELMO_model_weights_local.h5' % (ontology))
        print('saved model weights successfully in general environment!')
        # except:
        #     print('CANNOT SAVE MODEL LIKE THIS!')

        # output the predictions - TODO: make sure these are correct!
        ##predictions on evaluation set: output file to explore
        true_labels = []
        pred_labels = []  # TODO
        pred_biotag_count = {}
        pred_biotag_sentence_count = {}
        for b in all_biotags_set:
            pred_biotag_count[b] = 0
            pred_biotag_sentence_count[b] = 0

        ##y_te should be sufficient potentially with padded stuff as well - test labels - f
        # test_labels = pred2label_LSTM_ELMO(y_te, idx2biotag)  # Gets rid of the endpad stuff so it is not counting it - issues with endpad since not actually there! - len(sentence) = 410
        # print('test labels', test_labels[0:10])

        with open('%s%s/%s_%s_%s_LSTM_ELMO_training_set_pred.txt' % (
                save_models_path, ontology, ontology, batch_size, optimizer), 'w+') as pred_file:
            pred_file.write('%s\t%s\t%s\n' % ("Word", "Pred", "True"))

            ##ADD 1 TO BE ABLE TO GET ALL OF IT
            for i in range(split_value_te): #use to be split_value_te + 1 - perfect split so no need for this

                if i == split_value_te:
                    ##grab it all until the end of the values
                    ##pad the length of the sentences to be the same as the batch size!
                    if len(X_te[i * batch_size:]) != batch_size:
                        padded_sentences = []
                        for t in range(batch_size - len(X_te[i * batch_size:])):
                            new_pad_sentence = ["__PAD__" for m in range(max_sentence_length)]
                            padded_sentences += [new_pad_sentence]

                    else:
                        pass

                    # print(np.array(X_eval_LSTM_ELMO[i * batch_size:]+padded_sentences).shape)
                    ##grab it all until the end of the values
                    print('len of padded sentences:', len(padded_sentences))


                    p = model.predict(np.array(X_te[i * batch_size:] + padded_sentences))[0]
                    p = np.argmax(p, axis=-1)
                    for j in range(i * batch_size, len(X_te)):
                        pred_label_j = []  # sentence level
                        for w, tru_bin, pred in zip(X_te[j],y_te[j], p):
                            ##tru is a binary numpy array - need to find the index of the 1 and that is the biotag2idx label
                            if w != '__PAD__':  # not endpad
                                pred_file.write('%s\t%s\t%s\n' % (w, all_biotags_set[pred], all_biotags_set[tru_bin]))
                                pred_label_j += [all_biotags_set[pred]]
                        pred_labels += [pred_label_j]  # sentence level information biotags without endpad

                else:
                    p = model.predict(np.array(X_te[i * batch_size:i * batch_size + batch_size]))[
                        0]
                    p = np.argmax(p, axis=-1)

                    for j in range(i * batch_size, i * batch_size + batch_size):
                        pred_label_j = []  # sentence level
                        true_label_j = []
                        for w, tru_bin, pred in zip(X_te[j],y_te[j], p):
                            ##tru is a binary numpy array - need to find the index of the 1 and that is the biotag2idx label
                            if w != '__PAD__':  # not endpad
                                pred_file.write('%s\t%s\t%s\n' % (w, all_biotags_set[pred], all_biotags_set[tru_bin]))
                                pred_label_j += [all_biotags_set[pred]]
                                pred_biotag_count[all_biotags_set[pred]] += 1

                                true_label_j += [all_biotags_set[tru_bin]]



                        set_pred_label_j = list(set(pred_label_j))
                        for pred_b in set_pred_label_j:
                            pred_biotag_sentence_count[pred_b] += 1


                        pred_labels += [pred_label_j]  # sentence level information biotags without endpad

                        true_labels += [true_label_j] #sentence level biotags for true labels without endpad

        ##output the classification report:
        with open('%s%s/%s_%s_%s_LSTM_ELMO_training_set_pred_report.txt' % (
                save_models_path, ontology, ontology, batch_size, optimizer), 'w+') as pred_report_file:
            pred_report_file.write('%s\t%s\t%s\t%s\n\n' % ('hyperparameters:', ontology, batch_size, optimizer))
            pred_report_file.write('\%s\t%s\t%s\n' %('BIOTAG', 'SENTENCE_COUNT', 'TOTAL_COUNT'))
            for b in all_biotags_set:
                pred_report_file.write('\%s\t%s\t%s\n' %(b, pred_biotag_sentence_count[b], pred_biotag_count[b]))


            # pred_report_file.write('\n')
            # pred_report_file.write('%s\n' % (classification_report(test_labels, pred_labels)))

            pred_report_file.write('\n')
            pred_report_file.write('%s\n' %(classification_report(true_labels, pred_labels)))



        ##clear the models
        # tf2.keras.backend.clear_session()
        # tf2.reset_default_graph()

##TODO: run this mddel
def train_BERT_per_ontology(tokenized_file_path, save_models_path, excluded_files, ontology):
    ##SAME START AS AN LSTM WITH THE DATA FORMATTING AND STUFF - TAKEN FROM train_LSTM_per_ontology
    print('PROGRESS: current ontology is', ontology)
    ##ALL SET UP FOR EACH MODEL

    ##initialize all sentences:
    all_ontology_sentences = []

    all_ontology_sentences = load_data(tokenized_file_path, ontology, all_ontology_sentences, excluded_files)
    print('number of sentences:', len(all_ontology_sentences))

    ##histogram of sentence lengths
    plt.style.use("ggplot")
    plt.hist([len(s) for s in all_ontology_sentences], bins=50)
    plt.savefig('%s%s/%s/%s_BIOBERT_sentence_histogram.png' % (save_models_path, ontology, 'BIOBERT', ontology))
    # plt.show()
    max_sentence_length = max([len(s) for s in all_ontology_sentences])
    # print('max sentence length:', max_sentence_length)
    with open('%s%s/%s/%s_BIOBERT_max_length.txt' % (save_models_path, ontology, 'BIOBERT', ontology),
              'w+') as max_length_file:
        max_length_file.write('%s\t%s\n' % ('ontology:', ontology))
        max_length_file.write('%s\t%s\n' % ('max sentence length:', max_sentence_length))

    ##Create the biobert train.tsv and test.tsv
    # print(all_ontology_sentences[:10])
    # set up the words (X) and labels (y)
    X = [[w[0] for w in s] for s in all_ontology_sentences]
    y = [[b[2] for b in s] for s in all_ontology_sentences]
    # print(X[:10])
    # print(y[:10])

    # split into train and test sets
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)  # 10 percent held out test size
    # print(X_tr[:2])
    # print(y_tr[:2])

    # output the training files: train_dev = full set, train = only training, devel = testing within training, test = predicting (same as devel to get the actual predictions)
    with open('%s%s/%s/train_dev.tsv' % (save_models_path, ontology, 'BIOBERT'),
              'w+') as biobert_train_dev, \
            open('%s%s/%s/train.tsv' % (save_models_path, ontology, 'BIOBERT'), 'w+') as biobert_train, \
            open('%s%s/%s/devel.tsv' % (save_models_path, ontology, 'BIOBERT'), 'w+') as biobert_devel, \
            open('%s%s/%s/test.tsv' % (save_models_path, ontology, 'BIOBERT'), 'w+') as biobert_test:
        for i, sentence in enumerate(X_tr):
            biotags = y_tr[i]
            for j, word in enumerate(sentence):
                biotag = biotags[j]  # .replace('O-', 'X') #O- it doesnt like but it has X
                biobert_train_dev.write('%s\t%s\n' % (word, biotag))
                biobert_train.write('%s\t%s\n' % (word, biotag))

            biobert_train_dev.write('\n')
            biobert_train.write('\n')

        # with open('%s%s/%s/%s/test.tsv' % (save_models_path, ontology, 'models', 'BIOBERT'),'w+') as biobert_test:
        for i, sentence in enumerate(X_te):
            biotags = y_te[i]
            for j, word in enumerate(sentence):
                biotag = biotags[j]  # .replace('O-', 'X') #O- it doesnt like but it has X
                biobert_train_dev.write('%s\t%s\n' % (word, biotag))
                biobert_devel.write('%s\t%s\n' % (word, biotag))
                biobert_test.write('%s\t%s\n' % (word, biotag))

            biobert_train_dev.write('\n')
            biobert_devel.write('\n')
            biobert_test.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ontologies', type=str, help='a list of ontologies to use delimited with ,')
    parser.add_argument('-excluded_files', type=str,
                        help='a list of pmcids to exclude in the training delimited with ,')
    parser.add_argument('-biotags', type=str, help='a list of biotags used in the data delimited with ,')
    parser.add_argument('-closer_biotags', type=str, help='a list of closer biotags to what we want delimited with ,')
    parser.add_argument('-tokenized_file_path', type=str, help='the file path to the tokenized files to read in')
    parser.add_argument('-save_models_path', type=str,
                        help='the file path to where the models should be saved as output')
    parser.add_argument('-algo', type=str,
                        help='a list of the algorithms we are using to run for span detection delimtited with ,')
    parser.add_argument('-corpus', type=str, help='the corpus: either CRAFT or Ignorance')

    # defaults are None
    parser.add_argument('--pmcid_sentence_files_path', type=str,
                        help='the file path to the pmicd sentence files for the ontologies', default=None)
    parser.add_argument('--all_lcs_path', type=str, help='the file path to the lexical cues for the ignorance ontology', default=None)
    parser.add_argument('--crf_hyperparameters', type=str,
                        help='if hyperparameters exist set this to somethig, otherwise leave it blank and defaults to None',
                        default=None)
    parser.add_argument('--gpu_count', type=str, help='the number of gpus in the system to run (max 4 currently)',
                        default='1')
    parser.add_argument('--batch_size_list', type=str, help='a list of the batch sizes to try delimited with ,',
                        default=None)
    parser.add_argument('--optimizer_list', type=str, help='a list of the optimizers to try delimited with ,',
                        default=None)
    parser.add_argument('--loss_list', type=str, help='a list of the loss functions to try delimited with ,',
                        default=None)
    parser.add_argument('--epochs_list', type=str, help='a list of the num epochs to try delimited with ,',
                        default=None)
    parser.add_argument('--neurons_list', type=str, help='a list of the num neurons to try delimited with ,',
                        default=None)
    parser.add_argument('--save_model', type=str, help='True if we want to save the models', default=False)
    parser.add_argument('--LSTM_part2_tuning', type=str, help='True if we want to continue tuning the LSTMs using the determined batch sizes', default=False)

    args = parser.parse_args()

    ontologies = args.ontologies.split(',')
    excluded_files = args.excluded_files.split(',')
    biotags = args.biotags.split(',')
    closer_biotags = args.closer_biotags.split(',')

    algo = args.algo.split(',')
    algo = [a.upper() for a in algo] ##TODO: ALWAYS CAPITALIZED!!
    print('algos', algo)

    if args.corpus.upper() not in ['CRAFT', 'IGNORANCE']:
        raise Exception('ERROR: WRONG CORPUS - MUST BE EITHER CRAFT OR IGNORANCE!')

    if 'LSTM' in algo:

        gpu_count = int(args.gpu_count)

        ##hyperparameterization done
        if args.save_model:
            save_model = True

        ##tuning models!
        else:
            batch_size_list = [int(b) for b in args.batch_size_list.split(',')]
            optimizer_list = args.optimizer_list.split(',')
            loss_list = args.loss_list.split(',')
            epochs_list = [int(e) for e in args.epochs_list.split(',')]
            neurons_list = [int(n) for n in args.neurons_list.split(',')]
            save_model = False

        if args.LSTM_part2_tuning:
            LSTM_part2_tuning = True
        else:
            LSTM_part2_tuning = False

    else:
        pass

    # ontologies = ['CHEBI', 'CL', 'GO_BP', 'GO_CC', 'GO_MF', 'MOP', 'NCBITaxon', 'PR', 'SO', 'UBERON']
    # # ontologies = ['CHEBI']
    # excluded_files = ['11532192', '17696610']

    # biotags = ['B', 'I', 'O', 'O-']
    # closer_biotags = ['B', 'I']

    # tokenized_file_path = '/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/Tokenized_Files/'

    # save_models_path = '/Users/MaylaB/Dropbox/Documents/0_Thesis_stuff-Larry_Sonia/Negacy_seq_2_seq_NER_model/ConceptRecognition/PythonScripts/'

    ##fiji:
    # tokenized_file_path = '/Users/mabo1182/negacy_project/Tokenized_Files/'

    # save_models_path = '/Users/mabo1182/negacy_project/span_detection_models/'

    # algo = 'CRF'
    # algo = ['CRF', 'LSTM']
    # algo = ['LSTM']

    if 'CLASSICAL_ML' in algo:
        ###classical machine learning implementation with TFIDF and Naive Bayes right now
        if not args.pmcid_sentence_files_path:
            raise Exception('ERROR: NEED TO PUT IN THE FILE PATH FOR --PMCID_SENTENCE_FILE_PATH')
        else:
            for ontology in ontologies:
                classical_ML(ontology, args.pmcid_sentence_files_path, args.tokenized_file_path, args.save_models_path,
                             excluded_files, args.corpus, args.all_lcs_path)

    if 'CRF' in algo:
        ###CRF implementation

        ##get the hyperparameter dict from the output of the runs on fiji!

        hyperparameter_dict = {}  # ontology -> c1,c2
        for ontology in ontologies:
            if args.crf_hyperparameters:
                hyperparameter_dict = crf_collect_hyperparameters(ontology, args.save_models_path, hyperparameter_dict)

            train_crf_per_ontology(args.tokenized_file_path, args.save_models_path, excluded_files, hyperparameter_dict, ontology, biotags, args.corpus)

        print('PROGRESS: DONE GETTING THE HYPERPARAMETERS FOR ALL ONTOLOGIES')


        # print('PROGRESS: STARTING PARALELLIZATION')
        # ##parallelize over the ontologies:
        # pool = multiprocessing.Pool()
        # func = partial(train_crf_per_ontology, args.tokenized_file_path, args.save_models_path, excluded_files, hyperparameter_dict,)
        # pool.map(func, ontologies)
        # pool.close()
        # pool.join()

    if 'LSTM' in algo:
        ##LSTM implementation

        # batch_size_list = [318, 212, 159, 106, 53, 36, 18] #divisors of 1908 which is the validation set (10%)
        # batch_size_list = [53, 36, 18]
        # optimizer_list = ['rmsprop', 'adam']
        # optimizer_list = ['rmsprop']
        # loss_list = ['categorical_crossentropy', 'mean_squared_error']
        # loss_list = ['categorical_crossentropy']
        loss_shorthand_dict = {'categorical_crossentropy': 'cc', 'mean_squared_error': 'mse'}
        all_metrics_list = ['accuracy', 'mean_squared_error', 'categorical_crossentropy']
        # epochs_list = [10, 100, 1000]
        # epochs_list = [100, 1000]
        # neurons_list = [1, 50, 100] #units

        ##parallelizing the runs over each ontology to tune it to determine the final parameters: (ON FIJI or supercomputer)
        ##TODO: need to gather the hyperparameters for each ontology
        # pool = multiprocessing.Pool()
        # func = partial(train_LSTM_per_ontology, args.tokenized_file_path, args.save_models_path, excluded_files, batch_size_list, optimizer_list, loss_list, all_metrics_list, epochs_list, neurons_list)
        # pool.map(func, ontologies)
        # pool.close()
        # pool.join()

        ##output the report of the tuning locally
        for ontology in ontologies:
            # if ontology == 'CHEBI':
            start_time = time.time()
            print('CURRENT ONTOLOGY!', ontology)
            if save_model:
                optimizer, loss, neurons, epochs, batch_size = LSTM_collect_hyperparameters(ontology, args.save_models_path, 'LSTM')
                optimizer_list = [optimizer]
                loss_list = [loss]
                neurons_list = [neurons]
                epochs_list = [epochs]
                batch_size_list = [batch_size]

            else:
                pass


            if LSTM_part2_tuning:
                optimizer, loss, neurons, epochs, batch_size = LSTM_collect_hyperparameters(ontology,args.save_models_path, 'LSTM')

                batch_size_list = [batch_size]
            else:
                pass

            ##gather all needed training information for the LSTM
            max_sentence_length, all_words_set, all_biotags_set, biotag2idx, X_tr, X_te, y_tr, y_te = initialize_train_LSTM_per_ontology(args.tokenized_file_path, args.save_models_path, excluded_files)

            ##train the LSTM on the data = parallelize if possible - trying over batches!
            #loop over each batch_size
            for batch_size in batch_size_list:
                train_LSTM_per_ontology(args.tokenized_file_path, gpu_count, args.save_models_path, excluded_files, optimizer_list, loss_list, all_metrics_list, epochs_list, neurons_list, ontology, max_sentence_length, all_words_set, all_biotags_set, biotag2idx, X_tr, X_te, y_tr, y_te, batch_size, save_model) #batch_size now - not batch_size lise

            ##parallelized over batches
            # pool = multiprocessing.Pool()
            # func = partial(train_LSTM_per_ontology, args.tokenized_file_path, gpu_count, args.save_models_path, excluded_files, optimizer_list, loss_list, all_metrics_list, epochs_list, neurons_list, ontology, max_sentence_length, all_words_set, all_biotags_set, biotag2idx, X_tr, X_te, y_tr, y_te)
            # pool.map(func, batch_size_list)
            # pool.close()
            # pool.join()


            # LSTM_prediction_report(args.save_models_path, ontology, biotags, closer_biotags, batch_size, optimizer, loss, metrics)

            ##LSTM tuning info from outputs on fiji - in a separate script!
            # LSTM_tune_info(args.tokenized_file_path, args.save_models_path, excluded_files, batch_size_list, optimizer_list,loss_list, all_metrics_list, epochs_list, neurons_list, ontology, biotags, loss_shorthand_dict)

            print('FINAL TIME for %s:' % (ontology), time.time() - start_time)

    if 'LSTM-CRF' in algo:
        hyperparameter_dict = {}  # ontology -> c1,c2
        loss_shorthand_dict = {'categorical_crossentropy': 'cc', 'mean_squared_error': 'mse'}
        all_metrics_list = ['accuracy', 'mean_squared_error', 'categorical_crossentropy']
        for ontology in ontologies:
            #gather all crf hyperparameters
            hyperparameter_dict = crf_collect_hyperparameters(ontology, args.save_models_path, hyperparameter_dict)

            #gather all LSTM hyperparameters
            optimizer, loss, neurons, epochs, batch_size = LSTM_collect_hyperparameters(ontology, args.save_models_path, 'LSTM')

            train_LSTM_CRF_per_ontology(args.tokenized_file_path, args.save_models_path, excluded_files, batch_size, optimizer, epochs, neurons, ontology, hyperparameter_dict)


    if 'CHAR_EMBEDDINGS' in algo:
        # print('got here!!')
        loss_shorthand_dict = {'categorical_crossentropy': 'cc', 'mean_squared_error': 'mse'}
        all_metrics_list = ['accuracy', 'mean_squared_error', 'categorical_crossentropy']
        for ontology in ontologies:
            # print('ontology', ontology)
            # gather all LSTM hyperparameters
            optimizer, loss, neurons, epochs, batch_size = LSTM_collect_hyperparameters(ontology, args.save_models_path, 'LSTM')

            train_char_embeddings_per_ontology(args.tokenized_file_path, args.save_models_path, excluded_files, batch_size, optimizer,loss, epochs, neurons, ontology)


    if 'LSTM_ELMO' in algo:
        print('model: LSTM_ELMO')
        for ontology in ontologies:

            #gather LSTM hyperparameters per ontology
            optimizer, loss, neurons, epochs, batch_size = LSTM_collect_hyperparameters(ontology, args.save_models_path, 'LSTM_ELMO')

            gpu_count = int(args.gpu_count)

            #run the LSTM_ELMO
            train_LSTM_ELMO_per_ontology(args.tokenized_file_path, args.save_models_path, excluded_files, batch_size, optimizer, loss, epochs, neurons, ontology, gpu_count)

            print('finished LSTM-ELMO model!')


    if 'BERT' in algo:
        for ontology in ontologies:
            start_time = time.time()
            print('CURRENT ONTOLOGY!', ontology)

            train_BERT_per_ontology(args.tokenized_file_path, args.save_models_path, excluded_files, ontology)
            print('FINAL TIME for %s:' % (ontology), time.time() - start_time)
