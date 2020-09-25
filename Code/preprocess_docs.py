import os

import nltk
# nltk.download()
import nltk.data

import xml.etree.ElementTree as ET

from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

from nltk.tokenize import WordPunctTokenizer
import copy
import pickle
import csv
import argparse
import string


def sentence_tokenize(pmc_doc_text, pmc_doc_file):
    sentence_list = sent_tokenize(pmc_doc_text) #list of sentences (placement in the list is the sentence number
    sentence_list_indicies = [] #tuples of start and stop
    index = None
    for t in range(len(sentence_list)):
        s = sentence_list[t]

        if sentence_list_indicies:


            start = pmc_doc_text.index(s, sentence_list_indicies[-1][1]) #the sentence has to start after the last one added ended
            end = start + len(s)


            ##combining 2 sentences because some concepts span 2 sentences which is a problem
            if pmc_doc_file in  ['12585968.txt', '14723793.txt']:
                if s.endswith('Genic.') and start == 6553:
                    # print('got here')
                    index = t
                elif 'Sm.' in s and start == 5248:
                    index = t


        else:
            start = pmc_doc_text.index(s)
            end = start + len(s)

        sentence_list_indicies += [(start, end)]

    if index is not None:
        sentence_combo = sentence_list[index] + ' ' +  sentence_list[index+1]
        sentence_indicies_combo = [(sentence_list_indicies[index][0], sentence_list_indicies[index+1][1])]


        sentence_list = sentence_list[:index] + [sentence_combo] + sentence_list[index+2:]
        sentence_list_indicies = sentence_list_indicies[:index] + sentence_indicies_combo + sentence_list_indicies[index+2:]



    if len(sentence_list_indicies) != len(sentence_list):
        raise Exception('ERROR WITH SENTENCE INDICIES!')

    return sentence_list, sentence_list_indicies


def word_tokenize_sentences(pmcid, sentence, sentence_indicies, sentence_number, pmc_doc_text, mention_ID_dict, final_concept_count, running_concept_count, multi_label_count):
    # print(sentence_number)

    all_sent_word_info = []

    # word_indicies = list(TreebankWordTokenizer().span_tokenize(sentence)) #from the sentence starting so need add the start of the sentence
    word_indicies = list(WordPunctTokenizer().span_tokenize(sentence))  # from the sentence starting so need add the start of the sentence


    sentence_start = int(sentence_indicies[0])
    sentence_end = int(sentence_indicies[1])

    if running_concept_count <= final_concept_count:
        ##see if there is an annotation to this span - add the sentence number to mention_ID_dict to know it is done
        sentence_concepts = [] #mention_IDs that are in the sentence
        # dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number - NONE for now)
        for mention_ID in mention_ID_dict.keys():
            start_list0, end_list0, spanned_text0, mention_class_ID0, class_label0, sentence_number0 = mention_ID_dict[mention_ID]



            ##TODO: ISSUE WITH A CONCEPT THAT SPANS MULTIPLE SENTENCES - NEED TO COMBINE THE SENTENCES

            if sentence_number0 is None:

                ##disjoint cues - in order so we can take the first start and last end
                first_start = int(start_list0[0])
                last_end = int(end_list0[-1])
                if first_start >= sentence_start and last_end <= sentence_end:


                    ##adding sentence number to the annotations!
                    sentence_concepts += [mention_ID]
                    mention_ID_dict[mention_ID][5] = '%s_%s' %(pmcid, sentence_number)
                    running_concept_count += 1

                    continue

                else:
                    pass




    ##FINAL DOCUMENT WORD INDICIES
    doc_word_indicies = []
    for (s,e) in word_indicies:
        doc_word_indicies += [(sentence_start + s, sentence_start + e)]

    #check that the word indices match up with the end of the sentence
    if doc_word_indicies[-1][1] != sentence_end:
        print(sentence_end, doc_word_indicies[-1][1])
        raise Exception("ERROR WITH UPDATING THE WORD INDICIES TO BE FOR THE WHOLE DOCUMENT USING THE SENTENCE STARTS!")


    ##FINAL WORD TOKENS
    # word_tokens = list(TreebankWordTokenizer().tokenize(sentence))
    word_tokens = list(WordPunctTokenizer().tokenize(sentence))



    if len(doc_word_indicies) != len(word_tokens):
        raise Exception('ERROR WITH WORD TOKENIZING INTO WORDS AND INDICIES')



    ##FINAL POS TAGS
    word_tokens_pos_tags = nltk.pos_tag(word_tokens) #list of tuples of [(word, POS)]


    if len(word_tokens) != len(word_tokens_pos_tags):
        raise Exception("ERROR WITH WORD TOKENIZING AND POS TAGS!")



    ## connect the concept annotations to the word level: #dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)
    if running_concept_count <= final_concept_count and sentence_concepts:
        word_concept_indices = [] #tuple [(word indices in the tokens, concept mention id)]


        for m in sentence_concepts:

            (start_list_1, end_list_1, spanned_text_1, mention_class_ID_1, class_label_1, sentence_number_1) = mention_ID_dict[m]


            start_list_1_indices = []
            for a in range(len(start_list_1)):
                s2 = start_list_1[a]
                e2 = end_list_1[a]

                match = False


                for i in range(len(doc_word_indicies)):
                    (s1, e1) = doc_word_indicies[i]

                    # TODO: errors with weird characters so only checking if its alpha!
                    if pmc_doc_text[s1:e1].isalpha() and pmc_doc_text[s1:e1] != word_tokens[i]:
                        print(pmc_doc_text[s1:e1], word_tokens[i])
                        raise Exception("ERROR WITH CHECKING THAT THE SPANS PULL OUT THE CORRECT TEXT!")


                    ## initialize all_sent_word_info
                    if len(all_sent_word_info) < len(word_tokens):
                        (word, pos_tag) = word_tokens_pos_tags[i]


                        ##TODO: update the BIO_list so that it is not a list and capture if it will become a list
                        all_sent_word_info += [[pmcid, sentence_number_1, sentence_start, sentence_end, word, pos_tag, s1, e1, [], [], [], []]]  #pmcid, sentence_number, sentence_start, sentence_end, word, pos_tag, word_start, word_end, BIO list, mention ID list, concept_ID, class label list


                    ## gather the start indices and the conception mention id to update and add to - issues if starting mid word (see elif)
                    #     pass
                    if int(s2) == s1 and match == False:

                        match = True
                        start_list_1_indices += [(i, m)]
                        # continue
                    elif s1 > int(s2) and match == False: #previous one because it is included in it. ex: pigmented in 'hyperpigmented'


                        start_list_1_indices += [(i-1, m)]
                        match = True
                        # continue

                    elif s1 <= int(s2) and int(e2) <= e1 and match == False:
                        # print('inside match on the end')
                        start_list_1_indices += [(i, m)]
                        match = True
                        # raise Exception('hold')

            word_concept_indices += [start_list_1_indices]





        ##MAKE SURE ALL THE CONCEPTS ARE CAPTURED!
        if len(word_concept_indices) != len(sentence_concepts):
            raise Exception('ERROR WITH ALL CONCEPTS BEING IDENTIFIED USING THE STARTS OF THE WORDS AND CONCEPTS')



        ##LOOK AT THE STARTS AND UPDATE THE WORD INFO TO INCLUDE BIO FORMAT - TODO: update to only take one per word - largest span
        # if specific_sentence:
        #     print('WORD CONCEPT INDICES', word_concept_indices)
        # print(word_concept_indices)
        # print(sentence_concepts)

        for w in word_concept_indices:
            # print(w)


            i = w[0][0] #starting start
            m = w[0][1] #starting concept

            # print(i,m) - the first start

            ##sentence info - that we are adding to - word info
            pmcid3, sentence_number3, sentence_start3, sentence_end3, word3, pos_tag3, s3, e3, bio_list3, mention_IDs3, concept_IDs3, class_labels3 = all_sent_word_info[i]


            ##mention_ID info
            start_list_2, end_list_2, spanned_text_2, mention_class_ID_2, class_label_2, sentence_number_2 = mention_ID_dict[m]



            ##update the word level information with BIO format and stuff

            ##one word concepts with no spaces: #len(spanned_text_2.split(' ')) == 1
            if len(start_list_2) == 1 and len(list(WordPunctTokenizer().tokenize(spanned_text_2))) == 1:
                # check that the span is good
                if word3 in spanned_text_2 or spanned_text_2 in word3:
                    #(s3 >= int(start_list_2[0]) or e3 <= int(end_list_2[0])) and
                    all_sent_word_info[i][8] += ['B']
                    all_sent_word_info[i][9] += [m]
                    all_sent_word_info[i][10] += [mention_class_ID_2]
                    all_sent_word_info[i][11] += [class_label_2]

                else:
                    print(s3, start_list_2[0], e3, end_list_2[0], word3, spanned_text_2)
                    print(m)
                    raise Exception('ISSUE: WEIRD CHECKING THE SPANS!!')



            ##multiword concepts with spaces but no ... (not discontinuous/disjoint) - TODO: update the format
            elif '...' not in spanned_text_2 and len(list(WordPunctTokenizer().tokenize(spanned_text_2))) > 1:


                multiword_puncttokenize = list(WordPunctTokenizer().tokenize(spanned_text_2))
                # print(multiword_puncttokenize)

                ##initial start of the multiword
                all_sent_word_info[i][8] += ['B']
                all_sent_word_info[i][9] += [m]
                all_sent_word_info[i][10] += [mention_class_ID_2]
                all_sent_word_info[i][11] += [class_label_2]

                #the rest is with an I for inside
                for j in range(i+1, i+len(multiword_puncttokenize)):
                    all_sent_word_info[j][8] += ['I']
                    all_sent_word_info[j][9] += [m]
                    all_sent_word_info[j][10] += [mention_class_ID_2]
                    all_sent_word_info[j][11] += [class_label_2]



            ##discontinuous/disjoint concepts:
            else:
                if '...' not in spanned_text_2:
                    raise Exception('MISSING A CONCEPT TYPE!!')
                else:
                    pass



                disjoint_concept = spanned_text_2.split(' ... ') #error with ignorance stuff because needs a space on either side of ...


                for d in range(len(disjoint_concept)):
                    disjoint_piece = list(WordPunctTokenizer().tokenize(disjoint_concept[d]))

                    ##initial start of the discontinuous
                    if d == 0:
                        all_sent_word_info[w[d][0]][8] += ['B'] ##a flag for discontinuity starts
                        all_sent_word_info[w[d][0]][9] += [m]
                        all_sent_word_info[w[d][0]][10] += [mention_class_ID_2]
                        all_sent_word_info[w[d][0]][11] += [class_label_2]
                    else:
                        all_sent_word_info[w[d][0]][8] += ['B-'] ##changed to be included for the subequent starts
                        all_sent_word_info[w[d][0]][9] += [m]
                        all_sent_word_info[w[d][0]][10] += [mention_class_ID_2]
                        all_sent_word_info[w[d][0]][11] += [class_label_2]



                    for j in range(w[d][0]+1, w[d][0]+len(disjoint_piece)):

                        if d == 0:
                            all_sent_word_info[j][8] += ['I']
                            all_sent_word_info[j][9] += [m]
                            all_sent_word_info[j][10] += [mention_class_ID_2]
                            all_sent_word_info[j][11] += [class_label_2]
                        else:
                            all_sent_word_info[j][8] += ['I-']
                            all_sent_word_info[j][9] += [m]
                            all_sent_word_info[j][10] += [mention_class_ID_2]
                            all_sent_word_info[j][11] += [class_label_2]




        # all_sent_word_info:  [[pmcid, sentence_number, sentence_start, sentence_end, word, pos_tag, word_start, word_end, BIO list, mention ID list, concept_ID, class label list]]
        for t in range(len(all_sent_word_info)):
            # print(all_sent_word_info[t][8])
            if not all_sent_word_info[t][8]:
                # print('GOT HERE!')
                all_sent_word_info[t][8] += ['O']
                all_sent_word_info[t][9] += [None]
                all_sent_word_info[t][10] += [None]
                all_sent_word_info[t][11] += [None]




    ##CHECK THAT THE SPANS ARE CORRECT if they have no concepts in it:
    else:
        for i in range(len(doc_word_indicies)):
            (s1, e1) = doc_word_indicies[i]

            # TODO: errors with weird characters so only checking if its alpha!
            if pmc_doc_text[s1:e1].isalpha() and pmc_doc_text[s1:e1] != word_tokens[i]:
                print(pmc_doc_text[s1:e1], word_tokens[i])
                raise Exception("ERROR WITH CHECKING THAT THE SPANS PULL OUT THE CORRECT TEXT!")

            (word, pos_tag) = word_tokens_pos_tags[i]
            ## all_sent_word_info:  [[pmcid, sentence_number, sentence_start, sentence_end, word, pos_tag, word_start, word_end, BIO list, mention ID list, concept_ID, class label list]]
            all_sent_word_info += [[pmcid, '%s_%s' %(pmcid,sentence_number), sentence_start, sentence_end, word, pos_tag, s1, e1, ['O'], [None], [None], [None]]]  # outside



    ##check that all the lists are the same length - need flag for discontinuity for the normalization piece
    all_sent_word_info_full = copy.deepcopy(all_sent_word_info) ##the full information to output and save





    ## all_sent_word_info:  [[pmcid, sentence_number, sentence_start, sentence_end, word, pos_tag, word_start, word_end, BIO list, mention ID list, concept_ID, class label list]]
    ##dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)

    disc_end = False
    # disc_search_on = False
    disc_count = 0
    other_disc_count = 0
    no_concepts_count = 0
    overlap_count = 0
    for t in range(len(all_sent_word_info)):
        ##update the bio_tag list to get rid of discontinuity tags
        for n, i in enumerate(all_sent_word_info[t][8]):
            if i == 'B-' or i == 'I-':
                all_sent_word_info[t][8][n] = 'I'
                other_disc_count += 1
            elif i == 'O':
                no_concepts_count += 1



        ##MULTILABEL BIO LIST SOLUTION: favor B
        ##if the bio_list is more than one then we need to figure out what is going on
        if len(all_sent_word_info[t][8]) > 1:
            overlap_count += 1

            ###OVERLAP PROBLEM WHERE CONCEPTS ARE INTERACTING AND BOTH NEED TO BE THERE!
            ##concensus on the label
            if len(set(all_sent_word_info[t][8])) == 1:
                ##Take this BIO information because it is correct
                all_sent_word_info[t][8] = [all_sent_word_info[t][8][0]] #list of length 1
            ##not consistent -> take B because we want to know when the beginnings of words happen to make sure we capture almost everything.
            else:
                ##B is in all overlaps so far because there is a new concept interaction that is causing the issue.
                if 'B' not in set(all_sent_word_info[t][8]):
                    raise Exception('MISSING BEGINNING IN THE BIO LIST!')
                ##set the concept to B to signify a new word
                else:
                    all_sent_word_info[t][8] = ['B']




        ##DISCONTINUOUS SPANS SOLUTION: O -> O-
        ##words with concepts included
        if None not in all_sent_word_info[t][9]:
            for m in range(len(all_sent_word_info[t][8])):
                pmc_mention_id = all_sent_word_info[t][9][m]
                # print(pmc_mention_id)

                ##discontinuous span
                if '...' in mention_ID_dict[pmc_mention_id][2]:
                    # print('spanned text', mention_ID_dict[pmc_mention_id][2])
                    # print('sentence', sentence)


                    disc_start = min(mention_ID_dict[pmc_mention_id][0])
                    disc_end = max(mention_ID_dict[pmc_mention_id][1])
                    # raise Exception('PAUSE!')
                else:
                    pass

        ##words with no concepts included just outside words
        else:
            ##make sure all outside words only have an 'O'
            if all_sent_word_info[t][8] != ['O']:
                raise Exception('ERROR WITH OUTSIDE TERMS!')

            ##otherwise if there is discontinuity seen meaning there is a disc_end
            elif disc_end:
                ##word_end <= disc_end - included in discontinuity
                if int(all_sent_word_info[t][7]) <= int(disc_end):
                    all_sent_word_info[t][8] = ['O-']
                else:
                    pass




        if 'O-' in all_sent_word_info[t][8]:
            disc_count += 1
            # print('updated_all_sent_word_info:', all_sent_word_info[t])
            # raise Exception('PAUSE!')

        if len(all_sent_word_info[t][8]) > 1:
            raise Exception('ERROR WITH MAKING SURE ONLY ONE BIO-TAG REMAINS!')
        else:
            all_sent_word_info[t][8] = all_sent_word_info[t][8][0]



    ##check that everything is the same length - maybe won't be if i only change the ID
    if len(all_sent_word_info) != len(word_tokens) and len(all_sent_word_info_full) != len(word_tokens):
        print(len(all_sent_word_info), len(word_tokens))
        raise Exception('ERROR WITH KEEPING TRACK OF ALL_SENT_WORD_INFO WHEN ADDING CONCEPTS')


    ##per sentence
    return all_sent_word_info, running_concept_count, mention_ID_dict, multi_label_count, all_sent_word_info_full, disc_count, other_disc_count, no_concepts_count, overlap_count


def gather_concept_annotations(ontology, concept_annotation_path, filename, corpus, excluded_ignorance_types, all_lcs_dict, extensions):
    '''gathers all concept annotations no matter if overlaps depending on corpus'''

    if corpus.upper() == 'CRAFT':
        if extensions:
            ont_concept_annotation_path = concept_annotation_path + ontology.replace('_EXT', '') + '/' + ontology.replace('_EXT', '+extensions') + '/knowtator/' + filename + '.knowtator.xml'
        else:
            ont_concept_annotation_path = concept_annotation_path + ontology + '/' + ontology + '/knowtator/' + filename + '.knowtator.xml'


        annotation_tree = ET.parse(ont_concept_annotation_path)
        root = annotation_tree.getroot()
        # print(root.tag)
        # print(root.attrib)

        mention_ID_dict = {} #dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)
        no_span_mention_IDs_list = []

        for annotation in root.iter('annotation'):
            child_count = 0
            start_list = []
            end_list = []
            for child in annotation:
                child_count += 1
                # print(type(child.tag))
                # print(child.tag, child.attrib, child.text)

                if child.tag == 'mention':
                    mention_ID = child.attrib['id']

                ##TODO: disjoint cues exist! multiple spans!
                elif child.tag == 'span':
                    start = child.attrib['start']
                    start_list += [start]
                    end = child.attrib['end']
                    end_list += [end]

                elif child.tag == 'spannedText': #disjiont cues have ... in text
                    spanned_text = child.text
                    mention_ID_dict[mention_ID] = [start_list, end_list, spanned_text, None, None, None]


                else:
                    pass



            if not mention_ID_dict.get(mention_ID):
                no_span_mention_IDs_list += [mention_ID]
            else:
                pass

            """
            <annotation>
            <mention id="CHEBI_2015_08_26_Instance_170705" />
            <annotator id="CHEBI_2015_08_26_Instance_10000">Mike Bada, University of Colorado Anschutz Medical Campus</annotator>
            <span start="48508" end="48524" />
            <span start="48538" end="48545" />
            <spannedText>liver X receptor ... agonist</spannedText>
          </annotation>
            """


        if no_span_mention_IDs_list:
            print(no_span_mention_IDs_list)
        else:
            pass

        for class_mention in root.iter('classMention'):
            class_mention_ID = class_mention.attrib['id']
            if class_mention_ID in no_span_mention_IDs_list:
                print('got here')
                pass
            else:
                for child in class_mention:
                    if child.tag == 'mentionClass':
                        mention_class_ID = child.attrib['id'] #this is the ontology concepts
                        class_label = child.text

                        mention_ID_dict[class_mention_ID][3] = mention_class_ID
                        mention_ID_dict[class_mention_ID][4] = class_label



    elif corpus.upper().startswith('IGNORANCE'):
        # print(concept_annotation_path)
        # print(filename)
        ##create the annotation file path from the gold standard
        annotation_file_path = '%s%s' % (concept_annotation_path, filename.replace('.txt', '.xml'))
        # print(corpus, annotation_file_path)
        ##gold standard has the annotator names in it!
        #gold standard with all of us: emily, elizabeth, mayla
        if filename.split('.nxml')[0] in ['PMC1474522', 'PMC3205727']:
            annotation_file_path += '.EMILY_ELIZABETH_MAYLA.xml'

        #gold standard just elizabeth and mayla
        elif corpus.upper() == 'IGNORANCE':
            # print('got here')
            annotation_file_path += '.ELIZABETH_MAYLA.xml'

        #development set with no people's names
        else:
            pass


        with open(annotation_file_path, 'r') as annotation_file:
            tree = ET.parse(annotation_file)
            root = tree.getroot()
            # print(root)

            mention_ID_dict = {}  # dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)

            ##loop over all annotations
            for annotation in root.iter('annotation'):
                annotation_id = annotation.attrib['id']
                # print('annotation id', annotation_id)
                start_list = []
                end_list = []
                spanned_text = ''
                empty_annotation = False #empty annotations or subject_scopes for right now to do binary
                ##loop over all annotation information
                for child in annotation:
                    # print(child)
                    if child.tag == 'class':
                        ont_lc = child.attrib['id']
                        # print('ont_lc', ont_lc)

                        ##binary ignorance!
                        if ont_lc != 'subject_scope':
                            it = 'ignorance'  # ignorance type category
                        else:
                            empty_annotation = True
                    elif child.tag == 'span':
                        ##empty annotation - delete
                        if not child.text:
                            # print(child)
                            empty_annotation = True
                            # raise Exception('ERROR WITH EMPTY ANNOTATION')
                        else:
                            span_start = child.attrib['start'] #int
                            span_end = child.attrib['end'] #int
                            start_list += [span_start]
                            end_list += [span_end]
                            if spanned_text:
                                # discontinuous spans
                                spanned_text += ' ... %s' %child.text #str #this is how disjoint cues are in the CRAFT stuff
                            else:
                                spanned_text += '%s' %child.text #str

                    else:
                        print('got here weirdly')
                        raise Exception('ERROR WITH READING IN THE ANNOTATION FILES!')
                        pass

                if empty_annotation:
                    continue
                else:
                    #not an empty annotation - should be no duplicates because gold standard
                    ##fill the dictionary with the info
                    #ensure there are spans!
                    if start_list and end_list:
                        ## dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)
                        if excluded_ignorance_types:
                            #don't want to include the annotation id if it is in the excluded ignorance types
                            # print(all_lcs_dict[ont_lc])
                            if all_lcs_dict.get(ont_lc) and len(set(excluded_ignorance_types) & set(all_lcs_dict[ont_lc])) > 0:
                                # print('got here:', ont_lc, all_lcs_dict[ont_lc])
                                pass
                            elif ont_lc.upper() in excluded_ignorance_types:
                                # print('here also:', ont_lc)
                                pass
                            else:
                                mention_ID_dict[annotation_id] = [start_list, end_list, spanned_text, ont_lc, it, None]

                        else:
                            mention_ID_dict[annotation_id] = [start_list, end_list, spanned_text, ont_lc, it, None]
                        # print(mention_ID_dict[annotation_id])
                    else:
                        continue


    else:
        raise Exception('ERROR ON THE CORPUS CHOSEN: PLEASE CHOOSE CRAFT OR IGNORANCE!')





    ##check that all is captured for the BIO stuff

    for m in mention_ID_dict.keys():
        if None in mention_ID_dict[m][:5]:
            print(mention_ID_dict[m])
            raise Exception('ERROR WITH CAPTURING ALL CONCEPT INFORMATION FROM ANNOTATION FILES!')



    return mention_ID_dict




def preprocess_docs(craft_path, articles, concept_annotation, ontologies, output_path, pmcid_sentence_path, corpus, excluded_ignorance_types, all_lcs_path, extensions):

    max_files = 10
    file_num = 0
    # print('MAX FILES: ', max_files)


    ##capture the discontinuities per ontology
    if extensions:
        disc_output_path = output_path.replace('Tokenized_Files','PythonScripts') + 'discontinuous_info_per_ontology_EXT.txt'
    else:
        disc_output_path = output_path.replace('Tokenized_Files', 'PythonScripts') + 'discontinuous_info_per_ontology.txt'


    disc_output = open(disc_output_path, 'w+')
    disc_output.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %('ONTOLOGY', 'TOTAL_SENTENCE_COUNT','TOTAL_SENTENCE_DISCONTINUOUS_COUNT' , 'TOTAL_DISCONTINUOUS_WORDS', 'PERCENT_DISCONTINUOUS_SENTENCES', 'TOTAL_OTHER_DISCONTINUOUS_COUNT', 'TOTAL_OVERLAP_COUNT', 'TOTAL_WORD_COUNT', 'TOTAL_WORD_CONCEPT_COUNT',))

    ##FOR TFIDF PREDICTION AND CLASSIFICATION ON CRAFT
    pmcid_sentence_dict = {} #(pmcid, sentence_num) -> [sentence, sentence_indices, [ontology_concepts]]

    for ontology in ontologies:
        print('PROGRESS: current ontology is', ontology)
        # print(ontologies)
        multi_label_count = 0
        total_sentence_count = 0
        total_disc_count = 0
        total_sentence_disc_count = 0
        total_word_count = 0
        total_word_concept_count = 0
        total_overlap_count = 0
        total_other_disc_count = 0


        if ontology in ontologies:
            for root, directories, filenames in os.walk(craft_path+articles):

                for filename in sorted(filenames):
                    # if file_num < max_files:

                    if filename.endswith('.txt'):# and '15040800' in filename: #loop over txt files to tokenize sentence and word levels #disjoint concept file: and '17696610' in filename

                        print(filename)
                        with open(root+filename, 'r+', encoding='utf-8') as pmc_doc_file:



                            pmc_doc_text = pmc_doc_file.read()

                            ##pmc_doc_sentence_list_indicies = [(start, end)]
                            pmc_doc_sentence_list, pmc_doc_sentence_list_indicies = sentence_tokenize(pmc_doc_text, filename)
                            total_sentence_count += len(pmc_doc_sentence_list)
                            pmc_doc_word_list = []
                            pmc_doc_word_list_full = []


                            ###dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number - None for now)

                            all_lcs_dict = {} # lc -> [regex, ignorance_type]
                            all_ignorance_types = []
                            if excluded_ignorance_types:
                                print('PROGRESS: excluded files include ', excluded_ignorance_types)
                                ##get rid of the it to exclude using regex
                                # create all_lcs_dict: lc -> [regex, ignorance_type]
                                with open('%s' % all_lcs_path, 'r') as all_lcs_file:
                                    next(all_lcs_file)
                                    for line in all_lcs_file:
                                        lc, regex, it = line.strip('\n').split('\t')
                                        all_ignorance_types += [it]
                                        if all_lcs_dict.get(lc):
                                            all_lcs_dict[lc] += [it]
                                        else:
                                            all_lcs_dict[lc] = [it]
                                print('PROGRESS: all ignorance types include:', set(all_ignorance_types))
                                #ensure that we have the correct ignorace type list in excluded
                                for excluded_it in excluded_ignorance_types:
                                    if excluded_it not in set(all_ignorance_types):
                                        print(set(all_ignorance_types))
                                        raise Exception('ERROR ENTERING IN THE IGNORANCE TYPES TO EXCLUDE! (SEE ABOVE FOR THE LIST OF CORRECT TYPES)')

                            ##dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)
                            mention_ID_dict = gather_concept_annotations(ontology, craft_path+concept_annotation, filename, corpus, excluded_ignorance_types, all_lcs_dict, extensions)


                            final_concept_count = len(mention_ID_dict.keys())

                            running_concept_count = 0 #full document count

                            ##PER SENTENCE WORD TOKENIZE WITH SPANS (START AND END)
                            for s in range(len(pmc_doc_sentence_list)):
                                # print(s)
                                sentence = pmc_doc_sentence_list[s]
                                sentence_indicies = pmc_doc_sentence_list_indicies[s]
                                pmcid = filename.replace('.txt', '')
                                # print(sentence_indicies)


                                ##(pmcid, sentence_num) -> [sentence, sentence_indices, [ontology_concepts]]
                                if pmcid_sentence_dict.get((pmcid, s)):
                                   pass
                                else:
                                    pmcid_sentence_dict[(pmcid, s)] = [sentence, sentence_indicies, []] ##(pmcid, sentence_num) -> [sentence, [ontology_concepts]]


                                # # all_sent_word_info:  [[pmcid, sentence_number, sentence_start, sentence_end, word, pos_tag, word_start, word_end, BIO list, mention ID list, concept_ID, class label list]]
                                ##function: word_tokenize_sentences(pmcid, sentence, sentence_indicies, sentence_number, pmc_doc_text, mention_ID_dict, final_concept_count, running_concept_count, multi_label_count, current_BIO_hierarchy)
                                all_sent_word_info, running_concept_count, mention_ID_dict, multi_label_count, all_sent_word_info_full, disc_count, other_disc_count, no_concepts_count, overlap_count = word_tokenize_sentences(filename.replace('.txt', '') ,sentence, sentence_indicies, s, pmc_doc_text, mention_ID_dict, final_concept_count, running_concept_count, multi_label_count) #
                                ##total number of words with discontinuity
                                total_disc_count += disc_count
                                total_other_disc_count += other_disc_count
                                total_word_count += len(all_sent_word_info)
                                total_word_concept_count += (len(all_sent_word_info) - no_concepts_count)
                                total_overlap_count += overlap_count
                                ##number of sentences with discontinuity
                                if disc_count > 0:
                                    total_sentence_disc_count += 1



                                ##collect all the concept IDs for the TFIDF using the full information
                                for a in all_sent_word_info_full:  #looping over the words in the sentence
                                    if a[-2][0] != None:
                                        #print(a)
                                        pmcid_sentence_dict[(pmcid, s)][2] += a[-2] #list of concept ids added for this specific word
                                # print(pmcid_sentence_dict[(pmcid, s)])


                                pmc_doc_word_list += all_sent_word_info #lists of all the words in the document (length of the words!
                                pmc_doc_word_list_full += all_sent_word_info_full







                            ##check that all concepts matched to a sentence:
                            if final_concept_count != running_concept_count:

                                print(final_concept_count, running_concept_count)
                                text_start_index = pmc_doc_sentence_list_indicies[0][0]
                                text_end_index = pmc_doc_sentence_list_indicies[-1][-1]
                                print('text start index', text_start_index)
                                print('text end index', text_end_index)


                                issue = False
                                ##dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)
                                mention_ID_dict_addition = {}
                                mention_ID_dict_subtract = []
                                need_to_update_spans = []
                                need_to_update_mention_IDs = []
                                for m in mention_ID_dict:
                                    if mention_ID_dict[m][-1] is None:
                                        print(m)
                                        print(mention_ID_dict[m])
                                        (start_list1, end_list1, spanned_text1, mention_class_ID1, class_label1, sentence_number1) = mention_ID_dict[m]
                                        ##check if the concept spans 2 sentences
                                        for q in range(len(mention_ID_dict[m][0])):
                                            q_start = int(mention_ID_dict[m][0][q])
                                            q_end = int(mention_ID_dict[m][1][q])

                                            ##pmcid_sentence_dict: (pmcid, sentence_num) -> [sentence, sentence_indices, [ontology_concepts]]

                                            for s1 in range(len(pmc_doc_sentence_list)):
                                                sent_start = pmcid_sentence_dict[(pmcid, s1)][1][0]
                                                sent_end = pmcid_sentence_dict[(pmcid, s1)][1][1]

                                                if sent_start <= q_start and q_end <= sent_end:
                                                    mention_ID_dict_addition[m+string.ascii_lowercase[q]] = [[start_list1[q]], [end_list1[q]], pmc_doc_text[q_start:q_end], mention_class_ID1, class_label1, s1]
                                                    need_to_update_spans += [(q_start, q_end)]
                                                    need_to_update_mention_IDs += [m+string.ascii_lowercase[q]]

                                                    pmcid_sentence_dict[(pmcid, s1)][-1] += [m+string.ascii_lowercase[q]]
                                                    break

                                        mention_ID_dict_subtract += [m]
                                        running_concept_count += 1


                                ##get rid of the bad one!
                                for m1 in mention_ID_dict_subtract:
                                    mention_ID_dict.pop(m1)
                                ##add the split
                                print(mention_ID_dict_addition)
                                mention_ID_dict.update(mention_ID_dict_addition)
                                print(len(mention_ID_dict.keys()))
                                print(mention_ID_dict['CL_basic_2014_02_21_Instance_15795a'])
                                print(mention_ID_dict['CL_basic_2014_02_21_Instance_15795b'])

                                ##also update pmc_doc_word_list and pmc_doc_word_list_full
                                ### all_sent_word_info:  [[pmcid, sentence_number, sentence_start, sentence_end, word, pos_tag, word_start, word_end, BIO list, mention ID list, concept_ID, class label list]]
                                print(pmc_doc_word_list[:2])
                                # need_to_update_span_starts = [n[0] for n in need_to_update_spans]
                                for p in range(len(pmc_doc_word_list)):
                                    for t, n in enumerate(need_to_update_spans):
                                        if pmc_doc_word_list[p][6] == n[0] and pmc_doc_word_list[p][7] == n[1]:

                                            ##regular list (not full)
                                            pmc_doc_word_list[p][-4] = 'B'

                                            pmc_doc_word_list[p][-3] = [need_to_update_mention_IDs[t]]

                                            pmc_doc_word_list[p][-2] = [mention_ID_dict[need_to_update_mention_IDs[t]][3]]

                                            pmc_doc_word_list[p][-1] = [mention_ID_dict[need_to_update_mention_IDs[t]][4]]



                                            ##full list
                                            pmc_doc_word_list_full[p][-4] = ['B']
                                            pmc_doc_word_list_full[p][-3] = [need_to_update_mention_IDs[t]]
                                            pmc_doc_word_list_full[p][-2] = [
                                                mention_ID_dict[need_to_update_mention_IDs[t]][3]]
                                            pmc_doc_word_list_full[p][-1] = [
                                                mention_ID_dict[need_to_update_mention_IDs[t]][4]]




                                ##check to see if we are good!
                                if final_concept_count != running_concept_count:
                                    raise Exception('ERROR WITH CONCEPT MATCHING TO SENTENCES WITHIN THE TEXT!!')




                            # if len(pmc_doc_sentence_list) != len(pmc_doc_word_list):
                            #     raise Exception('ERROR WITH WORD PROCESSING AND MAKING SURE IT IS A LIST OF LISTS OF WORD TOKENIZATIONS!')


                            ##OUTPUT THE BIO FORMAT RESULTS pandas dataframe! all_sent_word_info_full and mention_ID_dict

                            mention_ID_dict_pkl = open(output_path+'%s/%s_mention_id_dict.pkl' %(ontology,filename.replace('.txt','')), 'wb')
                            pickle.dump(mention_ID_dict, mention_ID_dict_pkl)
                            mention_ID_dict_pkl.close()

                            mention_ID_dict_csv = csv.writer(open(output_path+'%s/%s_mention_id_dict.csv' %(ontology,filename.replace('.txt','')), "w"))
                            for key, val in mention_ID_dict.items():
                                mention_ID_dict_csv.writerow([key, val])


                            # with open(output_path+ontology+'%s.bio' %(filename), 'w+') as output_file:
                            columns = ['PMCID', 'SENTENCE_NUM', 'SENTENCE_START', 'SENTENCE_END', 'WORD', 'POS_TAG', 'WORD_START', 'WORD_END', 'BIO_TAG', 'PMC_MENTION_ID', 'ONTOLOGY_CONCEPT_ID', 'ONTOLOGY_LABEL']



                            ##DATAFRAME full
                            output_df = pd.DataFrame(pmc_doc_word_list_full, columns = columns)
                            # print(output_df)
                            output_df.to_pickle(output_path+'%s/%s_full.pkl' %(ontology,filename.replace('.txt','')))
                            output_df.to_csv(output_path+'%s/%s_full.tsv' %(ontology,filename.replace('.txt','')), '\t')

                            ##DATAFRAME hierarchy
                            output_df = pd.DataFrame(pmc_doc_word_list, columns=columns)
                            # print(output_df)
                            output_df.to_pickle(output_path + '%s/%s.pkl' % (ontology, filename.replace('.txt', '')))
                            output_df.to_csv(output_path + '%s/%s.tsv' % (ontology, filename.replace('.txt', '')), '\t')





                # print('PROGRESS: FINISHED ONTOLOGY ' + ontology)
                file_num += 1

            print('MULTI-LABEL COUNT:', multi_label_count)
            print('SENTENCE COUNT:', total_sentence_count)
            print('SENTENCE DISCONTINUOUS COUNT', total_sentence_disc_count)
            print('WORD DISCONTINUOUS COUNT:', total_disc_count)

            disc_output.write('%s\t%s\t%s\t%s\t%.2f\t%s\t%s\t%s\t%s\n' %(ontology, total_sentence_count, total_sentence_disc_count, total_disc_count, float(total_sentence_disc_count)/float(total_sentence_count), total_other_disc_count, total_overlap_count, total_word_count, total_word_concept_count))


    ##output the PMCID dictionary for TFIDF: pmcid_sentence_dict = {} #(pmcid, sentence_num) -> [sentence, sentence_indices, [ontology_concepts]]
    all_pmcids = [p for (p,s) in pmcid_sentence_dict.keys()]

    for pmcid in all_pmcids:

        with open('%s%s_%s.txt' %(output_path.replace('Tokenized_Files', pmcid_sentence_path), pmcid, 'sentence_info'), 'w+') as pmcid_output_file:
            pmcid_output_file.write('%s\t%s\t%s\t%s\t%s\n' %('PMCID', 'SENTENCE_NUMBER', 'SENTENCE', 'SENTENCE_INDICES', 'ONTOLOGY_CONCEPT_IDS_LIST'))
            i = 0
            while pmcid_sentence_dict.get((pmcid, i)):
                pmcid_output_file.write('%s\t%s\t%s\t%s\t%s\n' % (
                pmcid, i, [pmcid_sentence_dict[(pmcid, i)][0]], pmcid_sentence_dict[(pmcid, i)][1],
                pmcid_sentence_dict[(pmcid, i)][2]))
                i += 1



if __name__=='__main__':

    # current_BIO_hierarchy = ['B', 'B-', 'I', 'I-', 'O']



    parser = argparse.ArgumentParser()
    parser.add_argument('-craft_path', type=str, help='a string file path to the corpus annotations')
    parser.add_argument('-articles', type=str, help='a string file name to the plain text articles')
    parser.add_argument('-concept_annotation', type=str, help='a string file name to the concept annotations')
    parser.add_argument('-ontologies', type=str, help='a list of ontologies to use delimited with ,')
    parser.add_argument('-output_path', type=str, help='a file path to the output for the tokenized files')
    parser.add_argument('-pmcid_sentence_path', type=str, help='the file name for the pmcid sentences to output')
    parser.add_argument('-corpus', type=str, help='a string of either "craft" or "ignorance" to specify which corpus to use')

    ##optional: default is none
    parser.add_argument('--excluded_ignorance_types', type=str, help='a list of ignorance types to exclude delimited with , in all caps, default is None', default=None)
    parser.add_argument('--all_lcs_path', type=str, help='the file path to the lexical cues for the ignorance ontology',
                        default=None)
    parser.add_argument('--extensions', type=str,
                        help='whether or not we have extensions (leave blank if no extensions, default is None',
                        default=None)
    args = parser.parse_args()

    ontologies = args.ontologies.split(',')

    ##exclude ignorance types if you want
    if args.excluded_ignorance_types:
        excluded_ignorance_types = args.excluded_ignorance_types.split(',') #list of the ignorance types to exclude
        preprocess_docs(args.craft_path, args.articles, args.concept_annotation, ontologies, args.output_path, args.pmcid_sentence_path, args.corpus, excluded_ignorance_types, args.all_lcs_path, args.extensions)

    ##otherwise run the preprocess
    else:
        preprocess_docs(args.craft_path, args.articles, args.concept_annotation, ontologies, args.output_path, args.pmcid_sentence_path, args.corpus, args.excluded_ignorance_types, args.all_lcs_path, args.extensions)