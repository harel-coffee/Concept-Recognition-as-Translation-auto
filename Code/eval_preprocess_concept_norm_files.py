import os
import pandas as pd
import datetime
import argparse


def preprocess_data(tokenized_file_path, ontology, ontology_dict, concept_norm_files_path, evaluation_files):

    pmc_mention_id_index = 0  ##provide an id for each mention to use as its unique identifier - with pmcid

    for root, directories, filenames in os.walk(tokenized_file_path + ontology + '/'):
        for filename in sorted(filenames):
            # print(root+filename)

            ##grab all tokenized files per pmcid file - want to combine per ontology
            if filename.endswith('.txt') and ontology in filename and 'local' in filename and 'pred' not in filename and (filename.split('_')[-1].split('.')[0] in evaluation_files or evaluation_files[0].lower() == 'all'):
                print(filename)
                pmc_mention_id_index += 1 ##need to ensure we go up one always

                ##columns = ['PMCID', 'SENTENCE_NUM', 'SENTENCE_START', 'SENTENCE_END', 'WORD', 'POS_TAG', 'WORD_START', 'WORD_END', 'BIO_TAG', 'PMC_MENTION_ID', 'ONTOLOGY_CONCEPT_ID', 'ONTOLOGY_LABEL']

                ##read in the csv file for the tokenized dataframe
                pmc_tokenized_file_df = pd.read_csv(root+filename, sep='\t', header=0, quotechar='"', quoting=3)#, encoding='utf-8', engine='python')

                ##loop over each token grabbing all the mentions from the dataframe
                for index, row in pmc_tokenized_file_df.iterrows():

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

                    ##the dataframe reads the word 'null' as a NaN which is bad and so we change it to null
                    if word == 'nan':
                        word = 'null'

                    ##check that the nulls are changed
                    if {pmc_mention_id, ontology_concept_id, ontology_label} != {'None'}:
                        print(pmc_mention_id, ontology_concept_id, ontology_label)
                        raise Exception('ERROR WITH SPAN DETECTION NONES AT THE END!')
                    else:
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
                                pmc_mention_id_index += 1

                            ##no concept at all
                            else:
                                pass

                        else: #'O-' continuously
                            raise Exception('ERROR WITH A WEIRD TAG OTHER THAN THE 4!')


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
    ##loop over each id in the ontology dict to output
    for pmc_mention_id in ontology_dict.keys():
        sentence_num = ontology_dict[pmc_mention_id][0]


        word_list = ontology_dict[pmc_mention_id][1] #list of words
        word_indices_list = ontology_dict[pmc_mention_id][2]
        span_model = ontology_dict[pmc_mention_id][3]


        ##check to make sure all the data is collected and matching
        if len(word_list) != len(word_indices_list):
            print(len(word_list), len(word_indices_list))
            print(word_list)
            print(word_indices_list)
            raise Exception('ERROR WITH COLLECTING ALL WORDS AND INDICES!')

        # a concept based on O- with no other sequence tags around it
        ##skip this because no concept! (need to do set because you can have multiple of these in a row)
        if list(set(word_list)) == ['...']:
            # discontinuity_error_count += 1
            if disc_error_dict.get(span_model):
                disc_error_dict[span_model] += 1
            else:
                disc_error_dict[span_model] = 1

        # words in the word list
        else:
            updated_word = ''
            updated_word_indices_list = []  # [(start,end)]
            disc_count = 0  # count the number of discontinuities so we can get rid of their indices because you don't needs them

            for i, w in enumerate(word_list):
                ##I is first with no B
                if i == 0:  # always take the first word to start

                    ##word begins with a discontinuity
                    if w == '...':  # don't take indices for the discontinuity
                        disc_count += 1

                    ##get rid of the starts with '...' in general
                    elif w.startswith('...'):
                        updated_word += '%s' %(w.replace('...',''))
                        updated_word_indices_list += [word_indices_list[i]]


                    ##keep stuff that doesn't start with '...'
                    else:
                        updated_word += '%s' % w
                        updated_word_indices_list += [word_indices_list[i]]

                ##discontinuity in the middle of the word with no other discontinuity
                elif w == '...':  # and not disc_sign: #got rid of this cuz it shouldnt matter
                    updated_word += ' %s' % w  ##add the discontinuity piece that there is one
                    # updated_word_indices_list += [word_indices_list[i]] #10.16.20: do not add the indices of the discontinuity
                    disc_count += 1

                ##no discontinuity piece here
                elif w != '...':
                    updated_word += ' %s' % w
                    updated_word_indices_list += [word_indices_list[i]]

                else:
                    pass

            ##check that the updated word and indices match
            if len(updated_word.split(' ')) != len(updated_word_indices_list) + disc_count:
                print(updated_word)
                print(updated_word_indices_list)
                raise Exception('ERROR WITH UPDATING THE WORD TO GET THE FULL CONCEPT WITH INDICES!')
            else:
                ##create final updated word with correct spacing
                final_updated_word = ''
                for i, u in enumerate(updated_word.split(' ')):
                    if i == 0:
                        final_updated_word += '%s' %(u)
                    elif u == '...' and final_updated_word.endswith('...'):
                        pass
                    else:
                        final_updated_word += ' %s' %(u)
                ##get rid of trailing spaces in the beginning of the string
                updated_word = final_updated_word.lstrip()




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

            ##character word - word with spaces in between each character
            char_word = ''
            for c in updated_word:
                char_word += '%s ' %c

            char_word = char_word[:len(char_word)-1] ##cut off the last space
            combo_src_file_char.write('%s\n' %char_word)

    ##output the discontinuity errors so we can review
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


    ontologies = args.ontologies.split(',')
    evaluation_files = args.evaluation_files.split(',')



    for ontology in ontologies:
        print('PROGRESS:', ontology)
        ##the big dictionary with all the information for each ontology
        ontology_dict = {}

        ##create the discontinuity error output files
        disc_error_output_file = open('%s%s/%s_DISC_ERROR_SUMMARY.txt' %(args.concept_norm_files_path, ontology, ontology), 'w+')
        disc_error_output_file.write('%s\t%s\n' %('MODEL', 'NUM DISCONTINUITY ERRORS'))

        ##grab all the information for the ontology dict
        #ONTOLOGY_DICT - pmc_mention_id -> [sentence_num, word, [(word_indices)], span_model]
        ontology_dict = preprocess_data(args.results_span_detection_path, ontology, ontology_dict, args.concept_norm_files_path, evaluation_files)


        ##the output filenames
        filename_combo_list = ['combo_src_file', 'combo_link_file']

        ##output all the files
        output_all_files(args.concept_norm_files_path, ontology, ontology_dict, filename_combo_list, disc_error_output_file)

    print('PROGRESS: FINISHED CONCEPT NORMALIZATION PROCESSING FOR ALL FILES!')