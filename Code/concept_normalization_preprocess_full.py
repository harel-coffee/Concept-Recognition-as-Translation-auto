import os
import argparse
import ast
import pickle
import random
import sys
sys.setrecursionlimit(10000) #set based on how many recurssions it goes through.



def gather_spanned_text(tokenized_file_path, concept_norm_files_path , ontology, full_files_path, filename_combo_list, excluded_files):

    ##set up output files:
    all_output_files = [] ##list of all the output files
    for file in filename_combo_list:

        current_output_file = open('%s/%s/%s%s_%s.txt' %(concept_norm_files_path, ontology, full_files_path, ontology, file),'w+')
        current_output_file_val = open('%s/%s/%s%s_%s_val.txt' %(concept_norm_files_path, ontology, full_files_path, ontology, file),'w+')

        all_output_files += [current_output_file, current_output_file_val]

        if 'link' not in file:
            current_output_file_char = open('%s/%s/%s%s_%s_char.txt' %(concept_norm_files_path, ontology, full_files_path, ontology, file), 'w+')
            current_output_file_char_val = open('%s/%s/%s%s_%s_char_val.txt' % (concept_norm_files_path, ontology, full_files_path, ontology, file), 'w+')

            all_output_files += [current_output_file_char, current_output_file_char_val]


    total_concepts = 0

    print('NUMBER OF OUTPUT FILES:', len(all_output_files))
    ## mention_ID_dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)
    ##load the mention_ID_dict
    for root, directories, filenames in os.walk(tokenized_file_path + ontology + '/'):
        for filename in sorted(filenames):

            ##per pmcid file - want to combine per ontology
            if filename.endswith('.pkl') and 'mention_id_dict' in filename and filename.replace('_mention_id_dict.pkl','') not in excluded_files:
                mention_ID_dict_pkl = open(root+filename, 'rb')
                mention_ID_dict = pickle.load(mention_ID_dict_pkl)
                # print('NUMBER OF CONCEPTS PER PMCID FILE:', len(mention_ID_dict.keys()))
                total_concepts += len(mention_ID_dict.keys())
                for index, mention_ID in enumerate(mention_ID_dict.keys()):
                    (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number) = mention_ID_dict[mention_ID]


                    ##all the output files
                    ##SPANNED_TEXT
                    char_spanned_text = ''
                    for c in spanned_text:
                        char_spanned_text += '%s ' % c
                    char_spanned_text = char_spanned_text[
                                        :len(char_spanned_text) - 1]  # get rid of the extra whitespace at the end
                    if index % 10 != 0:
                        all_output_files[0].write('%s\n' %spanned_text)
                        all_output_files[2].write('%s\n' %char_spanned_text)
                    else: #evaluation files
                        all_output_files[1].write('%s\n' %spanned_text)
                        all_output_files[3].write('%s\n' % char_spanned_text)

                    ##MENTION_CLASS_ID
                    char_mention_class_ID = ''
                    for c in mention_class_ID:
                        char_mention_class_ID += '%s ' % c
                    char_mention_class_ID = char_mention_class_ID[:len(char_mention_class_ID) - 1]

                    if index % 10 != 0:
                        all_output_files[4].write('%s\n' %mention_class_ID)
                        all_output_files[6].write('%s\n' %char_mention_class_ID)
                    else: #evaluation files
                        all_output_files[5].write('%s\n' % mention_class_ID)
                        all_output_files[7].write('%s\n' % char_mention_class_ID)

                    ##CLASS LABEL
                    char_class_label = ''
                    for c in class_label:
                        char_class_label += '%s ' % c
                    char_class_label = char_class_label[:len(char_class_label) - 1]
                    if index % 10 != 0:
                        all_output_files[8].write('%s\n' %class_label)
                        all_output_files[10].write('%s\n' %char_class_label)
                    else: #evaluation
                        all_output_files[9].write('%s\n' % class_label)
                        all_output_files[11].write('%s\n' % char_class_label)

                    ##MENTION_ID
                    if index % 10 != 0:
                        all_output_files[12].write('%s\n' %mention_ID)
                    else: #evaluation
                        all_output_files[13].write('%s\n' % mention_ID)

                    ##SENTENCE NUMBER
                    if index % 10 != 0:
                        all_output_files[14].write('%s\n' %sentence_number)
                    else: #evaluation
                        all_output_files[15].write('%s\n' % sentence_number)

    print('TOTAL CONCEPTS:', total_concepts)
    return all_output_files


def additional_obo_concepts(ontology, concept_norm_files_path, all_output_files):


    with open('%s/%s/%s_addition.txt' %(concept_norm_files_path, ontology, ontology), 'r') as obo_addition_file:
        next(obo_addition_file) #the first line is the ontology and the number of terms TODO: maybe take in the first line to make sure we have all the concepts added

        count = 0

        for line in obo_addition_file:
            [ont_id, name, definition, synonyms] = line.split('\t')
            all_names = ast.literal_eval(synonyms) + [name] #convert to a list
            all_names_unique = list(set(all_names))

            for name in all_names_unique:
                count += 1
                ##all the output files


                ##SPANNED_TEXT - name
                char_name = ''
                for c in name:
                    char_name += '%s ' % c
                char_name = char_name[:len(char_name) - 1]  # get rid of the extra whitespace at the end
                if count % 10 != 0:
                    all_output_files[0].write('%s\n' % name)
                    all_output_files[2].write('%s\n' % char_name)
                else:  # evaluation files
                    all_output_files[1].write('%s\n' % name)
                    all_output_files[3].write('%s\n' % char_name)

                ##MENTION_CLASS_ID - ont_id
                char_ont_id = ''
                for c in ont_id:
                    char_ont_id += '%s ' % c
                char_ont_id = char_ont_id[:len(char_ont_id) - 1]

                if count % 10 != 0:
                    all_output_files[4].write('%s\n' % ont_id)
                    all_output_files[6].write('%s\n' % char_ont_id)
                else:  # evaluation files
                    all_output_files[5].write('%s\n' % ont_id)
                    all_output_files[7].write('%s\n' % char_ont_id)

                ##CLASS LABEL - name
                char_name = ''
                for c in name:
                    char_name += '%s ' % c
                char_name = char_name[:len(char_name) - 1]
                if count % 10 != 0:
                    all_output_files[8].write('%s\n' % name)
                    all_output_files[10].write('%s\n' % char_name)
                else:  # evaluation
                    all_output_files[9].write('%s\n' % name)
                    all_output_files[11].write('%s\n' % char_name)

                ##MENTION_ID - obo_addition
                if count % 10 != 0:
                    all_output_files[12].write('%s_addition\n' % ontology)
                else:  # evaluation
                    all_output_files[13].write('%s_addition\n' % ontology)

                ##SENTENCE NUMBER - obo_addition
                if count % 10 != 0:
                    all_output_files[14].write('%s_addition\n' % ontology)
                else:  # evaluation
                    all_output_files[15].write('%s_addition\n' % ontology)




def ontology_dictionary(ontology, concept_norm_path, full_files_path, character_info_file):
    #create a dictionary from concept ID to [ontology string concepts]
    ontology_dict = {}
    ontology_dict_val = {}


    ##lengths of all concepts and ids
    concept_char_lengths = []
    concept_id_char_lengths = []


    src_char_file='combo_src_file_char.txt'
    tgt_char_file = 'combo_tgt_concept_ids_char.txt'
    src_char_val_file = 'combo_src_file_char_val.txt'
    tgt_char_val_file = 'combo_tgt_concept_ids_char_val.txt'

    with open('%s/%s/%s%s_%s' %(concept_norm_path, ontology, full_files_path, ontology, src_char_file), 'r+') as src_file, open('%s/%s/%s%s_%s' %(concept_norm_path, ontology, full_files_path, ontology, tgt_char_file), 'r+') as tgt_file,  open('%s/%s/%s%s_%s' %(concept_norm_path, ontology, full_files_path, ontology, src_char_val_file), 'r+') as src_file_val, open('%s/%s/%s%s_%s' %(concept_norm_path, ontology, full_files_path, ontology, tgt_char_val_file), 'r+') as tgt_file_val:

        for concept, concept_id in zip(src_file, tgt_file):
            updated_concept = concept.strip('\n').lower()
            updated_concept_id = concept_id.strip('\n')

            concept_char_lengths += [len(updated_concept) - updated_concept.count(' ')]
            concept_id_char_lengths += [len(updated_concept_id) - updated_concept_id.count(' ')]


            if ontology_dict.get(updated_concept_id):
                ontology_dict[updated_concept_id].add(updated_concept)
            else:
                ontology_dict[updated_concept_id] = {updated_concept}


        for concept_val, concept_id_val in zip(src_file_val, tgt_file_val):
            updated_concept_val = concept_val.strip('\n').lower()
            updated_concept_id_val = concept_id_val.strip('\n')

            concept_char_lengths += [len(updated_concept_val) - updated_concept_val.count(' ')]
            concept_id_char_lengths += [len(updated_concept_id_val) - updated_concept_id_val.count(' ')]


            if ontology_dict_val.get(updated_concept_id_val):
                ontology_dict_val[updated_concept_id_val].add(updated_concept_val)
            else:
                ontology_dict_val[updated_concept_id_val] = {updated_concept_val}

    ##'%s\t%s\t%s\t%s\t%s\n' %('ONTOLOGY', 'MIN CONCEPT CHAR LENGTH', 'MAX CONCEPT CHAR LENGTH', 'MIN CONCEPT ID CHAR LENGTH', 'MAX CONCEPT ID CHAR LENGTH'))
    if concept_id_char_lengths and concept_char_lengths:
        character_info_file.write('%s\t%s\t%s\t%s\t%s\n' %(ontology, min(concept_char_lengths), max(concept_char_lengths), min(concept_id_char_lengths), max(concept_id_char_lengths)))
    else:
        print(concept_char_lengths)
        print(concept_id_char_lengths)
        character_info_file.write('%s\t%s\t%s\t%s\t%s\n' % (
        ontology, 0, 0, 0, 0))
        # raise Exception('hold!')

    return ontology_dict, ontology_dict_val



def randN(N):
    min = pow(10, N-1)
    max = pow(10, N) - 1
    return random.randint(min, max)

def random_id_generate(concept_id, all_random_ids):
    random_id = ''
    for d in concept_id.split(' '):
        if d.isdigit():
            r = random.randint(0, 9)
            random_id += ' %s' %r
        else:
            random_id += ' %s' %d

    if random_id[1:] in all_random_ids:
        return random_id_generate(concept_id, all_random_ids)
    else:
        return random_id[1:]

def shuffled_id_generate(concept_id, all_shuffled_ids, ontology_dict, all_concept_ids_nums_only):
    #we only shuffle if it has digits in it
    if any(map(str.isdigit, concept_id)):
        unused_concept_ids = list(set(all_concept_ids_nums_only) - set(all_shuffled_ids))
        shuffled_id = random.choice(unused_concept_ids)
        all_shuffled_ids += [shuffled_id]
        return shuffled_id

    #no digits so return itself
    else:
        shuffled_id = concept_id
        return shuffled_id


def no_duplicates_lower(ontology, ontology_dict, ontology_dict_val, concept_norm_files_path, main_files, additional_file_paths):
    ##set up output files:
    all_output_files = []  ##list of all the output files

    random_ids_dict = {} #dictionary from concept_id -> random_id
    all_random_ids = [] #list of random_ids

    shuffled_ids_dict = {} #dictionary from concept_id -> shuffled_id
    all_shuffled_ids = [] #list of shuffled_ids

    alphabetical_dict = {} #dictionary from concept -> alphabetical_id


    for file in main_files:
        for a in additional_file_paths:
            current_output_file = open(
                '%s/%s/%s%s_%s.txt' % (concept_norm_files_path, ontology, a, ontology, file), 'w+')
            current_output_file_val = open(
                '%s/%s/%s%s_%s_val.txt' % (concept_norm_files_path, ontology, a, ontology, file), 'w+')

            all_output_files += [current_output_file, current_output_file_val]


    ##all_output_files: src, src_val, tgt, tgt_val
    print('all output files', len(all_output_files)) #src, src_val (x3) ; tgt, tgt_val (x3)


    ##alphabetical information
    all_concepts = []
    all_concept_ids = list(ontology_dict.keys())

    all_concept_ids_nums_only = []
    for c_id in all_concept_ids:
        ##take all concepts for alphabetical
        all_concepts += ontology_dict[c_id]

        ##only take the digits for when we randomly assign and shuffle the ids
        if any(map(str.isdigit, c_id)):
            all_concept_ids_nums_only += [c_id]

        else:
            pass

    # print('only ontology dict length of concepts:', len(all_concepts))

    ##src/tgt val files
    all_concept_ids_val = list(ontology_dict_val.keys())
    all_concept_ids_val_nums_only = []
    for c_id_val in all_concept_ids_val:
        ##take all concepts for alphabetical
        all_concepts += ontology_dict_val[c_id_val]

        if any(map(str.isdigit, c_id_val)):
            ##only take the digits for when we randomly assign and shuffle the ids
            all_concept_ids_val_nums_only += [c_id_val]

        else:
            pass

    ##sort the list alphabetically
    all_concept_ids_full = set(all_concept_ids).copy()
    all_concept_ids_full.update(set(all_concept_ids_val))



    all_concepts_sorted = sorted(list(set(all_concepts))) #sorted set of concepts so we have no duplicates
    alphabet_num_digits = len(str(len(all_concepts_sorted))) #total number of digits we need for this ontology
    print('num digits needed for alphabet:', ontology, alphabet_num_digits)

    for r, sorted_concept in enumerate(all_concepts_sorted):
        alphabetical_id = ontology + ':' + str(r).zfill(alphabet_num_digits)
        alphabetical_id = ' '.join(alphabetical_id)
        alphabetical_dict[sorted_concept] = alphabetical_id



    ##src/tgt files - ontology_dict: concept_id -> [concept_list]
    for concept_id, concept_list in ontology_dict.items():
        if len(set(concept_list)) != len(concept_list):
            raise Exception('ERROR WITH MAKING SURE WE HAVE NO DUPLICATES DURING ONTOLOGY DICT BUILD!')
        else:
            pass


        for i,j in zip(range(0, int(len(all_output_files)/2), 2), range(0, len(additional_file_paths))):
            if 'random_ids' in additional_file_paths[j]:
                random_id = random_id_generate(concept_id, all_random_ids)
                all_random_ids += [random_id]
                random_ids_dict[concept_id] = random_id

            elif 'shuffled_ids' in additional_file_paths[j]:
                shuffled_id = shuffled_id_generate(concept_id, all_shuffled_ids, ontology_dict, all_concept_ids_nums_only)
                all_shuffled_ids += [shuffled_id]
                shuffled_ids_dict[concept_id] = shuffled_id



            ##for each concept in the list we have the same concept_id
            for c in concept_list:
                all_output_files[i].write('%s\n' %c)


                if 'no_duplicates' in additional_file_paths[j]:
                    all_output_files[i+int(len(all_output_files)/2)].write('%s\n' %concept_id)

                elif 'random_ids' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % random_id)

                elif 'shuffled_ids' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % shuffled_id)

                elif 'alphabetical' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % alphabetical_dict[c])




    unique_concept_id_val = list(set(all_concept_ids_val_nums_only) - set(all_concept_ids_nums_only))


    for concept_id_val, concept_list_val in ontology_dict_val.items():
        if len(set(concept_list_val)) != len(concept_list_val):
            raise Exception('ERROR WITH MAKING SURE WE HAVE NO DUPLICATES DURING ONTOLOGY DICT BUILD!')
        else:
            pass

        for i,j in zip(range(1, int(len(all_output_files)/2), 2), range(0, len(additional_file_paths))):
            if 'random_ids' in additional_file_paths[j]:
                if random_ids_dict.get(concept_id_val):
                    random_id_val = random_ids_dict[concept_id_val]
                else:
                    random_id_val = random_id_generate(concept_id_val, all_random_ids)
                    all_random_ids += [random_id_val]
                    random_ids_dict[concept_id_val] = random_id_val

            elif 'shuffled_ids' in additional_file_paths[j]:
                if shuffled_ids_dict.get(concept_id_val):
                    shuffled_id_val = shuffled_ids_dict[concept_id_val]
                else:
                    shuffled_id_val = shuffled_id_generate(concept_id_val, all_shuffled_ids, ontology_dict_val, unique_concept_id_val)
                    all_shuffled_ids += [shuffled_id_val]
                    shuffled_ids_dict[concept_id_val] = shuffled_id_val



            ##for each concept in the list we have the same concept_id
            for c_val in concept_list_val:
                all_output_files[i].write('%s\n' %c_val)


                if 'no_duplicates' in additional_file_paths[j]:
                    all_output_files[i+int(len(all_output_files)/2)].write('%s\n' %concept_id_val)

                elif 'random_ids' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % random_id_val)

                elif 'shuffled_ids' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % shuffled_id_val)

                elif 'alphabetical' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % alphabetical_dict[c_val])




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-tokenized_file_path', type=str, help='a string file path to the tokenized files')
    parser.add_argument('-excluded_files', type=str, help='a string list of the files to exclude in this run - comma delimited with no spaces')
    # parser.add_argument('-save_models_path', type=str, help='a string file path to the saved models')
    parser.add_argument('-ontologies', type=str, help='a list of ontologies to use delimited with ,')
    parser.add_argument('-concept_norm_files_path', type=str, help='a file path to the output for the concept norm files for concept normalization')
    parser.add_argument('-full_files_path', type=str,help='the string folder with the full files in it')

    parser.add_argument('--extensions', type=str, help='whether or not we have extensions (leave blank if no extensions, default is None', default=None)

    args = parser.parse_args()

    ontologies = args.ontologies.split(',')
    excluded_files = args.excluded_files.split(',')

    if args.extensions:
        character_info_file = open('%s%s.txt' %(args.concept_norm_files_path, 'character_info_EXT'), 'w+')
    else:
        character_info_file = open('%s%s.txt' % (args.concept_norm_files_path, 'character_info'), 'w+')

    character_info_file.write('%s\t%s\t%s\t%s\t%s\n' %('ONTOLOGY', 'MIN CONCEPT CHAR LENGTH', 'MAX CONCEPT CHAR LENGTH', 'MIN CONCEPT ID CHAR LENGTH', 'MAX CONCEPT ID CHAR LENGTH'))

    for ontology in ontologies:
        print('PROGRESS:', ontology)

        #ONTOLOGY_DICT - pmc_mention_id -> [sentence_num, word, ontology_concept_id, ontology_label]
        filename_combo_list = ['combo_src_file', 'combo_tgt_concept_ids', 'combo_tgt_ont_labels','combo_link_mention_ids','combo_link_sent_nums']


        ### output_all_files(concept_norm_files_path, ontology, ontology_dict, od_indices, filename_combo_list)
        all_output_files = gather_spanned_text(args.tokenized_file_path, args.concept_norm_files_path , ontology, args.full_files_path, filename_combo_list, excluded_files)

        additional_obo_concepts(ontology, args.concept_norm_files_path, all_output_files)


        ##TODO: make all of them lowercase and uniform! also duplicates!
        ##now that we have all the regular files with the full stuff let's do more
        #ontology dictionary by ID
        ontology_dict, ontology_dict_val = ontology_dictionary(ontology, args.concept_norm_files_path, args.full_files_path, character_info_file)
        print('total ontology concept ids:', len(ontology_dict.keys()))

        main_files = ['combo_src_file_char', 'combo_tgt_concept_ids_char']
        additional_file_paths = ['no_duplicates/', 'random_ids/', 'shuffled_ids/', 'alphabetical/']


        no_duplicates_lower(ontology, ontology_dict, ontology_dict_val, args.concept_norm_files_path, main_files, additional_file_paths)

