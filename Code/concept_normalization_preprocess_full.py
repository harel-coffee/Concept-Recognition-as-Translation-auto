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

        current_output_file = open('%s%s/%s%s_%s.txt' %(concept_norm_files_path, ontology, full_files_path, ontology, file),'w+')
        current_output_file_val = open('%s%s/%s%s_%s_val.txt' %(concept_norm_files_path, ontology, full_files_path, ontology, file),'w+')

        all_output_files += [current_output_file, current_output_file_val]

        if 'link' not in file:
            current_output_file_char = open('%s%s/%s%s_%s_char.txt' %(concept_norm_files_path, ontology, full_files_path, ontology, file), 'w+')
            current_output_file_char_val = open('%s%s/%s%s_%s_char_val.txt' % (concept_norm_files_path, ontology, full_files_path, ontology, file), 'w+')

            all_output_files += [current_output_file_char, current_output_file_char_val]


    total_concepts = 0

    print('NUMBER OF OUTPUT FILES:', len(all_output_files))
    ## mention_ID_dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)
    ##load the mention_ID_dict for each pmcid
    for root, directories, filenames in os.walk(tokenized_file_path + ontology + '/'):
        for filename in sorted(filenames):

            ##per pmcid file - want to combine per ontology
            if filename.endswith('.pkl') and 'mention_id_dict' in filename and filename.replace('_mention_id_dict.pkl','') not in excluded_files:
                mention_ID_dict_pkl = open(root+filename, 'rb')
                mention_ID_dict = pickle.load(mention_ID_dict_pkl)
                ##collect the number of concepts
                total_concepts += len(mention_ID_dict.keys())

                ##grab all information for each mention
                for index, mention_ID in enumerate(mention_ID_dict.keys()):
                    (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number) = mention_ID_dict[mention_ID]


                    ##all the output files
                    ##SPANNED_TEXT
                    char_spanned_text = ''
                    for c in spanned_text:
                        char_spanned_text += '%s ' % c
                    char_spanned_text = char_spanned_text[:len(char_spanned_text) - 1]  # get rid of the extra whitespace at the end

                    ##training file
                    if index % 10 != 0:
                        all_output_files[0].write('%s\n' %spanned_text)
                        all_output_files[2].write('%s\n' %char_spanned_text)
                    ##validation file - 10%
                    else:
                        all_output_files[1].write('%s\n' %spanned_text)
                        all_output_files[3].write('%s\n' % char_spanned_text)

                    ##MENTION_CLASS_ID
                    char_mention_class_ID = ''
                    for c in mention_class_ID:
                        char_mention_class_ID += '%s ' % c
                    char_mention_class_ID = char_mention_class_ID[:len(char_mention_class_ID) - 1]

                    ##training file
                    if index % 10 != 0:
                        all_output_files[4].write('%s\n' %mention_class_ID)
                        all_output_files[6].write('%s\n' %char_mention_class_ID)
                    ##validation file - 10%
                    else:
                        all_output_files[5].write('%s\n' % mention_class_ID)
                        all_output_files[7].write('%s\n' % char_mention_class_ID)

                    ##CLASS LABEL
                    char_class_label = ''
                    for c in class_label:
                        char_class_label += '%s ' % c
                    char_class_label = char_class_label[:len(char_class_label) - 1]

                    ##training file
                    if index % 10 != 0:
                        all_output_files[8].write('%s\n' %class_label)
                        all_output_files[10].write('%s\n' %char_class_label)
                    ##validation file - 10%
                    else:
                        all_output_files[9].write('%s\n' % class_label)
                        all_output_files[11].write('%s\n' % char_class_label)

                    ##MENTION_ID
                    ##training file
                    if index % 10 != 0:
                        all_output_files[12].write('%s\n' %mention_ID)
                    ##validation file - 10%
                    else:
                        all_output_files[13].write('%s\n' % mention_ID)

                    ##SENTENCE NUMBER
                    ##training file
                    if index % 10 != 0:
                        all_output_files[14].write('%s\n' %sentence_number)
                    ##validation file - 10%
                    else:
                        all_output_files[15].write('%s\n' % sentence_number)

    print('TOTAL CONCEPTS:', total_concepts)
    return all_output_files


def additional_obo_concepts(ontology, concept_norm_files_path, all_output_files):
    ##grab all the additional obo concepts not seen in CRAFT bur from the .obo files
    with open('%s%s/%s_addition.txt' %(concept_norm_files_path, ontology, ontology), 'r') as obo_addition_file:
        next(obo_addition_file) #the first line is the ontology and the number of terms

        count = 0
        char_name_count = 0
        char_ont_id_count = 0
        char_label_count = 0
        char_mention_id_count = 0
        char_sent_num_count = 0

        ##loop over each line grabbing the concept information
        for line in obo_addition_file:
            [ont_id, name, definition, synonyms] = line.split('\t')
            all_names = ast.literal_eval(synonyms) + [name] #convert to a list
            all_names_unique = list(set(all_names))

            for name in all_names_unique:
                count += 1
                ##add info to all the output files - training and validation (10%)
                ##SPANNED_TEXT - name
                char_name = ''
                for c in name:
                    char_name += '%s ' % c
                char_name = char_name[:len(char_name) - 1]  # get rid of the extra whitespace at the end
                ##training file
                if count % 10 != 0:
                    all_output_files[0].write('%s\n' % name)
                    all_output_files[2].write('%s\n' % char_name)
                    char_name_count += 1
                ##validation file - 10%
                else:
                    all_output_files[1].write('%s\n' % name)
                    all_output_files[3].write('%s\n' % char_name)
                    char_name_count += 1

                ##MENTION_CLASS_ID - ont_id
                char_ont_id = ''
                for c in ont_id:
                    char_ont_id += '%s ' % c
                char_ont_id = char_ont_id[:len(char_ont_id) - 1]
                ##training file
                if count % 10 != 0:
                    all_output_files[4].write('%s\n' % ont_id)
                    all_output_files[6].write('%s\n' % char_ont_id)
                    char_ont_id_count += 1
                ##validation file - 10%
                else:
                    all_output_files[5].write('%s\n' % ont_id)
                    all_output_files[7].write('%s\n' % char_ont_id)
                    char_ont_id_count += 1

                ##CLASS LABEL - name
                char_name = ''
                for c in name:
                    char_name += '%s ' % c
                char_name = char_name[:len(char_name) - 1]
                ##training file
                if count % 10 != 0:
                    all_output_files[8].write('%s\n' % name)
                    all_output_files[10].write('%s\n' % char_name)
                    char_label_count += 1
                ##validation file - 10%
                else:
                    all_output_files[9].write('%s\n' % name)
                    all_output_files[11].write('%s\n' % char_name)
                    char_label_count += 1

                ##MENTION_ID - obo_addition
                ##training file
                if count % 10 != 0:
                    all_output_files[12].write('%s_addition\n' % ontology)
                    char_mention_id_count += 1
                ##validation file - 10%
                else:
                    all_output_files[13].write('%s_addition\n' % ontology)
                    char_mention_id_count += 1

                ##SENTENCE NUMBER - obo_addition
                ##training file
                if count % 10 != 0:
                    all_output_files[14].write('%s_addition\n' % ontology)
                    char_sent_num_count += 1
                ##validation file - 10%
                else:
                    all_output_files[15].write('%s_addition\n' % ontology)
                    char_sent_num_count += 1

            ##check that everything is outputted correctly and together
            if len({char_name_count, char_ont_id_count, char_label_count, char_mention_id_count, char_sent_num_count}) != 1:
                print('line error:')
                print(line)
                print('count errors:')
                print(char_name_count)
                print(char_ont_id_count)
                print(char_label_count)
                print(char_mention_id_count)
                print(char_sent_num_count)
                raise Exception('ERROR: issue with outputting everything together')

        print('PROGRESS:finished obo addition!')

def ontology_dictionary(ontology, concept_norm_path, full_files_path, character_info_file):
    #create a dictionary from concept ID to [ontology string concepts]
    ontology_dict = {}
    ontology_dict_val = {}


    ##lengths of all concepts and ids
    concept_char_lengths = []
    concept_id_char_lengths = []

    ##filenames
    src_char_file='combo_src_file_char.txt'
    tgt_char_file = 'combo_tgt_concept_ids_char.txt'
    src_char_val_file = 'combo_src_file_char_val.txt'
    tgt_char_val_file = 'combo_tgt_concept_ids_char_val.txt'

    ##create the ontology dictionary from all the concept norm training and validation files to conduct experiments by randomizing, shuffling, and alphebatizing ids
    with open('%s%s/%s%s_%s' %(concept_norm_path, ontology, full_files_path, ontology, src_char_file), 'r+') as src_file, open('%s%s/%s%s_%s' %(concept_norm_path, ontology, full_files_path, ontology, tgt_char_file), 'r+') as tgt_file,  open('%s%s/%s%s_%s' %(concept_norm_path, ontology, full_files_path, ontology, src_char_val_file), 'r+') as src_file_val, open('%s%s/%s%s_%s' %(concept_norm_path, ontology, full_files_path, ontology, tgt_char_val_file), 'r+') as tgt_file_val:

        ##grab all concepts in src and target file - training - there are multiple concept names per concept id
        for concept, concept_id in zip(src_file, tgt_file):
            updated_concept = concept.strip('\n').lower()
            updated_concept_id = concept_id.strip('\n')

            concept_char_lengths += [len(updated_concept) - updated_concept.count(' ')]
            concept_id_char_lengths += [len(updated_concept_id) - updated_concept_id.count(' ')]


            if ontology_dict.get(updated_concept_id):
                ontology_dict[updated_concept_id].add(updated_concept)
            else:
                ontology_dict[updated_concept_id] = {updated_concept}

        ##grab all concepts in src_val, target_val files - validation - there are multiple concept names per concept id
        for concept_val, concept_id_val in zip(src_file_val, tgt_file_val):
            updated_concept_val = concept_val.strip('\n').lower()
            updated_concept_id_val = concept_id_val.strip('\n')

            concept_char_lengths += [len(updated_concept_val) - updated_concept_val.count(' ')]
            concept_id_char_lengths += [len(updated_concept_id_val) - updated_concept_id_val.count(' ')]


            if ontology_dict_val.get(updated_concept_id_val):
                ontology_dict_val[updated_concept_id_val].add(updated_concept_val)
            else:
                ontology_dict_val[updated_concept_id_val] = {updated_concept_val}


    ##output all information for character info
    ##'%s\t%s\t%s\t%s\t%s\n' %('ONTOLOGY', 'MIN CONCEPT CHAR LENGTH', 'MAX CONCEPT CHAR LENGTH', 'MIN CONCEPT ID CHAR LENGTH', 'MAX CONCEPT ID CHAR LENGTH'))
    if concept_id_char_lengths and concept_char_lengths:
        character_info_file.write('%s\t%s\t%s\t%s\t%s\n' %(ontology, min(concept_char_lengths), max(concept_char_lengths), min(concept_id_char_lengths), max(concept_id_char_lengths)))
    else:
        print(concept_char_lengths)
        print(concept_id_char_lengths)
        character_info_file.write('%s\t%s\t%s\t%s\t%s\n' % (ontology, 0, 0, 0, 0))

    return ontology_dict, ontology_dict_val


##determine a random number
def randN(N):
    min = pow(10, N-1)
    max = pow(10, N) - 1
    return random.randint(min, max)

##generate a random concept ID of a specific length
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

##generate a shuffled id if it has digits in it only
def shuffled_id_generate(concept_id, all_shuffled_ids, ontology_dict, all_concept_ids_nums_only):
    #we only shuffle if it has digits in it
    if any(map(str.isdigit, concept_id)):
        unused_concept_ids = list(set(all_concept_ids_nums_only).difference(set(all_shuffled_ids)))
        shuffled_id = random.choice(unused_concept_ids)
        all_shuffled_ids += [shuffled_id]
        return shuffled_id

    #no digits so return itself
    else:
        shuffled_id = concept_id
        return shuffled_id

##get rid of all duplicate ontology concepts
def no_duplicates_lower(ontology, ontology_dict, ontology_dict_val, concept_norm_files_path, main_files, additional_file_paths):
    ##set up output files:
    all_output_files = []  ##list of all the output files

    random_ids_dict = {} #dictionary from concept_id -> random_id
    all_random_ids = [] #list of random_ids

    shuffled_ids_dict = {} #dictionary from concept_id -> shuffled_id
    all_shuffled_ids = [] #list of shuffled_ids


    for file in main_files:
        for a in additional_file_paths:
            current_output_file = open(
                '%s%s/%s%s_%s.txt' % (concept_norm_files_path, ontology, a, ontology, file), 'w+')
            current_output_file_val = open(
                '%s%s/%s%s_%s_val.txt' % (concept_norm_files_path, ontology, a, ontology, file), 'w+')

            all_output_files += [current_output_file, current_output_file_val]


    ##all_output_files: src, src_val, tgt, tgt_val
    print('all output files', len(all_output_files)) #src, src_val (x3) ; tgt, tgt_val (x3)


    ##alphabetical information
    all_concepts = []
    all_concept_ids = list(ontology_dict.keys())

    ##only number concept ids for random and shuffled ids
    all_concept_ids_nums_only = []
    for c_id in all_concept_ids:
        ##take all concepts for alphabetical
        all_concepts += ontology_dict[c_id]

        ##only take the digits for when we randomly assign and shuffle the ids
        if any(map(str.isdigit, c_id)):
            all_concept_ids_nums_only += [c_id]

        else:
            pass


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



    ##create all output files for experiments
    ##src/tgt files - ontology_dict: concept_id -> [concept_list]
    for concept_id, concept_list in ontology_dict.items():
        ##ensure no duplicates for the experiments
        if len(set(concept_list)) != len(concept_list):
            raise Exception('ERROR WITH MAKING SURE WE HAVE NO DUPLICATES DURING ONTOLOGY DICT BUILD!')
        else:
            pass

        ##create dictionary for random and shuffled ids and output the experiment files
        for i,j in zip(range(0, int(len(all_output_files)/2), 2), range(0, len(additional_file_paths))):
            if 'random_ids' in additional_file_paths[j]:
                random_id = random_id_generate(concept_id, all_random_ids)
                all_random_ids += [random_id]
                random_ids_dict[concept_id] = random_id

            elif 'shuffled_ids' in additional_file_paths[j]:
                shuffled_id = shuffled_id_generate(concept_id, all_shuffled_ids, ontology_dict, all_concept_ids_nums_only)
                all_shuffled_ids += [shuffled_id]
                shuffled_ids_dict[concept_id] = shuffled_id



            ##for each concept in the list we have the same concept_id - output all files
            for c in concept_list:
                all_output_files[i].write('%s\n' %c)
                if 'full_files' in all_output_files[i + int(len(all_output_files) / 2)]:
                    print(all_output_files[i + int(len(all_output_files) / 2)])
                    raise Exception('ERROR: We already have full files so need to get rid of it from the all_output_files!')
                if 'no_duplicates' in additional_file_paths[j]:
                    all_output_files[i+int(len(all_output_files)/2)].write('%s\n' %concept_id)

                elif 'random_ids' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % random_id)

                elif 'shuffled_ids' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % shuffled_id)

    ##now focus on validation information for experiments
    print('PROGRESS: finished with src file now onto validation!')
    unique_concept_id_val = list(set(all_concept_ids_val_nums_only) - set(all_concept_ids_nums_only))

    ##set up all dictionaries for all experiments
    for concept_id_val, concept_list_val in ontology_dict_val.items():
        if len(set(concept_list_val)) != len(concept_list_val):
            raise Exception('ERROR WITH MAKING SURE WE HAVE NO DUPLICATES DURING ONTOLOGY DICT BUILD!')
        else:
            pass

        ##experiment dictionaries
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


            ##output all information for exerpiments for validation files
            ##for each concept in the list we have the same concept_id
            for c_val in concept_list_val:
                all_output_files[i].write('%s\n' %c_val)


                if 'no_duplicates' in additional_file_paths[j]:
                    all_output_files[i+int(len(all_output_files)/2)].write('%s\n' %concept_id_val)

                elif 'random_ids' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % random_id_val)

                elif 'shuffled_ids' in additional_file_paths[j]:
                    all_output_files[i + int(len(all_output_files) / 2)].write('%s\n' % shuffled_id_val)




##alphabetical experiment
def alphabetical_output(ontology, concept_norm_files_path, main_files):

    ##final dictionary
    alphabetical_dict = {}  # dictionary from span -> alphabetical_id

    ##open the no duplicates and read in just the spans from both regular and val
    src_file_list = []
    src_file_val_list = []
    with open('%s%s/%s/%s_%s.txt' %(concept_norm_files_path, ontology, 'no_duplicates', ontology, 'combo_src_file_char'), 'r+') as src_file, open('%s%s/%s/%s_%s.txt' %(concept_norm_files_path, ontology, 'no_duplicates', ontology, 'combo_src_file_char_val'), 'r+') as src_file_val:
        ##src file
        for line in src_file:
            span = line.strip('\n')
            src_file_list += [span]

        ##src file val
        for line in src_file_val:
            span = line.strip('\n')
            src_file_val_list += [span]


    all_no_duplicate_spans = src_file_list + src_file_val_list ##list of all spans in order
    ##alphabetize them (sort)
    unique_spans_sorted = sorted(list(set(all_no_duplicate_spans)))
    alphabet_num_digits = len(str(len(unique_spans_sorted)))  # total number of digits we need for this ontology


    ##assign them a new id in order: alphabetized concept id via dictionary
    for r, sorted_span in enumerate(unique_spans_sorted):
        alphabetical_id = ontology + ':' + str(r).zfill(alphabet_num_digits) #pads with zeros up to the correct length - alphabet num digits
        alphabetical_id = ' '.join(alphabetical_id)
        alphabetical_dict[sorted_span] = alphabetical_id

    ##output them to the files following the order of the training and validation files
    all_output_files = []
    for file in main_files:
        current_output_file = open(
            '%s%s/%s/%s_%s.txt' % (concept_norm_files_path, ontology, 'alphabetical', ontology, file), 'w+')
        current_output_file_val = open(
            '%s%s/%s/%s_%s_val.txt' % (concept_norm_files_path, ontology, 'alphabetical', ontology, file), 'w+')

        all_output_files += [current_output_file, current_output_file_val]

    ##all_output_files: src, src_val, tgt, tgt_val
    print('all output files', len(all_output_files))

    #output combos file
    for span in src_file_list:
        all_output_files[0].write('%s\n' % (span))
        all_output_files[2].write('%s\n' %(alphabetical_dict[span]))


    #output combo files val
    for span in src_file_val_list:
        all_output_files[1].write('%s\n' % (span))
        all_output_files[3].write('%s\n' % (alphabetical_dict[span]))






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-tokenized_file_path', type=str, help='a string file path to the tokenized files')
    parser.add_argument('-excluded_files', type=str, help='a string list of the files to exclude in this run - comma delimited with no spaces')
    parser.add_argument('-ontologies', type=str, help='a list of ontologies to use delimited with ,')
    parser.add_argument('-concept_norm_files_path', type=str, help='a file path to the output for the concept norm files for concept normalization')
    parser.add_argument('-full_files_path', type=str,help='the string folder with the full files in it')

    parser.add_argument('--extensions', type=str, help='whether or not we have extensions (leave blank if no extensions, default is None', default=None)

    args = parser.parse_args()

    ontologies = args.ontologies.split(',')
    excluded_files = args.excluded_files.split(',')


    ##character info files
    if args.extensions:
        character_info_file = open('%s%s.txt' %(args.concept_norm_files_path, 'character_info_EXT'), 'w+')
    else:
        character_info_file = open('%s%s.txt' % (args.concept_norm_files_path, 'character_info'), 'w+')

    character_info_file.write('%s\t%s\t%s\t%s\t%s\n' %('ONTOLOGY', 'MIN CONCEPT CHAR LENGTH', 'MAX CONCEPT CHAR LENGTH', 'MIN CONCEPT ID CHAR LENGTH', 'MAX CONCEPT ID CHAR LENGTH'))

    ##loop over each ontology and create concept normalizaiton training data along with experiments data for both training and validation
    for ontology in ontologies:
        print('PROGRESS:', ontology)

        #ONTOLOGY_DICT - pmc_mention_id -> [sentence_num, word, ontology_concept_id, ontology_label]
        filename_combo_list = ['combo_src_file', 'combo_tgt_concept_ids', 'combo_tgt_ont_labels','combo_link_mention_ids','combo_link_sent_nums']


        ##gather all spanned text to then output
        all_output_files = gather_spanned_text(args.tokenized_file_path, args.concept_norm_files_path , ontology, args.full_files_path, filename_combo_list, excluded_files)

        ##add in the additional obo concepts
        additional_obo_concepts(ontology, args.concept_norm_files_path, all_output_files)


        ##make all of them lowercase and uniform! also duplicates! = experiments!
        ##now that we have all the regular files with the full stuff let's do more
        #ontology dictionary by ID
        ontology_dict, ontology_dict_val = ontology_dictionary(ontology, args.concept_norm_files_path, args.full_files_path, character_info_file)
        print('total ontology concept ids:', len(ontology_dict.keys()))

        main_files = ['combo_src_file_char', 'combo_tgt_concept_ids_char']
        ##the experiment file paths with alphetical separately
        additional_file_paths = ['no_duplicates/', 'random_ids/', 'shuffled_ids/']

        ##get rid of duplicates
        no_duplicates_lower(ontology, ontology_dict, ontology_dict_val, args.concept_norm_files_path, main_files, additional_file_paths)

        print('PROGRESS: finished no duplicates, random ids, and shuffled ids')

        ##alphabetical stuff specifically based on no_duplicates
        alphabetical_output(ontology, args.concept_norm_files_path, main_files)

