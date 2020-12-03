import os
import gzip
import argparse



def read_obo_file(obo_file_path, ontology, GO_dict, extensions):

    ontology_concept_dict = {} #ontology_concept_id -> [name, def, [synonyms]]

    ##gather all unused concepts - filepath for extensions vs not
    if extensions:
        #unused_classes_and_substitute_extension_classes_for_CHEBI+extensions_annotations.txt
        unused_concept_file_path = '%s%s/%s/unused_classes_and_substitute_extension_classes_for_%s_annotations.txt' % (obo_file_path, ontology.replace('_EXT', ''), ontology.replace('_EXT', '+extensions'), ontology.replace('_EXT', '+extensions'))

    else:
        unused_concept_file_path = '%s%s/%s/unused_classes_for_%s_annotations.txt' %(obo_file_path, ontology, ontology, ontology)



    unused_concept_list = []
    unused_concept_dict = {} #only concepts that are the same but with _EXT added

    ##gather all unused concepts so we don't use them
    with open(unused_concept_file_path, 'r') as unused_concepts_file:
        for line in unused_concepts_file:
            line_list = line.strip('\n').split('\t')
            if extensions:
                unused_concept_list += [line_list[0]]
                if len(line.split('\t')) == 2 and line_list[0] == line_list[1].replace('_EXT', ''):
                    unused_concept_dict[line_list[0]] = line_list[1]
                else:
                    pass
            else:
                unused_concept_list += [line.strip('\n')]

    ##grab all obo concepts from .obo files for each ontology - obo file path
    ##only take the namespace that is correct. For GO look at the namespace category also
    if 'GO' in ontology:
        if extensions:
            if 'GO_MF' in ontology:
                obo_file_path_full = '%s%s/%s/%s%s.obo.gz' % (obo_file_path, ontology.replace('_EXT', ''), ontology.replace('_EXT', '+extensions'), 'GO_MF_stub+', ontology.replace('_EXT', '_extensions'))
            else:
                obo_file_path_full = '%s%s/%s/%s%s.obo.gz' % (obo_file_path, ontology.replace('_EXT', ''), ontology.replace('_EXT', '+extensions'), 'GO+', ontology.replace('_EXT','_extensions'))

        else:
            obo_file_path_full = '%s%s/%s/%s.obo.gz' % (obo_file_path, ontology, ontology, 'GO')

    else:
        if extensions:
            obo_file_path_full = '%s%s/%s/%s.obo.gz' % (obo_file_path, ontology.replace('_EXT',''), ontology.replace('_EXT', '+extensions'), ontology.replace('_EXT', '+extensions'))
            # print(obo_file_path_full)
        else:
            obo_file_path_full = '%s%s/%s/%s.obo.gz' %(obo_file_path, ontology, ontology, ontology)


    ##gather all concepts from .obo file that are relevant to the ontology
    with gzip.open(obo_file_path_full, 'rb') as obo_file:
        current_concept = None
        ##file in bytes and need to convert to strings - decode(utf-8)
        valid_term = False
        for line in obo_file:

            ##obofile: term, id, name, (namespace), def, synonym
            str_line = line.decode('utf-8').strip('\n')

            ##capture the term
            if '[Term]' == str_line:
                valid_term = True

            ##concept id
            elif valid_term and str_line.startswith('id:'):

                ##determine the current concept id with extensions caveat
                if ontology_concept_dict.get(str_line.split(' ')[-1]):
                    print(str_line.split(' ')[-1])
                    print(ontology_concept_dict[str_line.split(' ')[-1]])
                    raise Exception('ERROR WITH ONTOLOGY DICT INITIALIZING!')
                elif extensions and unused_concept_dict.get(str_line.split(' ')[-1]):
                    current_concept = unused_concept_dict[str_line.split(' ')[-1]]
                elif str_line.split(' ')[-1] not in unused_concept_list and (ontology.replace('_EXT', '') in str_line.split(' ')[-1].split(':')[0] or ('GO' in ontology and 'GO' in str_line.split(' ')[-1].split(':')[0])):
                    current_concept = str_line.split(' ')[-1]
                else:
                    current_concept = None


                ##initialize the ontology_concept_dict
                if current_concept:
                    ontology_concept_dict[current_concept] = [None, None, []]


                else:
                    ##unused concepts
                    pass

                ##reset the valid term for next round
                valid_term = False

            ##name of concept
            elif current_concept and str_line.startswith('name:'):
                ontology_concept_dict[current_concept][0] = str_line[len('name: '):]

            ##for GO need namespace to be correct ontology (MF, BP, CC)
            elif current_concept and 'GO' in ontology and str_line.startswith('namespace:'):
                name_space = str_line.split(': ')[-1]
                ##need to make sure we have the correct namespace for GO: BP, CC, MF
                if name_space != GO_dict[ontology]:
                    ontology_concept_dict.pop(current_concept)
                    current_concept = None
                    continue


            ##definition of concept
            elif current_concept and str_line.startswith('def:'):
                ontology_concept_dict[current_concept][1] = str_line.split('"')[1]

            ##synonyms of concept - only take if Exact or special for PR
            elif current_concept and str_line.startswith('synonym:'):
                synonym_info = str_line.split('"')[2]
                ##only take exact synonyms except for PR take the related synonyms also


                if 'EXACT' in synonym_info:
                    ##or ('RELATED' in synonym_info and 'InChI' not in synonym_info and 'InChIKey' not in synonym_info) and str_line.split('"')[1] != '.'
                    ontology_concept_dict[current_concept][2] += [str_line.split('"')[1]]
                elif (ontology == 'PR_EXT' or ontology == 'PR') and 'RELATED' in synonym_info:
                    ontology_concept_dict[current_concept][2] += [str_line.split('"')[1]]


            else:
                pass


    return ontology_concept_dict


def output_ontology_addition_file(ontology, ontology_concept_dict, concept_norm_files_path):
    ##output the additional obo concepts
    with open('%s%s/%s_addition.txt' %(concept_norm_files_path, ontology, ontology), 'w+') as additional_ont_concepts_file:
        additional_ont_concepts_file.write('%s\t%s\n' %(ontology, len(ontology_concept_dict)))

        ##ontology_concept_id -> [name, def, [synonyms]]
        for ont_id in ontology_concept_dict.keys():
            [name, definition, synonyms] = ontology_concept_dict[ont_id]

            ##not everything has a definition and that is okay!
            if definition and 'OBSOLETE' in definition:
                pass
            else:
                additional_ont_concepts_file.write('%s\t%s\t%s\t%s\n' %(ont_id, name, definition, synonyms))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-obo_file_path', type=str, help='a string file path to the corpus concept annotations')
    parser.add_argument('-concept_norm_files_path', type=str, help='a string file path tot he concept norm files for the output')
    parser.add_argument('-ontologies', type=str, help='a list of ontologies to use delimited with , with no spaces')

    ##optional: default is none
    parser.add_argument('--extensions', type=str,
                        help='whether or not we have extensions (leave blank if no extensions, default is None', default=None)

    args = parser.parse_args()

    ontologies = args.ontologies.split(',')

    ##GO is weird with namespaces so we help it
    GO_dict = {'GO_BP':'biological_process', 'GO_CC':'cellular_component', 'GO_MF':'molecular_function', 'GO_BP_EXT':'biological_process', 'GO_CC_EXT':'cellular_component', 'GO_MF_EXT':'molecular_function'}

    ##grab all additional obo concepts for each ontology
    for ontology in ontologies:
        print('PROGRESS:', ontology)

        ##gather all ontology concepts from the .obo file
        ##ontology_concept_id -> [name, def, [synonyms]]
        ontology_concept_dict = read_obo_file(args.obo_file_path, ontology, GO_dict, args.extensions)

        ##output the files to the concept_normalized spot to use to train for concept normalization
        output_ontology_addition_file(ontology, ontology_concept_dict, args.concept_norm_files_path)

