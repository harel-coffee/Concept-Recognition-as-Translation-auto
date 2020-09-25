import os
import gzip
import argparse



def read_obo_file(obo_file_path, ontology, GO_dict, extensions):

    ontology_concept_dict = {} #ontology_concept_id -> [name, def, [synonyms]]

    ##gather all unused concepts
    if extensions:
        #unused_classes_and_substitute_extension_classes_for_CHEBI+extensions_annotations.txt
        unused_concept_file_path = '%s%s/%s/unused_classes_and_substitute_extension_classes_for_%s_annotations.txt' % (obo_file_path, ontology.replace('_EXT', ''), ontology.replace('_EXT', '+extensions'), ontology.replace('_EXT', '+extensions'))

    else:
        unused_concept_file_path = '%s%s/%s/unused_classes_for_%s_annotations.txt' %(obo_file_path, ontology, ontology, ontology)



    unused_concept_list = []



    with open(unused_concept_file_path, 'r') as unused_concepts_file:
        for line in unused_concepts_file:
            if extensions:
                unused_concept_list += [line.strip('\n').split('\t')[0]]
            else:
                unused_concept_list += [line.strip('\n')]

    ##only take the namespace that is correct. For go look at the namespace category also
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
        else:
            obo_file_path_full = '%s%s/%s/%s.obo.gz' %(obo_file_path, ontology, ontology, ontology)




    with gzip.open(obo_file_path_full, 'rb') as obo_file:
        current_concept = None
        ##file in bytes and need to convert to strings - decode(utf-8)
        for line in obo_file:
            ##obofile: term, id, name, (namespace), def, synonym
            str_line = line.decode('utf-8').strip('\n')

            ##capture the term
            if '[Term]' == str_line:
                pass

            elif str_line.startswith('id:'):
                if ontology_concept_dict.get(str_line.split(' ')[-1]):
                    raise Exception('ERROR WITH ONTOLOGY DICT INITIALIZING!')
                elif str_line.split(' ')[-1] not in unused_concept_list:
                    current_concept = str_line.split(' ')[-1]
                    if current_concept.split(':')[0] == ontology:
                        ontology_concept_dict[str_line.split(' ')[-1]] = [None, None, []]
                    elif 'GO' in ontology and current_concept.split(':')[0] == 'GO':
                        ##need to also check the namespace category to see if CC, MF, or BP
                        ontology_concept_dict[str_line.split(' ')[-1]] = [None, None, []]
                    else:
                        ##making sure to take the correct name space for the ontology
                        current_concept = None
                        pass
                else:
                    ##unused concepts
                    pass

            elif current_concept and str_line.startswith('name:'):
                ontology_concept_dict[current_concept][0] = str_line[len('name: '): ]

            elif current_concept and 'GO' in ontology and str_line.startswith('namespace:'):
                name_space = str_line.split(': ')[-1]
                ##need to make sure we have the correct namespace for GO: BP, CC, MF
                if name_space != GO_dict[ontology]:
                    ontology_concept_dict.pop(current_concept)
                    current_concept = None
                    continue



            elif current_concept and str_line.startswith('def:'):
                ontology_concept_dict[current_concept][1] = str_line.split('"')[1]

            elif current_concept and str_line.startswith('synonym:'):
                synonym_info = str_line.split('"')[2]
                ##only take exact synonyms except for PR take the related synonyms also


                if 'EXACT' in synonym_info:
                    ##or ('RELATED' in synonym_info and 'InChI' not in synonym_info and 'InChIKey' not in synonym_info) and str_line.split('"')[1] != '.'
                    ontology_concept_dict[current_concept][2] += [str_line.split('"')[1]]
                elif ontology == 'PR' and 'RELATED' in synonym_info:
                    ontology_concept_dict[current_concept][2] += [str_line.split('"')[1]]


            else:
                pass


    return ontology_concept_dict


def output_ontology_addition_file(ontology, ontology_concept_dict, concept_norm_files_path):
    with open('%s%s/%s_addition.txt' %(concept_norm_files_path, ontology, ontology), 'w+') as additional_ont_concepts_file:
        additional_ont_concepts_file.write('%s\t%s\n' %(ontology, len(ontology_concept_dict)))


        ##ontology_concept_id -> [name, def, [synonyms]]
        for ont_id in ontology_concept_dict.keys():
            [name, definition, synonyms] = ontology_concept_dict[ont_id]
            if 'OBSOLETE' not in definition:
                additional_ont_concepts_file.write('%s\t%s\t%s\t%s\n' %(ont_id, name, definition, synonyms))
            else:
                pass



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

    GO_dict = {'GO_BP':'biological_process', 'GO_CC':'cellular_component', 'GO_MF':'molecular_function', 'GO_BP_EXT':'biological_process', 'GO_CC_EXT':'cellular_component', 'GO_MF_EXT':'molecular_function'}


    for ontology in ontologies:
        print('PROGRESS:', ontology)

        ##ontology_concept_id -> [name, def, [synonyms]]
        ontology_concept_dict = read_obo_file(args.obo_file_path, ontology, GO_dict, args.extensions)

        ##output the files to the concept_normalized spot to use to train for normalization
        output_ontology_addition_file(ontology, ontology_concept_dict, args.concept_norm_files_path)

