import os
import argparse
import random



def get_gold_standard_spans(evaluation_files, gold_standard_bionlp_path, ontology):

    ##return list of gold standard spans and list of concept IDs
    ont_gs_spanned_text = []
    ont_gs_concept_ids = []

    ##loop over all ontology files to get these things
    for root, directories, filenames in os.walk('%s%s/' % (gold_standard_bionlp_path, ontology.lower())):
        for filename in sorted(filenames):
            pmcid = filename.split('.')[0]
            if pmcid in evaluation_files:
                with open('%s' % (root + filename), 'r+') as bionlp_file:
                    for line in bionlp_file:
                        ##line needs to start with T for annotation purposes (bad lines with Relationships starting with R)
                        if line.startswith('T'):
                            ont_gs_spanned_text += [line.strip('\n').split('\t')[-1]] ##last column contains the spanned text
                            ont_gs_concept_ids += [line.split('\t')[1].split(' ')[0]] ##second column contains the concept ID but space delimited with the span start and end

                        ##bad lines starting with R in the files
                        else:
                            pass


    if len(ont_gs_spanned_text) != len(ont_gs_concept_ids):
        raise Exception('ERROR: error with gathering spans and concept ids with length!')

    return ont_gs_spanned_text, ont_gs_concept_ids


def output_gold_standard_spans(ont_gs_spanned_text, ont_gs_concept_ids, ontology, output_path, gs_combo_src_char, gs_combo_tgt_concept_id_char):
    ##output files for gold standard spans
    with open('%s%s/%s_%s_%s.txt' %(output_path, ontology, 'gs', ontology,  gs_combo_src_char), 'w+') as src_char_file, open('%s%s/%s_%s_%s.txt' %(output_path, ontology, 'gs', ontology,  gs_combo_tgt_concept_id_char), 'w+') as tgt_concept_id_char_file, open('%s%s/%s_%s_%s.txt' %(output_path, ontology, 'gs', ontology,  gs_combo_src_char.replace('_char', '')), 'w+') as src_file, open('%s%s/%s_%s_%s.txt' %(output_path, ontology, 'gs', ontology,  gs_combo_tgt_concept_id_char.replace('_char', '')), 'w+') as tgt_concept_id_file:
        for i, span_text in enumerate(ont_gs_spanned_text):

            ##full file with no spaces
            src_file.write('%s\n' %(span_text))
            tgt_concept_id_file.write('%s\n' %(ont_gs_concept_ids[i]))

            ##file with spaces on char level
            word_char = ' '.join(span_text)
            src_char_file.write('%s\n' %(word_char))

            concept_id_char = ' '.join(ont_gs_concept_ids[i])
            tgt_concept_id_char_file.write('%s\n' %(concept_id_char))



def get_experiment_info(concept_norm_file_path, experiment, ontology):
    ##get the experiment info and return a dictionary from span text -> concept id

    experiment_dict = {} #dict from span_text -> concept id
    all_span_text = []
    all_concept_ids = []

    ##loop over files of both training and val
    for root, directories, filenames in os.walk('%s%s/%s/' % (concept_norm_file_path, ontology, experiment)):
        for filename in sorted(filenames):
            ##grab all text spans
            if 'combo_src_file_char' in filename and not filename.startswith('gs'):
                with open(root + filename, 'r') as src_file:
                    for line in src_file:
                        all_span_text += [line.strip('\n')]
            ##grab all concept ids
            elif 'combo_tgt_concept_ids' in filename and not filename.startswith('gs'):
                with open(root + filename, 'r') as tgt_concept_id_file:
                    for line in tgt_concept_id_file:
                        all_concept_ids += [line.strip('\n')]

            else:
                pass

    ##check that the lengths are the same
    if len(all_span_text) != len(all_concept_ids):
        print(len(all_span_text), len(all_concept_ids))
        raise Exception('ERROR: issue with matching lengths for spans and concept ids!')
    else:
        pass

    ##create the dictionary connecting the span text to the concept id (pairs should be unique)
    for a, span_text in enumerate(all_span_text):
        if experiment_dict.get(span_text) and all_concept_ids[a] not in experiment_dict[span_text]:
            experiment_dict[span_text] += [all_concept_ids[a]]
        else:
            experiment_dict[span_text] = [all_concept_ids[a]]


    return experiment_dict


def create_concept_id_dicts(experiments, all_experiment_dicts, ontology):
    ##initialize dictionary for each experiment
    all_concept_id_experiment_dicts = {} #dict from duplicate_concept_id -> experiment_concept_id
    for experiment in experiments:
        all_concept_id_experiment_dicts[experiment] = {}

    ##make sure no duplicates exist because otherwise we have no reference
    if 'no_duplicates' not in experiments:
        raise Exception('ERROR: need to put no_duplicates in the list ideally at the start')
    else:
        pass

    ##loop over each span and create concept id dictionaries
    for span, concept_id_list in all_experiment_dicts['no_duplicates'].items():
        ##random_ids
        if 'random_ids' in experiments:
            if len(all_experiment_dicts['random_ids'][span]) != len(concept_id_list):
                raise Exception('ERROR: issue with random id concept id list for each span!')
            else:
                for i, c_id in enumerate(concept_id_list):
                    if all_concept_id_experiment_dicts['random_ids'].get(c_id):

                        ##check that there are no duplicates
                        if all_concept_id_experiment_dicts['random_ids'][c_id] != all_experiment_dicts['random_ids'][span][i]:
                            print(c_id)
                            print(all_concept_id_experiment_dicts['random_ids'][c_id])
                            print(all_experiment_dicts['random_ids'][span][i])
                            raise Exception('ERROR: issue with duplicate c_ids random ids')
                        else:
                            pass
                    else:
                        all_concept_id_experiment_dicts['random_ids'][c_id] = all_experiment_dicts['random_ids'][span][i]
        ##shuffled_ids
        if 'shuffled_ids' in experiments:
            if len(all_experiment_dicts['shuffled_ids'][span]) != len(concept_id_list):
                raise Exception('ERROR: issue with shuffled id concept id list for each span!')
            else:
                for i, c_id in enumerate(concept_id_list):
                    if all_concept_id_experiment_dicts['shuffled_ids'].get(c_id):
                        if all_concept_id_experiment_dicts['shuffled_ids'][c_id] != all_experiment_dicts['shuffled_ids'][span][i]:
                            print(c_id)
                            print(all_concept_id_experiment_dicts['shuffled_ids'][c_id])
                            print(all_experiment_dicts['shuffled_ids'][span][i])
                            raise Exception('ERROR: issue with duplicate c_ids shuffled ids')
                        else:
                            pass
                    else:
                        all_concept_id_experiment_dicts['shuffled_ids'][c_id] = all_experiment_dicts['shuffled_ids'][span][i]


        ##alphabetical: check that there is only one unique concept id for each span and then set that for the final dictionary from span -> unique concept id
        if 'alphabetical' in experiments:
            if len(all_experiment_dicts['alphabetical'][span]) > 1:
                print(span)
                print(all_experiment_dicts['alphabetical'][span])
                print(len(all_experiment_dicts['alphabetical'][span]))
                raise Exception('ERROR: issue with alphabetical dict having multiple concepts for same span')
            else:
                all_concept_id_experiment_dicts['alphabetical'][span] = all_experiment_dicts['alphabetical'][span][0]


    return all_concept_id_experiment_dicts




def output_gold_standard_spans_for_experiments(ont_gs_spanned_text, ont_gs_concept_ids, ontology, experiments, all_concept_id_experiment_dicts, output_path, gs_combo_src_char, gs_combo_tgt_concept_id_char):

    ##output the gold standard spans for the experiments with the changed concept ids

    ##errors
    error_count_list = []
    unique_error_count_list = []

    ##output for each experiment
    for experiment in experiments:
        experiment_error_count = 0
        unique_errors = set()
        experiment_concept_dict = all_concept_id_experiment_dicts[experiment]

        with open('%s%s/%s/%s_%s_%s.txt' %(output_path, ontology, experiment.lower(), 'gs', ontology,  gs_combo_src_char), 'w+') as src_char_file, open('%s%s/%s/%s_%s_%s.txt' %(output_path, ontology, experiment.lower(), 'gs', ontology,  gs_combo_tgt_concept_id_char), 'w+') as tgt_concept_id_char_file:

            for i, span_text in enumerate(ont_gs_spanned_text):

                ##file with spaces on char level - the gs files are not char level
                word_char = ' '.join(span_text.lower())
                src_char_file.write('%s\n' % (word_char))

                ##the no duplicates has the same ids as the original gold standard - no id change
                if experiment.lower() == 'no_duplicates':
                    experimental_concept_id_char = ' '.join(ont_gs_concept_ids[i])


                ##alphabetical is based on spans - can be new spans so we are keeping the original gs concept id
                elif experiment.lower() == 'alphabetical':
                    ##make the id the new one based on alphabetical concepts

                    try:
                        experimental_concept_id_char = experiment_concept_dict[word_char.lower()] #all lowercase
                    ##if the spanned text is not seen before we leave the original concept ID - maybe assign the next one in order to ones we don't know
                    except KeyError:
                        unique_errors.add(word_char)
                        experiment_error_count += 1
                        experimental_concept_id_char = ' '.join(ont_gs_concept_ids[i]) #original concept ID since we don't have a new one

                ##shuffled id and random ids we can change the concept id via the dictionaries
                else:
                    ##make the id the new one
                    concept_id_char = ' '.join(ont_gs_concept_ids[i])
                    ##if the concept id is not seen before we leave the original concept ID
                    try:
                        experimental_concept_id_char = experiment_concept_dict[concept_id_char]
                    except KeyError:
                        if ontology.replace('_EXT', '') in ont_gs_concept_ids[i]:
                            print(ontology)
                            print(ont_gs_concept_ids[i])
                            raise Exception('ERROR: issue with making sure to get all the obo addition classes!')
                        else:
                            experimental_concept_id_char = ' '.join(ont_gs_concept_ids[i])
                            unique_errors.add(concept_id_char)
                            experiment_error_count += 1


                ##output the tgt concept id - from the experiment
                tgt_concept_id_char_file.write('%s\n' %(experimental_concept_id_char))

        print(experiment, 'num errors:', experiment_error_count)
        error_count_list += [experiment_error_count]
        unique_error_count_list += [len(unique_errors)]
    return error_count_list, unique_error_count_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-evaluation_files', type=str,
                        help='a list of pmcids to evaluate the models on delimited with , with no spaces in between PMC IDs')
    parser.add_argument('-gold_standard_bionlp_path', type=str,
                        help='the file path to the gold standard bionlp format for all the PMC IDs of interest')
    parser.add_argument('-ontologies', type=str, help='a list of all the ontologies of interest')
    parser.add_argument('-output_path', type=str, help='a file path to the output path for the final results')


    # ##optional arguments to add
    # parser.add_argument('--extensions', type=str,
    #                     help='optional: true if these are extension classes otherwise leave blank ', default=None)
    parser.add_argument('--experiments', type=str, help='a list of experiments to grab delimited with a comma and no spaces', default=None)
    parser.add_argument('--concept_norm_file_path', type=str, help='the file path to the experiment concept norm files', default=None)
    args = parser.parse_args()

    ##split up the 2 lists so we have them
    evaluation_files = args.evaluation_files.split(',')
    ontologies = args.ontologies.split(',')
    if args.experiments:
        experiments = args.experiments.split(',')

    # if args.extensions:
    #     gs_combo_src_char = 'combo_src_file_char_EXT'
    #     gs_combo_tgt_concept_id_char = 'combo_tgt_concept_ids_char_EXT'
    # else:
    gs_combo_src_char = 'combo_src_file_char'
    gs_combo_tgt_concept_id_char = 'combo_tgt_concept_ids_char'


    ##output the number of errors in concept id/span for each ontology for all experiments
    if args.experiments:
        if 'EXT' in ontologies[0]:
            experiment_error_file = open('%s%s.txt' %(args.concept_norm_file_path, 'gs_experiment_error_ids_counts_EXT'), 'w+')
        else:
            experiment_error_file = open('%s%s.txt' %(args.concept_norm_file_path, 'gs_experiment_error_ids_counts'), 'w+')

        experiment_error_file.write('%s\t' %('ONTOLOGY'))
        for i, experiment in enumerate(experiments):
            if i == len(experiments) - 1:
                experiment_error_file.write('%s\t%s\n' % (experiment.upper(), 'unique ' + experiment.upper()))
            else:
                experiment_error_file.write('%s\t%s\t' %(experiment.upper(), 'unique ' + experiment.upper()))
    else:
        pass


    for ontology in ontologies:
        print('ONTOLOGY:', ontology)

        ##gather all gs spans and concept ids
        ont_gs_spanned_text, ont_gs_concept_ids = get_gold_standard_spans(evaluation_files, args.gold_standard_bionlp_path, ontology)

        ##output the spans and concept ids
        output_gold_standard_spans(ont_gs_spanned_text, ont_gs_concept_ids, ontology, args.output_path, gs_combo_src_char, gs_combo_tgt_concept_id_char)


        ##prep the experiment information for gold standard
        if args.experiments:
            experiment_error_file.write('%s\t' %ontology)

            ##grab all experiment dicts for this ontology
            all_experiment_dicts = {} #dictionary from experiment to the dictionary
            for experiment in experiments:
                experiment_dict = get_experiment_info(args.concept_norm_file_path, experiment, ontology)
                print('experiment:', experiment, 'num pairs:', len(experiment_dict.keys()))
                all_experiment_dicts[experiment] = experiment_dict



            ##need to create experiment_dicts from no_duplicate concept id -> other experiment concept
            if len(set([len(all_experiment_dicts[experiment].keys()) for experiment in experiments])) != 1:
                for experiment in experiments:
                    print(experiment, len(all_experiment_dicts[experiment].keys()))
                raise Exception('ERROR: issue with lengths of experimental dicts which should be the same with alphabetical!')
            else:
                pass



            ##gather all concept id dicts
            all_concept_id_experiment_dicts = create_concept_id_dicts(experiments, all_experiment_dicts, ontology)

            ##output all for the gold standard files
            error_count_list, unique_error_count_list = output_gold_standard_spans_for_experiments(ont_gs_spanned_text, ont_gs_concept_ids, ontology, experiments, all_concept_id_experiment_dicts, args.concept_norm_file_path,gs_combo_src_char, gs_combo_tgt_concept_id_char)


            ##output the error count for each experiment
            for i, ec in enumerate(error_count_list):
                if i == len(experiments) - 1:
                    experiment_error_file.write('%s\t%s\n' % (ec, unique_error_count_list[i]))
                else:
                    experiment_error_file.write('%s\t%s\t' % (ec, unique_error_count_list[i]))










