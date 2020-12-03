import argparse
import os
import pickle
from statistics import mean, median


def training_data_info(ontology, training_annotation_path):
    ##annotation counts both total and unique for the training data
    annotation_count = []
    unique_annotation_count = []

    ## mention_ID_dict: mention_ID -> (start_list, end_list, spanned_text, mention_class_ID, class_label, sentence_number)
    ##load the mention_ID_dict for each ontology and count
    for root, directories, filenames in os.walk(training_annotation_path + ontology + '/'):
        for filename in sorted(filenames):

            ##currently per pmcid file but want to combine per ontology the concept mentions - both total and unique
            if filename.endswith('.pkl') and 'mention_id_dict' in filename:
                mention_ID_dict_pkl = open(root+filename, 'rb')
                mention_ID_dict = pickle.load(mention_ID_dict_pkl)
                annotation_count += [len(mention_ID_dict.keys())]
                unique_annotation_set = set()
                for mention_ID in mention_ID_dict.keys():
                    unique_annotation_set.add(mention_ID_dict[mention_ID][3])

                unique_annotation_count += [len(unique_annotation_set)]

    ##counts
    total_annotation_count = sum(annotation_count)
    total_unique_annotation_count = sum(unique_annotation_count)

    average_annotation_count = mean(annotation_count)
    average_unique_annotation_count = mean(unique_annotation_count)

    median_annotation_count = median(annotation_count)
    median_unique_annotation_count = median(unique_annotation_count)


    return total_annotation_count, total_unique_annotation_count, average_annotation_count, average_unique_annotation_count, median_annotation_count, median_unique_annotation_count



def count_obo_addition(ontology, obo_addition_path):
    ##the first line has the ontology and number of additional obo concepts added - grab that
    with open('%s%s/%s_%s.txt' %(obo_addition_path, ontology, ontology,'addition'), 'r+') as obo_addition_file:
        obo_addition_count = int(obo_addition_file.readline().strip('\n').split('\t')[-1])
        return obo_addition_count


def gs_data_info(ontology, gold_standard_path):
    ##count the number of annotations for the gold standard data
    gs_annotation_count = []
    gs_unique_annotation_count = []

    ##loop over all ontology files to get these things
    for root, directories, filenames in os.walk('%s%s/' % (gold_standard_path, ontology.lower())):
        for filename in sorted(filenames):
            ont_gs_concept_ids = []
            with open('%s' % (root + filename), 'r+') as bionlp_file:
                for line in bionlp_file:
                    ##line needs to start with T for annotation purposes (bad lines with Relationships starting with R)
                    if line.startswith('T'):
                        ont_gs_concept_ids += [line.split('\t')[1].split(' ')[0]]  ##second column contains the concept ID but space delimited with the span start and end

                    ##bad lines starting with R in the files
                    else:
                        pass

            gs_annotation_count += [len(ont_gs_concept_ids)]
            gs_unique_annotation_count += [len(set(ont_gs_concept_ids))]


    ##final counts
    gs_total_annotation_count = sum(gs_annotation_count)
    gs_total_unique_annotation_count = sum(gs_unique_annotation_count)

    gs_average_annotation_count = mean(gs_annotation_count)
    gs_average_unique_annotation_count = mean(gs_unique_annotation_count)

    gs_median_annotation_count = median(gs_annotation_count)
    gs_median_unique_annotation_count = median(gs_unique_annotation_count)

    return gs_total_annotation_count, gs_total_unique_annotation_count, gs_average_annotation_count, gs_average_unique_annotation_count, gs_median_annotation_count, gs_median_unique_annotation_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-ontologies', type=str, help='a list of all the ontologies of interest')
    parser.add_argument('-training_annotation_path', type=str, help='file path to all the tokenized files with the mention dict')
    parser.add_argument('-obo_addition_path', type=str, help='file path to the obo addition files for concept normalization')
    parser.add_argument('-gold_standard_path', type=str, help='file path the gold standard bionlp files')

    parser.add_argument('-output_path', type=str, help='a file path to the output path for the final results')


    args = parser.parse_args()

    ontologies = args.ontologies.split(',')

    ##overview table output
    with open('%s%s.txt' %(args.output_path, 'overview_table_summary'), 'w+') as overview_file:
        overview_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %('ONTOLOGY', 'total training annotation count', 'average training annotation count per article', 'median training annotation count per article', 'total unique training annotation count', 'average unique training annotation count per article', 'median unique training annotation count per article', 'OBO addition count', 'total annotations for concept norm', 'total unique annotations for concept norm', 'GS total annotation count', 'GS average annotation count per article', 'GS median annotation count per article', 'GS total unique annotation count', 'GS average unique annotation count per article', 'GS median unique annotation count per article'))

        ## get all stats per ontology
        for ontology in ontologies:
            print('PROGRESS:', ontology)

            ##total annotation counts - total and unique
            total_annotation_count, total_unique_annotation_count, average_annotation_count, average_unique_annotation_count, median_annotation_count, median_unique_annotation_count = training_data_info(ontology, args.training_annotation_path)

            ##obo addition count
            obo_addition_count = count_obo_addition(ontology, args.obo_addition_path)

            ##total for concept norm count - obo addition + total annotation count
            total_for_concept_norm = obo_addition_count + total_annotation_count
            total_unique_for_concept_norm = obo_addition_count + total_unique_annotation_count

            ##gold standard total annotation counts - total and unique
            gs_total_annotation_count, gs_total_unique_annotation_count, gs_average_annotation_count, gs_average_unique_annotation_count, gs_median_annotation_count, gs_median_unique_annotation_count = gs_data_info(ontology, args.gold_standard_path)

            ##full output per ontology
            overview_file.write('%s\t%s\t%.4f\t%.4f\t%s\t%.4f\t%.4f\t%s\t%s\t%s\t%s\t%.4f\t%.4f\t%s\t%.4f\t%.4f\n' %(ontology, total_annotation_count, average_annotation_count, median_annotation_count, total_unique_annotation_count, average_unique_annotation_count, median_unique_annotation_count, obo_addition_count, total_for_concept_norm, total_unique_for_concept_norm, gs_total_annotation_count, gs_average_annotation_count, gs_median_annotation_count, gs_total_unique_annotation_count, gs_average_unique_annotation_count, gs_median_unique_annotation_count))
