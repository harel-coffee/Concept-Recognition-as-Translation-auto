import argparse
import os
import sklearn.metrics

def get_spans_from_bionlp(full_bionlp_path, evaluation_files):
    #return a list of tuples (pmcid, span) #should be unique (make sure to check)

    ##grab all the gold standard information
    # if algo[0].lower() == 'gold_standard':
    full_span_list = []
    discontinuous_count = 0
    discontinuous_span_list = []

    ##loop over all bionlp files
    for root, directories, filenames in os.walk('%s' % (full_bionlp_path)):
        for filename in sorted(filenames):
            pmcid = filename.split('.')[0]

            ##check that the pmcid is in the evaluation files list
            if pmcid in evaluation_files:
                current_span_list = []
                with open('%s' %(root + filename), 'r+') as bionlp_file:
                    ##gather the total number of annotations per file
                    total_annotations = 0
                    ##loop over each line to grab the annotations
                    for line in bionlp_file:

                        ##all strings - ensure that the start is with the T for an annotation number
                        if line.strip('\n').split('\t')[0].startswith('T'):
                            (annotation_num, concept_info, text_span) = line.strip('\n').split('\t')

                            ##concept information with spans
                            concept_id = concept_info.split(' ')[0]
                            span_info = concept_info.split(' ')[1:]  # taking everything since we also have discontinuous spans

                            current_span_list += [(pmcid, span_info)]  # all strings
                            total_annotations += 1

                            ## collect discontinuous spans
                            if len(concept_info.split(' ')) > 3:
                                discontinuous_count += 1
                                discontinuous_span_list += [(pmcid, span_info)]
                            else:
                                pass


                        ##weird lines with R which should be relations
                        else:
                            # print(pmcid, line)
                            pass
                    ##ensure that we have gathered all data
                    if total_annotations != len(current_span_list):
                        print(pmcid)
                        print(total_annotations)
                        print(len(current_span_list))
                        raise Exception('ERROR: the gs_span list should be the number of lines in the files - check if they are unique')
                    else:
                        full_span_list += current_span_list

            ##only take the pmcids we care about otherwise pass
            else:
                pass



    return current_span_list, discontinuous_count, discontinuous_span_list


def compare_gs_and_pred_spans(gold_standard_span_list, predicted_span_list):
    #return precision, recall, and f1 measure for spans comparing gold standard and predicted

    full_span_set = [] #all the annotations from both predicted and gold standard ordered


    ##cannot set the two lists so we combine them
    if len(gold_standard_span_list) > len(predicted_span_list):
        larger_list = gold_standard_span_list
        smaller_list = predicted_span_list
    else:
        larger_list = predicted_span_list
        smaller_list = gold_standard_span_list

    ##create the full span set list - all span texts in both gold standard and predicted
    full_span_set += larger_list
    for s in smaller_list:
        if s in larger_list:
            pass
        else:
            full_span_set += [s]

    print('full span set length', len(full_span_set))


    ##labels for classification: 0 = not gs span, 1 = gs span
    gold_standard_labels = []
    predicted_labels = []
    for f in full_span_set:
        if f in gold_standard_span_list:
            gold_standard_labels += [1]
        else:
            gold_standard_labels += [0]

        if f in predicted_span_list:
            predicted_labels += [1]
        else:
            predicted_labels += [0]


    ##binary metrics
    precision = sklearn.metrics.precision_score(gold_standard_labels, predicted_labels)
    recall = sklearn.metrics.recall_score(gold_standard_labels, predicted_labels)
    f1_score = sklearn.metrics.f1_score(gold_standard_labels, predicted_labels)




    return precision, recall, f1_score





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-evaluation_files', type=str,
                        help='a list of pmcids to evaluate the models on delimited with , with no spaces in between PMC IDs')
    parser.add_argument('-gold_standard_bionlp_path', type=str, help = 'the file path to the gold standard bionlp format for all the PMC IDs of interest')
    parser.add_argument('-predicted_bionlp_path', type=str, help='the file path to the predicted bionlp format')
    parser.add_argument('-algo_list', type=str, help='a list of all the algorithms to evaluate the results of delimited with , with no spaces in between PMC IDs')
    parser.add_argument('-ontologies', type=str, help='a list of all the ontologies of interest')
    parser.add_argument('-output_path', type=str, help='a file path to the output path for the final results')
    parser.add_argument('--extensions', type=str, help='optional: true if these are extension classes otherwise leave blank ', default=None)

    args = parser.parse_args()

    ##split up the 2 lists so we have them
    evaluation_files = args.evaluation_files.split(',')
    algo_list = args.algo_list.split(',')
    ontologies = args.ontologies.split(',')

    if args.extensions:
        results_summary = 'span_detection_results_summary_EXT'
    else:
        results_summary = 'span_detection_results_summary'

    ##summary output file
    with open('%s%s.txt' %(args.output_path, results_summary), 'w+') as full_summary_output_file, open('%s%s_%s.txt' %(args.output_path, results_summary, 'discontinuous_spans_only'), 'w+') as discontinuous_summary_output_file:
        ##both output files for full and discontinuity separately
        full_summary_output_file.write('%s\t' % ('Ontology'))
        discontinuous_summary_output_file.write('%s\t%s\t' % ('Ontology', 'support'))

        ##loop over each algorithm to get all the scores
        algo_list_files = {} #algo ->  file
        for algo in algo_list:
            ##create the column for each algorithm
            if algo == algo_list[-1]:
                full_summary_output_file.write('%s\n' %(algo))
                discontinuous_summary_output_file.write('%s\n' % (algo))
            else:
                full_summary_output_file.write('%s\t' %(algo))
                discontinuous_summary_output_file.write('%s\t' % (algo))

            ##create the files for the full output per algorithm both full information and discontinuous spans only
            if args.extensions:
                algo_file_name = 'span_detection_results_EXT'
            else:
                algo_file_name = 'span_detection_results'

            algo_file = open('%s%s_%s.txt' %(args.output_path, algo, algo_file_name), 'w+')
            algo_file_disc = open('%s%s_%s_%s.txt' %(args.output_path, algo, algo_file_name, 'discontinuous_spans_only'), 'w+')

            algo_file.write('%s\t%s\t%s\t%s\n' %('Ontology', 'precision', 'recall', 'f1 score'))
            algo_file_disc.write('%s\t%s\t%s\t%s\t%s\n' % ('Ontology', 'support', 'precision', 'recall', 'f1 score'))

            algo_list_files[algo] = [algo_file, algo_file_disc]

        ##loop over all ontologies and then within each article to get info
        for ontology in ontologies:

            print('PROGRESS: ontology currently working on is:', ontology)
            ##the output information for each ontology for the summary
            full_summary_output_file.write('%s\t' % (ontology))
            discontinuous_summary_output_file.write('%s\t' % (ontology))


            ##loop over each article in the ontology
            ##gold_standard stuff
            gs_full_bionlp_path = '%s%s/' %(args.gold_standard_bionlp_path, ontology.lower()) #lowercase ontology for gold standard
            gold_standard_span_list, gs_discontinuous_count, gs_discontinuous_span_list = get_spans_from_bionlp(gs_full_bionlp_path, evaluation_files)
            ##output the discontinuous information
            discontinuous_summary_output_file.write('%s\t' % (gs_discontinuous_count))


            ##predicted stuff by algorithm (loop over each algorithm)
            for algo in algo_list:
                pred_full_bionlp_path = '%s%s/%s/' %(args.predicted_bionlp_path, algo.upper(), ontology.upper())
                current_pred_span_list, pred_discontinuous_count, pred_discontinuous_span_list = get_spans_from_bionlp(pred_full_bionlp_path, evaluation_files)

                ##comparison of gold standard to predicted
                print('PROGRESS: COMPARE PREDICTED TO GOLD STANDARD FOR ALGORITHM:', algo)
                precision, recall, f1_score = compare_gs_and_pred_spans(gold_standard_span_list, current_pred_span_list)

                ##comparison for just discontinuous spans
                precision_disc, recall_disc, f1_score_disc = compare_gs_and_pred_spans(gs_discontinuous_span_list, pred_discontinuous_span_list)

                ##output full summary which is just f1-score
                if algo == algo_list[-1]:
                    full_summary_output_file.write('%.4f\n' % (f1_score))
                    discontinuous_summary_output_file.write('%.4f\n' % (f1_score_disc))
                else:
                    full_summary_output_file.write('%.4f\t' % (f1_score))
                    discontinuous_summary_output_file.write('%.4f\t' % (f1_score_disc))

                ##output each algorithm summary
                current_algo_file = algo_list_files[algo][0]
                current_algo_file_disc = algo_list_files[algo][1]
                current_algo_file.write('%s\t%.4f\t%.4f\t%.4f\n' %(ontology, precision, recall, f1_score))
                current_algo_file_disc.write('%s\t%.4f\t%.4f\t%.4f\n' %(ontology, precision_disc, recall_disc, f1_score_disc))


