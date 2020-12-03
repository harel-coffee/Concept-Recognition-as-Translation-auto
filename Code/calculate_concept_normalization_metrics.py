import argparse
import os
import sklearn.metrics


def read_in_concept_norm_results(concept_id_file_path, src_file):

    ##gather all concept ids from the concept_id_file_path and if there is src_file also grab the spanned text

    ont_concept_id_list = [] #list of all concept_ids put together
    ont_concept_id_char_list = [] #list of list of concept_ids in char

    with open('%s' %(concept_id_file_path), 'r+') as concept_id_file:
        for line in concept_id_file:
            ##ConceptMapper can have multiple predictions and so we gather the multiples that can come from conceptmapper delimited by |
            if '|' in line:
                final_concept_list = []
                concept_list = line.strip('\n').replace('#','').split('|')
                for c in concept_list:
                    if 'EXT:' in c:
                        ext_updated_concept = c.replace(':','_').replace('EXT_', 'EXT:')
                        final_concept_list += [ext_updated_concept]
                    else:
                        final_concept_list += [c]


                ont_concept_id_list += [final_concept_list]

            ##otherwise we don't have any issue as OpenNMT only predicts one concept ID
            else:
                ##put the character concept id together
                concept_id = ''.join(line.strip('\n').replace('#', '').split(' '))  # put them back together
                if 'EXT:' in concept_id:
                    updated_concept = concept_id.replace(':','_').replace('EXT_', 'EXT:')
                else:
                    updated_concept = concept_id


                ont_concept_id_list += [updated_concept]

            ##create the final concept id character list
            if len(line.strip('\n').split(' ')) == 1:
                if '|' in line:
                    concept_char_list = [(' '.join(concept)).split(' ') for concept in final_concept_list]
                    ont_concept_id_char_list += [concept_char_list]
                else:
                    concept_id_char = (' '.join(updated_concept)).split(' ')
                    ont_concept_id_char_list += [concept_id_char]
            else:
                ont_concept_id_char_list += [line.strip('\n').strip('#').split(' ')]


    ##if there is a src file: also return the span text lists
    if src_file:
        ont_span_text_list = []
        ont_span_text_char_list = []
        with open('%s' %(concept_id_file_path.replace('tgt_concept_ids', 'src_file')), 'r+') as span_text_file:
            ##grab all the span text line by line in character and full word form
            for line in span_text_file:
                chars = line.strip('\n').split(' ')
                full_word = ''
                for char in chars:
                    if char == '':
                        full_word += ' '
                    else:
                        full_word += '%s' %(char)

                ont_span_text_list += [full_word]

                ont_span_text_char_list += [line.strip('\n').split(' ')]
        return ont_concept_id_list, ont_concept_id_char_list, ont_span_text_list, ont_span_text_char_list

    ##only return the concept id lists
    else:
        return ont_concept_id_list, ont_concept_id_char_list


def char_comparison(pred, gs_ont_list, l):
    ##compare the prediction and gold standard based on the character level for the two lists

    predicted_labels = []
    gold_standard_labels = []

    ##pred = list of characters
    if len(pred) > len(gs_ont_list[l]):
        for k, p in enumerate(pred):
            try:
                if p == gs_ont_list[l][k]:
                    # pred_match_list += [1]
                    predicted_labels += [1]
                else:
                    # pred_match_list += [0]
                    predicted_labels += [0]

                gold_standard_labels += [1]

            ##if gold standard is shorter than these are extra characters in gold standard and it should be 0 in gold standard - false positive for predicted
            except IndexError:
                gold_standard_labels += [0]
                predicted_labels += [1]

    ##gold standard is longer
    else:
        for j, g in enumerate(gs_ont_list[l]):
            gold_standard_labels += [1]
            try:
                if pred[j] == g:
                    predicted_labels += [1]
                else:
                    predicted_labels += [0]

            ##if predicted labels is shorter than it missed stuff and so it should be 0 in predicted for missing stuff (false negative
            except IndexError:
                predicted_labels += [0]
    return gold_standard_labels, predicted_labels



def compare_gs_and_pred_concept_normalization(gs_ont_list, pred_ont_list, comparison_type):
    # return precision, recall, and f1 measure for concept normalization

    ##check that the lenght of the predicted and gold standard are the same
    if len(gs_ont_list) != len(pred_ont_list):
        print(len(gs_ont_list), len(pred_ont_list))
        raise Exception('ERROR: error with predictions accounting for all spanned text because different than gold standard length!')

    ##labels for classification: 0 = not gs, 1 = gs
    gold_standard_labels = []
    predicted_labels = []
    empty_annotation_count = 0

    ##concept id comparison
    if comparison_type.lower() == 'concept_id':
        ##loop over and see if the predicted matches the gold standard (1) otherwise 0 (no match)
        for l, pred in enumerate(pred_ont_list):
            gold_standard_labels += [1]
            if type(pred) == list and gs_ont_list[l] in pred:
                predicted_labels += [1]
            elif pred == gs_ont_list[l]:
                predicted_labels += [1]
            else:
                predicted_labels += [0]


            if pred == '':
                empty_annotation_count += 1

    ##character level comparison
    elif comparison_type.lower().startswith('char'):
        for l, pred in enumerate(pred_ont_list):
            ##find the one with the highest matching value - since there are multiple options for the prediction sometimes
            if type(pred[0]) == list:
                list_gs_labels = []
                list_pred_labels = []
                match_count_list = []
                for p in pred:
                    #character comparison of prediction and gold standard
                    single_gs_labels, single_pred_labels = char_comparison(p, gs_ont_list, l)
                    list_gs_labels += [single_gs_labels]
                    list_pred_labels += [single_pred_labels]
                    match_count_list += [sum(single_pred_labels)]

                ##update all the matching labels
                final_match_index = match_count_list.index(max(match_count_list))
                gold_standard_labels += list_gs_labels[final_match_index]
                predicted_labels += list_pred_labels[final_match_index]
            else:
                ##if only one prediction then calculate character match
                single_gs_labels, single_pred_labels = char_comparison(pred, gs_ont_list, l)
                gold_standard_labels += single_gs_labels
                predicted_labels += single_pred_labels



    else:
        raise Exception('ERROR: need to choose concept_id or character level for comparison')


    ##final metrics to return
    ##check that all label lengths are the same
    if len(gold_standard_labels) != len(predicted_labels):
        print('gold standard lenght', len(gold_standard_labels))
        print('predicted length', len(predicted_labels))
        raise Exception('ERROR: the predicted and gold standard labels are different lengths when they should be the same')

    ##calculate the final metrics
    else:
        ##binary
        precision = sklearn.metrics.precision_score(gold_standard_labels, predicted_labels)
        recall = sklearn.metrics.recall_score(gold_standard_labels, predicted_labels)
        f1_score = sklearn.metrics.f1_score(gold_standard_labels, predicted_labels)

    return precision, recall, f1_score, predicted_labels, gold_standard_labels, empty_annotation_count



def fake_concept_ids(pred_ont_concept_id_list, gs_ont_concept_id_list, training_concept_ids, training_concept_ids_val, pred_labels, gs_labels):
    ##calculate the % of fake concept ids out of the total number of mismatches

    ##calculate the total number of mismatches
    total_num_mismatch = 0
    for i, p in enumerate(pred_labels):
        if p != gs_labels[i]:
            total_num_mismatch += 1

    ##gather all the concept ids that exist
    all_true_concept_ids_set = set(gs_ont_concept_id_list + training_concept_ids + training_concept_ids_val)
    ##all the prediction concept ids
    pred_set = set(pred_ont_concept_id_list)
    ##determine the fake concept ids
    fake_concept_ids_set = pred_set.difference(all_true_concept_ids_set)
    ##calculate the total number of fake concept ids in the predictions
    total_fake_concept_ids = 0
    for pred in pred_ont_concept_id_list:
        if pred in fake_concept_ids_set:
            total_fake_concept_ids += 1

    ##check that the total fake concept ids are smaller than the total mismatches and error if not
    if total_num_mismatch < total_fake_concept_ids:
        print('total num mismatch', total_num_mismatch)
        print('total fake concept ids', total_fake_concept_ids)
        raise Exception('ERROR: issue with num mismatch and fake concept ids')
    else:
        pass

    return float(total_fake_concept_ids)/float(total_num_mismatch)



def new_thing_count(training_list, gs_list):
    ##count how many new things exist in the gold standard evaluation set compared to the training

    training_set = set(training_list) ##all training
    gs_set = set(gs_list) #all gold standard evaluation

    gs_new = gs_set.difference(training_set) #subtract out the training set from the gs set to get new things
    ##count the number the of new thing of interest
    full_count_new_things = 0
    for gs in gs_list:
        if gs in gs_new:
            full_count_new_things += 1

    return full_count_new_things, len(gs_new), gs_new


def check_new_concept_ids(ontology, unique_concept_ids, obo_addition_path):

    ##check how many additional obo concepts came from the .obo files
    all_additional_concept_ids = []
    with open('%s%s/%s_addition.txt' %(obo_addition_path, ontology, ontology), 'r+') as obo_addition_file:
        next(obo_addition_file)
        for line in obo_addition_file:
            all_additional_concept_ids += [line.split('\t')[0]]

    missing_concept_ids = []
    for cid in unique_concept_ids:
        if cid not in all_additional_concept_ids:
            missing_concept_ids += [cid]

    return missing_concept_ids





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ontologies', type=str, help='a list of all the ontologies of interest')
    parser.add_argument('-gold_standard_path', type=str,
                        help='the file path to the gold standard concept ids')
    parser.add_argument('-gs_file_name', type=str, help='the file name extension for the gold standard file of concept ids')
    parser.add_argument('-predicted_path', type=str, help='the file path to the predicted concept ids')
    parser.add_argument('-pred_file_name', type=str, help='the filename extension for the predicted file of concept ids')
    parser.add_argument('-output_path', type=str, help='a file path to the output path for the final results')
    parser.add_argument('-obo_addition_path', type=str, help='a file path to the additional obo files')


    ##optional (default = None)
    parser.add_argument('--extensions', type=str,
                        help='optional: true if these are extension classes otherwise leave blank ', default=None)
    parser.add_argument('--experiments', type=str, help='optional: a list of the experiments of interest , delimited with no spaces', default=None)
    parser.add_argument('--training_path', type=str, help='optional: file path to the training files with all the concept ids', default=None)

    args = parser.parse_args()



    ontologies = args.ontologies.split(',')
    #training path for the fake concept ids calculation
    if args.training_path and args.experiments.lower() == 'full_files':
        full_files = args.experiments
        args.experiments = None


    ##experiments calculations
    if args.experiments:
        experiments = args.experiments.split(',')


    print(args.experiments)

    ##if we don't have the experiments then we output summary files in total
    if not args.experiments:
        if args.extensions:
            results_summary = 'concept_normalization_results_summary_EXT'
        else:
            results_summary = 'concept_normalization_results_summary'

        ##example filenames
        # gs_CHEBI_combo_tgt_concept_ids_char.txt (gold standard)
        # gs_CHEBI-model-char_step_100000_pred.txt (predictions)


        ##full summary output
        with open('%s%s.txt' %(args.output_path, results_summary), 'w+') as full_summary_output_file:
            #open('%s%s_%s.txt' %(args.output_path, results_summary, 'discontinuous_spans_only'), 'w+') as discontinuous_summary_output_file:

            full_summary_output_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % ('Ontology', 'total annotations', 'precision (concept id)', 'recall (concept id)', 'f1 score (concept id)', 'precision (character)', 'recall (character)', 'f1 score (character)'))

            if args.training_path:
                full_summary_output_file.write('\t%s\t%s\t%s\t%s\t%s\n' %('% fake concept id (out of total mismatch)', '# new concept ids', '# unique new concept ids', '# new spanned text', '# unique new spanned text'))

                if args.extensions:
                    ##training span text concept norm metrics
                    training_concept_norm_file = open('%s%s.txt' %(args.output_path, 'training_spans_concept_norm_results_summary_EXT'), 'w+')

                    ##new span text concept norm metrics
                    new_concept_norm_file = open(
                        '%s%s.txt' % (args.output_path, 'new_spans_concept_norm_results_summary_EXT'), 'w+')

                else:
                    ##training span text concept norm metrics
                    training_concept_norm_file = open(
                        '%s%s.txt' % (args.output_path, 'training_spans_concept_norm_results_summary'), 'w+')

                    ##new span text concept norm metrics
                    new_concept_norm_file = open(
                        '%s%s.txt' % (args.output_path, 'new_spans_concept_norm_results_summary'), 'w+')

                training_concept_norm_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('Ontology', 'total annotations', 'precision (concept id)', 'recall (concept id)', 'f1 score (concept id)', 'precision (character)', 'recall (character)', 'f1 score (character)'))

                new_concept_norm_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('Ontology', 'total annotations', 'precision (concept id)', 'recall (concept id)', 'f1 score (concept id)', 'precision (character)', 'recall (character)', 'f1 score (character)'))


            else:
                full_summary_output_file.write('\n')





            for ontology in ontologies:
                print('PROGRESS:', ontology)
                ##get gold standard stuff
                gs_concept_id_file_path = '%s%s/%s_%s%s.txt' %(args.gold_standard_path, ontology, 'gs', ontology, args.gs_file_name)

                gs_ont_concept_id_list, gs_ont_char_list, gs_ont_span_text_list, gs_ont_span_text_char_list = read_in_concept_norm_results(gs_concept_id_file_path, 'true')

                ##get predicted stuff
                pred_concept_id_file_path = '%s%s/%s_%s%s.txt' %(args.predicted_path, ontology, 'gs', ontology, args.pred_file_name)

                pred_ont_concept_id_list, pred_ont_char_list = read_in_concept_norm_results(pred_concept_id_file_path, None)
                ##comparison type = concept id
                comparison_type = 'concept_id'
                precision, recall, f1_score, pred_labels, gs_labels, empty_annotation_count = compare_gs_and_pred_concept_normalization(gs_ont_concept_id_list, pred_ont_concept_id_list, comparison_type)

                ##comparison type = character level
                comparison_type = 'character'
                precision_char, recall_char, f1_score_char, pred_labels_char, gs_labels_char, empty_annotation_count_char = compare_gs_and_pred_concept_normalization(gs_ont_char_list, pred_ont_char_list, comparison_type)

                full_summary_output_file.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
                ontology, len(gs_ont_span_text_list), precision, recall, f1_score, precision_char, recall_char, f1_score_char))

                ##fake concept id information + more
                if args.training_path:
                    training_concept_ids, training_chars, training_span_text, training_span_text_char = read_in_concept_norm_results('%s%s/%s/%s%s.txt' %(args.training_path, ontology, full_files, ontology, args.gs_file_name), 'true')

                    training_concept_ids_val, training_chars_val, training_span_text_val, training_span_text_char_val = read_in_concept_norm_results('%s%s/%s/%s%s_%s.txt' %(args.training_path, ontology, full_files, ontology, args.gs_file_name, 'val'), 'true')

                    fake_concept_ids_fraction = fake_concept_ids(pred_ont_concept_id_list, gs_ont_concept_id_list, training_concept_ids, training_concept_ids_val, pred_labels, gs_labels)

                    full_summary_output_file.write('\t%.4f' % fake_concept_ids_fraction)

                    training_concept_id_list = training_concept_ids + training_concept_ids_val
                    training_span_text_list = training_span_text + training_span_text_val
                    ##new concept ids
                    full_count_new_concept_ids, num_unique_new_concept_ids, new_concept_ids = new_thing_count(training_concept_id_list, gs_ont_concept_id_list)
                    if num_unique_new_concept_ids != 0:
                        print(num_unique_new_concept_ids) #all of them exist in the obo_addition file so we can rerun everything and make sure we are good
                        missing_concept_ids = check_new_concept_ids(ontology, new_concept_ids, args.obo_addition_path)
                        print(missing_concept_ids) #the ones missing from the obo_addition file
                    ##new char (span text)
                    full_count_new_span_text, num_unique_new_span_text, new_span_text = new_thing_count(training_span_text_list, gs_ont_span_text_list)


                    ##output the summary info
                    full_summary_output_file.write('\t%s\t%s\t%s\t%s' %(full_count_new_concept_ids, num_unique_new_concept_ids, full_count_new_span_text, num_unique_new_span_text))

                    ##new line
                    full_summary_output_file.write('\n')


                    ##TRAINING SPANS AND NEW SPANS INFORMATION FOR OUTPUT SUMMARY
                    ##separate the results for training span text and new span text:
                    training_gs_spans_concept_id_list = []
                    training_gs_spans_concept_id_list_char = []
                    new_gs_spans_concept_id_list = []
                    new_gs_spans_concept_id_list_char = []

                    training_pred_spans_concept_id_list = []
                    training_pred_spans_concept_id_list_char = []
                    new_pred_spans_concept_id_list = []
                    new_pred_spans_concept_id_list_char = []

                    for i, gs_ont_span_text in enumerate(gs_ont_span_text_list):
                        ##new span texts
                        if gs_ont_span_text in new_span_text:
                            new_gs_spans_concept_id_list += [gs_ont_concept_id_list[i]]
                            new_gs_spans_concept_id_list_char += [gs_ont_char_list[i]]
                            new_pred_spans_concept_id_list += [pred_ont_concept_id_list[i]]
                            new_pred_spans_concept_id_list_char += [pred_ont_char_list[i]]

                        else:
                            training_gs_spans_concept_id_list += [gs_ont_concept_id_list[i]]
                            training_gs_spans_concept_id_list_char += [gs_ont_char_list[i]]
                            training_pred_spans_concept_id_list += [pred_ont_concept_id_list[i]]
                            training_pred_spans_concept_id_list_char += [pred_ont_char_list[i]]


                    #check that we have all the correct new spans:

                    if list({len(new_gs_spans_concept_id_list), len(new_pred_spans_concept_id_list), len(new_gs_spans_concept_id_list_char), len(new_pred_spans_concept_id_list_char)})[0] != full_count_new_span_text or len({len(training_gs_spans_concept_id_list), len(training_pred_spans_concept_id_list), len(training_gs_spans_concept_id_list_char), len(training_pred_spans_concept_id_list_char)}) != 1:
                        print('new information:', [len(new_gs_spans_concept_id_list), len(new_pred_spans_concept_id_list), len(new_gs_spans_concept_id_list_char), len(new_pred_spans_concept_id_list_char)])
                        print('full count new spans', full_count_new_span_text)

                        print('training info:', [len(training_gs_spans_concept_id_list), len(training_pred_spans_concept_id_list), len(training_gs_spans_concept_id_list_char), len(training_pred_spans_concept_id_list_char)])

                        raise Exception('ERROR: issue with separating new and training span text lengths!')

                    else:
                        pass


                    ##new span information:
                    ##comparison type = concept id
                    comparison_type = 'concept_id'
                    new_precision, new_recall, new_f1_score, new_pred_labels, new_gs_labels, new_empty_annotation_count = compare_gs_and_pred_concept_normalization(new_gs_spans_concept_id_list, new_pred_spans_concept_id_list, comparison_type)

                    training_precision, training_recall, training_f1_score, training_pred_labels, training_gs_labels, tranining_empty_annotation_count = compare_gs_and_pred_concept_normalization(
                        training_gs_spans_concept_id_list, training_pred_spans_concept_id_list, comparison_type)

                    ##comparison type = character level
                    comparison_type = 'character'
                    new_precision_char, new_recall_char, new_f1_score_char, new_pred_labels_char, new_gs_labels_char, new_empty_annotation_count_char = compare_gs_and_pred_concept_normalization(
                        new_gs_spans_concept_id_list_char, new_pred_spans_concept_id_list_char, comparison_type)

                    training_precision_char, training_recall_char, training_f1_score_char, training_pred_labels_char, training_gs_labels_char, training_empty_annotation_count_char = compare_gs_and_pred_concept_normalization(training_gs_spans_concept_id_list_char, training_pred_spans_concept_id_list_char, comparison_type)

                    ##new concept norm files
                    new_concept_norm_file.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                        ontology, len(new_gs_spans_concept_id_list), new_precision, new_recall, new_f1_score, new_precision_char, new_recall_char, new_f1_score_char))


                    ##training concept norm files
                    training_concept_norm_file.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                        ontology, len(training_gs_spans_concept_id_list), training_precision, training_recall, training_f1_score, precision_char, training_recall_char, training_f1_score_char))







                else:
                    full_summary_output_file.write('\n')




    ##experiments results
    else:
        ##output files for all experiments - summary files
        for experiment in experiments:
            print('experiment:', experiment)
            if args.extensions:
                results_summary = '%s_%s' %(experiment, 'concept_normalization_results_summary_EXT')
            else:
                results_summary = '%s_%s' %(experiment, 'concept_normalization_results_summary')

            with open('%s%s.txt' % (args.output_path, results_summary), 'w+') as full_summary_output_file:
                if experiment.lower() == 'conceptmapper':
                    full_summary_output_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('Ontology', 'total annotations', 'precision (concept id)', 'recall (concept id)', 'f1 score (concept id)', 'precision (character)', 'recall (character)', 'f1 score (character)', 'empty annotation count'))
                else:
                    full_summary_output_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('Ontology', 'total annotations', 'precision (concept id)', 'recall (concept id)', 'f1 score (concept id)', 'precision (character)', 'recall (character)', 'f1 score (character)', '% fake concept ids'))


                ##loop through each ontology
                for ontology in ontologies:
                    print('PROGRESS:', ontology)

                    ##get gold standard stuff
                    if experiment.lower() == 'conceptmapper':
                        gs_concept_id_file_path = '%s%s/%s_%s%s.txt' %(args.gold_standard_path, ontology, 'gs', ontology, args.gs_file_name)
                    else:
                        gs_concept_id_file_path = '%s%s/%s/%s_%s%s.txt' % (args.gold_standard_path, ontology, experiment, 'gs', ontology, args.gs_file_name)

                    gs_ont_concept_id_list, gs_ont_char_list = read_in_concept_norm_results(gs_concept_id_file_path, None)


                    ##get predicted stuff
                    if experiment.lower() == 'conceptmapper':
                        pred_concept_id_file_path = '%s%s_%s_%s.txt' % (
                            args.predicted_path, 'gs',  ontology.upper(), 'combo_tgt_concept_ids')
                    else:
                        pred_concept_id_file_path = '%s%s/%s_%s_%s%s.txt' % (
                    args.predicted_path, ontology, 'gs', experiment, ontology, args.pred_file_name)

                    #read in the files to create lists of all the annotations/concept ids
                    pred_ont_concept_id_list, pred_ont_char_list = read_in_concept_norm_results(pred_concept_id_file_path, None)


                    ##comparison type = concept id
                    comparison_type = 'concept_id'
                    precision, recall, f1_score, pred_labels, gs_labels, empty_annotation_count = compare_gs_and_pred_concept_normalization(gs_ont_concept_id_list, pred_ont_concept_id_list, comparison_type)

                    ##comparison type = character level
                    comparison_type = 'character'
                    precision_char, recall_char, f1_score_char, pred_labels_char, gs_labels_char, empty_annotation_count_char = compare_gs_and_pred_concept_normalization(
                        gs_ont_char_list, pred_ont_char_list, comparison_type)

                    ##ConceptMapper results for baseline
                    if experiment.lower() == 'conceptmapper':
                        full_summary_output_file.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s' % (ontology, len(gs_ont_concept_id_list), precision, recall, f1_score, precision_char, recall_char, f1_score_char, empty_annotation_count))
                    else:
                        full_summary_output_file.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (ontology, len(gs_ont_concept_id_list), precision, recall, f1_score, precision_char, recall_char,f1_score_char))

                    ##fake concept id information + more
                    if args.training_path:
                        training_concept_ids, training_chars, training_span_text, training_span_text_char = read_in_concept_norm_results('%s%s/%s/%s%s.txt' % (args.training_path, ontology, experiment, ontology, args.gs_file_name), 'true')

                        training_concept_ids_val, training_chars_val, training_span_text_val, training_span_text_char_val = read_in_concept_norm_results('%s%s/%s/%s%s_%s.txt' % (args.training_path, ontology, experiment, ontology, args.gs_file_name, 'val'), 'true')

                        fake_concept_ids_fraction = fake_concept_ids(pred_ont_concept_id_list, gs_ont_concept_id_list,training_concept_ids, training_concept_ids_val, pred_labels, gs_labels)

                        full_summary_output_file.write('\t%.4f\n' % fake_concept_ids_fraction)


                    else:
                        full_summary_output_file.write('\n')
