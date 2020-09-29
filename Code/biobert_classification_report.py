import argparse
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report



def read_in_ner_conll_results(ner_conll_results_path, biotags, ontology, output_path, gold_standard):
    if gold_standard.lower() == 'true':
        true_labels = [] #sentence level
        pred_labels = [] #sentence_level
        true_discontinuous_spans = 0
        true_discontinuous_sentences = 0
        with open('%s%s' %(ner_conll_results_path, 'NER_result_conll.txt'), 'r+') as ner_conll_results_file:
            next(ner_conll_results_file) #header: TOKEN	TRUE	PREDICTED
            sentence_true = []
            sentence_pred = []
            for line in ner_conll_results_file:
                ##new line
                if line in ['\n', '\r\n']:
                    if 'I-disc' in sentence_true:
                        true_discontinuous_sentences += 1
                    else:
                        pass


                    true_labels += [sentence_true]
                    pred_labels += [sentence_pred]

                    #reset!
                    sentence_true = []
                    sentence_pred = []
                else:
                    # print(line.strip('\n').split('\t'))
                    token, true, pred = line.strip('\n').split('\t')
                    if true not in biotags:
                        raise Exception('ERROR WITH TRUE BIOTAGS!')
                    elif pred not in biotags:
                        raise Exception('ERROR WITH PRED BIOTAGS')
                    else:
                        pass

                    #TODO: change O- to something it can see - I-disc
                    if true == 'O-':
                        # print('got here')
                        true_discontinuous_spans += 1
                        true = 'I-disc'

                    if pred == 'O-':
                        # print('and here')
                        pred = 'I-disc'



                    sentence_true += [true]
                    sentence_pred += [pred]


                # raise Exception('hold!')

        #classification report
        with open('%s%s' %(output_path, 'biobert_classification_report.txt'), 'w+') as output_file:
            output_file.write('%s\n' %(ontology))
            output_file.write('%s\t%s\n' %('NUM SENTENCES WITH DISCONTINUITIES', true_discontinuous_sentences))

            output_file.write('%s\t%s\n\n' %('NUM TOTAL DISCONTINUITIES:', true_discontinuous_spans))
            try:
                output_file.write(classification_report(true_labels, pred_labels))
            except ZeroDivisionError:
                output_file.write('NO ONTOLOGY TERMS: ONLY O FOR OUTSIDE!')


    ##only has predictions so we can only do that to read it in
    #no gold standard!
    else:
        pass








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ner_conll_results_path', type=str, help='')
    parser.add_argument('--biotags', type=str, help='')
    parser.add_argument('--ontology', type=str, help='')
    parser.add_argument('--output_path', type=str, help='')
    parser.add_argument('--gold_standard', type=str, help='')
    args = parser.parse_args()

    biotag_list = args.biotags.split(',')
    read_in_ner_conll_results(args.ner_conll_results_path, biotag_list, args.ontology, args.output_path, args.gold_standard)