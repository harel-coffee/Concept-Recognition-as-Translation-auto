import argparse


def get_algo_results(final_output_path, algo, ontologies):
    algo_dict = {}  # dict from: ontology -> [SER, precision, recall, F1-score]

    ##output file for all results
    with open('%s%s_%s.txt' %(final_output_path, algo, 'full_craft_docker_results'), 'w+') as algo_output_file:
        algo_output_file.write('%s\t%s\t%s\t%s\t%s\n' %('Ontology', 'SER', 'Precision', 'Recall', 'F1-score'))

        ##loop over all ontologies and grab the information we want
        for ontology in ontologies:
            with open('%s%s/%s/%s_%s.tsv' %(final_output_path, algo, ontology.upper(), ontology.lower(), 'results'), 'r+') as final_results_file:
                last_line = final_results_file.readlines()[-1].strip('\n').split('\t') #has the total information for the algorithm

                ##ensure that the last line is the true total information
                if last_line[0] != 'TOTAL':
                    print(algo, ontology)
                    print(last_line)
                    raise Exception('ERROR: issue with the last line of the results file - prob docker errors!')
                else:
                    pass

                ##get the SER, precision, recall, F1-score
                SER = last_line[-4]
                precision = last_line[-3]
                recall = last_line[-2]
                f1_score = last_line[-1]

                algo_dict[ontology] = [SER, precision, recall, f1_score]

                ##output all results
                algo_output_file.write('%s\t%s\t%s\t%s\t%s\n' %(ontology, SER, precision, recall, f1_score))


    return algo_dict


def output_results(algo_list, ontologies, final_output_path, algo_dict_list, final_filename):

    ##output the full results with just F1 measure
    with open('%s%s.txt' % (final_output_path, final_filename), 'w+') as full_output_file:
        ##the header of the file/table
        full_output_file.write('%s' %('Ontology'))
        for algo in algo_list:
            full_output_file.write('\t%s' %(algo))
        full_output_file.write('\n')

        ##loop over the ontologies to output the final results
        for ontology in ontologies:
            full_output_file.write('%s' %(ontology))

            for a, algo in enumerate(algo_list):
                algo_dict = algo_dict_list[a]
                full_output_file.write('\t%s' %(algo_dict[ontology][-1])) #grab the f1-score only

            full_output_file.write('\n')







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-ontologies', type=str, help='a list of ontologies to use delimited with ,')
    parser.add_argument('-final_output_path', type=str, help='the file path to the final output results for the full run of craft')
    parser.add_argument('-algo', type=str, help='a list of all the algos of interest to merge results for comma delimited with no spaces')

    ##optional arguments to add
    parser.add_argument('--extensions', type=str,
                        help='optional: true if these are extension classes otherwise leave blank ', default=None)



    args = parser.parse_args()

    ontologies = args.ontologies.split(',')
    algo_list = args.algo.split(',')

    if args.extensions:
        final_filename = 'full_craft_docker_results_EXT'
    else:
        final_filename = 'full_craft_docker_results'

    ##collect all the information from each algo (per ontology)
    #list of dictionaries per algo: ontology -> [SER, precision, recall, F1-score]
    algo_dict_list = []
    for algo in algo_list:
        print('CURRENT ALGORITHM:', algo)
        # algo_dict = {} #dict from: ontology -> [SER, precision, recall, F1-score]
        algo_dict = get_algo_results(args.final_output_path, algo, ontologies)
        algo_dict_list += [algo_dict]

    ##output all the information
    output_results(algo_list, ontologies, args.final_output_path, algo_dict_list, final_filename)

