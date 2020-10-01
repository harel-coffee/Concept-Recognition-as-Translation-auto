import argparse
import os


def eval_result_model_num(eval_results_path, ontology, algo):
    if 'BIOBERT' not in algo:
        raise Exception('ERROR WITH ALGORITHM! WE ONLY DO THIS FOR BIOBERT!')
    else:
        with open('%s%s' %(eval_results_path, 'eval_results.txt'), 'r+') as eval_results_file:
            for line in eval_results_file:
                if line.startswith('global_step'):
                    global_step = int(line.strip('\n').split(' ')[-1])

                else:
                    pass


        ##output file
        with open('%s%s' %(eval_results_path, 'global_step_num.txt'), 'w+') as global_step_file:
            global_step_file.write('%s' %global_step)







if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ontology', type=str, help='the ontology of interest')


    parser.add_argument('-eval_results_path', type=str,
                        help='the file path to where the biobert model results are saved (eval_results.txt specifically')

    parser.add_argument('-algo', type=str,
                        help='a list of the algorithms we are using to run for span detection delimtited with ,')

    args = parser.parse_args()

    eval_result_model_num(args.eval_results_path, args.ontology, args.algo)
