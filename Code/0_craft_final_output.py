import os
import argparse



def final_craft_output(ontology, concept_system_output_path, evaluation_files, final_output_path, algo, algo_filename_info):

    ##run over all files in the model concept system output
    for root, directories, filenames in os.walk('%s%s/' % (concept_system_output_path, ontology)):
        for filename in sorted(filenames):
            ##add in the different models as we add them (
            if filename.endswith('.bionlp') and filename.startswith('%s_%s' %(ontology, algo_filename_info)) and filename.split('_')[-1].replace('.bionlp','') in evaluation_files:
                ##the pmcid that will be the name of the file
                pmcid = filename.split('_')[-1].replace('.bionlp','')
                ##read in the current output file from the model runs
                with open(root+filename, 'r+') as model_output_file:
                    model_output_text = model_output_file.read()
                ##output the final output to the final output path while renaming the file
                with open('%s%s/%s/%s.bionlp' %(final_output_path, algo, ontology, pmcid), 'w+') as final_output_file:
                    final_output_file.write(model_output_text)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ontologies', type=str, help='a list of ontologies to use delimited with ,')
    parser.add_argument('-eval_path', type=str, help='the file path to the evaluation folder')
    parser.add_argument('-evaluation_files', type=str, help='a list of the evaluation files delimited by ,')
    parser.add_argument('-concept_system_output', type=str, help='the file path to the concept system output')
    parser.add_argument('-final_output', type=str, help='the file path to the final output for the full craft evaluation with docker')
    parser.add_argument('-algo', type=str, help='algorithm of choice that matches the folder used (all uppercase)')
    parser.add_argument('-algo_filename_info', type=str, help='the unique identifier for the algorithm in the output files/ the model name')
    args = parser.parse_args()

    ontologies = args.ontologies.split(',')
    evaluation_files = args.evaluation_files.split(',')
    concept_system_output_path = args.eval_path + args.concept_system_output

    ##loop over each ontology to gather the final output
    for ontology in ontologies:
        print('PROGRESS:', ontology)
        final_craft_output(ontology, concept_system_output_path, evaluation_files, args.final_output, args.algo.upper(), args.algo_filename_info)


