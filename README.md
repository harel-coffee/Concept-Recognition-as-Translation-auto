# Concept-Recognition-as-Translation

All code and models for reformulating Concept Recognition as a machine translation task using the 2019 CRAFT Shared Tasks (https://sites.google.com/view/craft-shared-task-2019/home) for Concept Annotation. 

## Contents

### Code
All of the code to generate the knowtator files from CRAFT to BIO- format, and tune, train, and test the models that appear in the /Models folder. All of the ten ontologies in CRAFT are processed separately creating a model for each one. We split the task of Concept Recognition into two:
1. Span Detection (also referred to as named entity recognition or mention detection): to delimit a particluar textual region that refers to some ontoloical concept.
2. Concept Normalization (also referred to as named entity normalization): to identify the specific ontological concept to which the text span (from span detection) refers to.

### Models
All of the models for span detection and concept normalization with separate models for each ontology.

### Output Folders
All output folders for all algorithms
1. Tokenized Files: All articles by ontology preprocessed into BIO- Format. 
2. PMCID_files_sentences and PMCID_files_sentences_EXT: All articles by sentence with sentence information and concept annotation information for both the 10 ontologies without and with extensions (EXT) respectively. 
3. Concept_Norm_Files: All files related to the concept normalization process including the preprocessed files to the prediction files for each ontology with and without extensions. There are 5 different folders within each ontology corresponding to the different runs of Open NMT:

	a. full_files = the token level where all mappings of spans to concepts, even though some are the same string and concept, only occurring in different places in the text. This is the original run used for all training algorithms. Token-ids in the paper.

	b. no_duplicates = the type level where there is one mapping of a given span to a concept regardless of frequency. Type-ids in the paper.

	c. random_ids = an experimental run replacing the concept IDs with random numbers of the same length, drawn without replacement. Random-ids in the paper. 

	d. shuffled_ids = an experimental run that scrambles the relationship between concept and ID. Shuffled-ids in the paper. 

	e. alphabetical = an experimental run that alphabetizes the concepts by label and assigns consecutive IDs. Alphabetical-ids in the paper.


### CRAFT-3.1.3
Version 3 of the CRAFT corpus that was used for the 2019 CRAFT Shared Task. More details can be found here: https://github.com/UCDenver-ccp/CRAFT/releases/tag/v3.1.3. We use the concept-annotation folder for all annotations. 

### craft-st-2019
All of the files from the CRAFT Shared Task held out set of 30 documents for gold standard evaluation. The final evaluation utilizes these documents. To run the formal final evaluation:
1. Follow the instructions for Evaluation via Docker: https://github.com/UCDenver-ccp/craft-shared-tasks/wiki/Evaluation-via-Docker-(Recommended-Method). We are using VERSION = 3.1.3_0.1.2 for the docker images to use.
2. Follow the instructions for the Concept Annotation Task Evaluation for the shared task: https://github.com/UCDenver-ccp/craft-shared-tasks/wiki/Concept-Annotation-Task-Evaluation
3. The final formats for the evaluation in the bionlp format is in /Output_Folders/concept_system_output/MODEL/, where MODEL is one of the 6 models tested for span detection. 
4. The general command to run the docker is as follows:
docker run --rm -v /Output_Folders/concept_system_output/MODEL:/files-to-evaluate -v /craft-st-2019:/corpus-distribution ucdenverccp/craft-eval:3.1.3_0.1.2 sh -c '(cd /home/craft/evaluation && boot javac eval-concept-annotations -c /corpus-distribution -i /files-to-evaluate -g /corpus-distribution/gold-annotations/craft-ca)'. The paths to the output files can be changed if need be. 
5. The output file, ONTOLOGY_results.tsv where ONTOLOGY is one of the 10 ontologies or an ontology with extension classes, will output in the ontology file within the MODEL folder. 
