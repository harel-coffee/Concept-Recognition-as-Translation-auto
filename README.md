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

### CRAFT-3.1.3
Version 3 of the CRAFT corpus that was used for the 2019 CRAFT Shared Task. More details can be found here: https://github.com/UCDenver-ccp/CRAFT/releases/tag/v3.1.3. We use the concept-annotation folder for all annotations. 

### craft-st-2019
All of the files from the CRAFT Shared Task held out set of 30 documents for gold standard evaluation. The final evaluation utilizes these documents. To run the formal final evaluation:
1. Follow the instructions for Evaluation via Docker: https://github.com/UCDenver-ccp/craft-shared-tasks/wiki/Evaluation-via-Docker-(Recommended-Method). We are using VERSION = 3.1.3_0.1.2 for the docker images to use.
2. Follwow the instructions for the Concept Annotation Task Evaluation for the shared task: https://github.com/UCDenver-ccp/craft-shared-tasks/wiki/Concept-Annotation-Task-Evaluation
3. The final formats for the evaluation in the bionlp format is in /Output_Folders/concept_system_output/MODEL/, where MODEL is one of the 6 models tested for span detection. 
4. The general command to run the docker is as follows:
docker run --rm -v /Output_Folders/concept_system_output/MODEL:/files-to-evaluate -v /craft-st-2019:/corpus-distribution ucdenverccp/craft-eval:3.1.3_0.1.2 sh -c '(cd /home/craft/evaluation && boot javac eval-concept-annotations -c /corpus-distribution -i /files-to-evaluate -g /corpus-distribution/gold-annotations/craft-ca)'
The paths to the output files can be changed if need be. 
5. The output file, ONTOLOGY_results.tsv where ONTOLOGY is one of the 10 ontologies or an ontology with extension classes, will output in the ontology file within the MODEL folder. 
