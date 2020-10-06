# Concept-Recognition-as-Translation

All code and models for reformulating Concept Recognition as a machine translation task using the 2019 CRAFT Shared Tasks (https://sites.google.com/view/craft-shared-task-2019/home) for Concept Annotation. We split the task of Concept Recognition into two:
1. Span Detection (also referred to as named entity recognition or mention detection): to delimit a particluar textual region that refers to some ontoloical concept.
2. Concept Normalization (also referred to as named entity normalization): to identify the specific ontological concept to which the text span (from span detection) refers to.

Clone this github repository. Some files were too large to upload here so we uploaded them to Zenodo. 
Download all the files and put all the .tar.gz files within the Concept-Recognition-as-Translation/ folder. 
Then in the command line navigate to the folder you just put them in and run Code/untar_all_files.sh which will untar all of the files (FILENAME) using the command: tar -xzvf FILENAME. This will put all the files in the correct place on your local machine. Do not unzip them using the builtin compression because it will make duplicate copies of everything. The files can be found here: 
They include:
1. Output_Folders-Tokenized_Files.tar.gz
2. Output_Folders-Evaluation_Files.tar.gz
3. Output_Folders-Concept_Norm_Files.tar.gz
4. Code-biobert_v1.0_pubmed_pmc.tar.gz
5. Models-SPAN_DETECTION-ONTOLOGY-BIOBERT.tar.gz (where ONTOLOGY is the ontology of choice - 20 ontologies with a corresponding Biobert file)

Then follow the instructions within the Code section. The Models section discusses the different models evaluated here. The Output Folders section provides details on how all output from the code is organized and stored. Lastly, the two CRAFT sections are details on the corpus we used for this project. 

To use the models we have here, skip to the Evaluation Pipeline section or the CRAFT Evaluation Pipeline section and edit the bash scripts mentioned. 

To tune and train models from scratch, start from the beginning of the Code section. 

## Contents

### Code
All of the code to generate BIO- format from the knowtator files from CRAFT, and tune, train, test, and evaluate the models that appear in the /Models folder. All of the ten ontologies along with their extension classes, in CRAFT are processed separately creating a model for each one. 
#####Tuning and Training Models
1. run_preprocess_docs.sh: processes all concept annotations into BIO- format, collect extra ontology concepts from the ontologies that are not in CRAFT, and processes all ontology concepts into OpenNMT format for concept normalization. 

	a. Inputs: File paths to CRAFT (articles and annotations), a list of the ontologies, the output paths, and any excluded files

	b. Outputs: Tokenized_Files/, PMCID_sentence_files/, PMCID_sentence_files_EXT/, and Concept_Norm_Files/

2. run_span_detection.sh: training for all the span detection algorithms. The algorithm is indicated by algo and can be: CRF, LSTM, LSTM-CRF, char_embeddings, LSTM_ELMO, BIOBERT
	
	a. Inputs: a list of the ontologies, any excluded files, the biotags to use, file paths to the tokenized files and to where to save the models, the algorithm type along with any hyperparameters, whether or not to save the model, and whether to use crf hyperparameters.

	b. Outputs: Span detection model files in Models/SPAN_DETECTION/.

3. fiji_run_open_nmt_full.sh: training for concept normalization algorithm OpenNMT for all ontologies. Ideally these are run on GPUs on a supercomputer (the name of ours is fiji, hence the name). Also all models can be trained independently in parallel for each ontology, and thus there is a file for all ontologies following the format: ONTOLOGY_fiji_run_open_nmt_full.sh. Lastly, all experiments for concept normalization can be run with this script as well.
	
	a. Inputs: one ontology to run, any excluded files, file paths to the tokenized files and the concept normalization files

	b. Outputs: OpenNMT model files for each ontology in Models/CONCEPT_NORMALIZATION/. Also prediction files and summary files in Output_Folders/Concept_Norm_Files/

##### Evaluation Pipeline
1. run_eval_pipeline_1.sh: preprocess all articles to be word tokenized, run span detection models either locally or on a supercomputer (LSTM-ELMo and BioBERT), process the spans detected for concept normalization using OpenNMT
	
	a. Inputs: CRAFT folder path for articles to process, all output folders for the evaluation files, list of ontologies to include, a list of the PMC articles to evaluate, if there exists a gold standard or not, the algorithm type 

	b. Outputs: spans detected in /Output_Folders/Evaluation_Files/Results_span_detection/ and the input for the concept normalization step from the spans detected in /Output_Folders/Evaluation_Files/Concept_Norm_Files/

2. run_eval_open_nmt.sh: Run the concept normalization pipeline (OpenNMT) on a supercomputer for all ontologies. 
	
	a. Inputs: the folder with the spans detected on the character level, the ontologies of interest, the output folder for the concepts

	b. Outputs: concept IDs for the spans detected in /Output_Folders/Evaluation_Files/Results_concept_norm_files/

3. run_eval_pipeline_2.sh: Processes the concept norm files along with the span detection files to create the full concept recognition final output in bionlp format. 
	
	a. Inputs: a list of the ontologies of interest, the results of the concept normalization, the output folder, if there exists a gold standard or not, a list of PMC articles to evaluate, whether or not to perform an evaluation analysis

	b. Outputs: the final output of the full concept recognition path in bionlp format, summary metrics of the full pipeline if there is a gold standard

##### CRAFT Evaluation Pipeline
The exact same process as the Evaluation Pipeline (steps 1-3 as above) except focusing on the 30 held out documents for the CRAFT shared task evaluation. Add "0_craft" to the beginning of each bash filename for this CRAFT evaluation. Also there is one extra step for this pipeline. 

1. 0_craft_run_eval_pipeline_1.sh
2. 0_craft_run_eval_open_nmt.sh
3. 0_craft_run_eval_pipeline_2.sh
4. 0_craft_run_final_output.sh: process the final results of the full concept recognition pipeline into the format for the final CRAFT Shared Task performance analysis via Docker.
	
	a. Inputs: the evaluation path, a list of the ontologies of interest, the full concept recognition output folder, a list of PMC articles to evaluate, the complete final output folder for each span detection algorithm, the span detection algorithm used and the corresponding model name

	b. Outputs: the final output format for the final performance analysis by span detection algorithm for each PMC article


### Models
All of the models for span detection and concept normalization with separate models for each ontology.
##### Span Detection
We explored 6 different methods for span detection. We also provide the optimal hyperparameterization for each algorithm. 

1. Conditional Random Fields (CRF): a discriminative algorithm that utilizes a combination of arbitrary, overlapping, and agglomerative observation features from boththe past and future to predict the output sequence. 
2. Bidirectional Long Short Term Memory (BiLSTM): a special form of a recurrent neural network that by default remembers information for long periods of time, allowing for more distant context to be included in the algorithm.
3. BiLSTM combined with a CRF (BiLSTM-CRF): the architecture of a regular BiLSTM with a CRF as the last layer.
4. BiLSTM with character embeddings (char-embeddings): a BiLSTM with a different underlying sequence representation based on character embedddings. 
5. BiLSTM and Embeddings from a Language Model (BiLSTM-ELMo): a BiLSTM with a new underlying sequence representation from the language model ELMo. ELMo is a language model trained on the 1 Billion Word Benchmark set with representation that are contextual, deep, and character based. 
6. Bidirectional Encoder Representations from Transformers fro Biomedical Text Mining (BioBERT): a biomedical-specific language model pre-trained on biomedical documents from both PubMed abstracts and PubMed Central full-text articles based on the original BERT architecture. 

##### Concept Normalization
Open-source Toolkit for Neural Machine Translation (OpenNMT) implements stacked BiLSTMs with attention models and learns condensed vector representations of characters from the training data, processing one character at a time.

### Output Folders
All output folders for all algorithms:
1. Tokenized Files: All articles by ontology preprocessed into BIO- Format. 
2. PMCID_files_sentences and PMCID_files_sentences_EXT: All articles by sentence with sentence information and concept annotation information for both the 10 ontologies without and with extensions (EXT) respectively. 
3. Concept_Norm_Files: All files related to the concept normalization process including the preprocessed files to the prediction files for each ontology with and without extensions. There are 5 different folders within each ontology corresponding to the different runs of OpenNMT:

	a. full_files = the token level where all mappings of spans to concepts, even though some are the same string and concept, only occurring in different places in the text. This is the original run used for all training algorithms. Token-ids in the paper.

	b. no_duplicates = the type level where there is one mapping of a given span to a concept regardless of frequency. Type-ids in the paper.

	c. random_ids = an experimental run replacing the concept IDs with random numbers of the same length, drawn without replacement. Random-ids in the paper. 

	d. shuffled_ids = an experimental run that scrambles the relationship between concept and ID. Shuffled-ids in the paper. 

	e. alphabetical = an experimental run that alphabetizes the concepts by label and assigns consecutive IDs. Alphabetical-ids in the paper.

4. Evaluation_Files: All of the folders and files for the evaluation pipeline. 
	
	a. Articles = the gold standard articles to perform the full concept recognition pipeline on.

	b. Tokenized_Files = the gold standard articles preprocessed into BIO- format as well as BIOBERT format (in the BIOBERT folder)

	c. PMCID_files_sentences and PMCID_files_sentences_EXT = the sentence information for the gold standard articles for both without and with extensions (EXT) respectively.

	d. Results_span_detection = the spans identified from running the span detection models in BIO- format of the gold standard articles.

	e. Concept_Norm_Files = the preprocessed span detection files adding the concept ID information for the concept normalization step by ontology with and without extensions.

	f. Results_concept_norm_files = the resulting prediction files from concept normalization in the format of OpenNMT.

	g. concept_system_output = the final output of the full concept recognition run in the bionlp format by span detection model for all ontologies with and without extensions.

5. concept_system_output: The final full system output by MODEL (CRF, LSTM, LSTM_CRF, CHAR_EMBEDDINGS, LSTEM_ELMO, and BIOBERT) and then by ontology with and without extensions. The summary results files are also included: Full_results_all_models.xlsx and Full_results_all_models_EXT.xlsx, as well as results for each span detection model type.

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
