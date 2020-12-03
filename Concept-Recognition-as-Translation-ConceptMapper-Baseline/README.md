# Concept Recognition as Translation ConceptMapper Baseline

## Requirements
* Docker
* Git

## How to reproduce

 1. Clone this repository
 
```bash
> git clone https://github.com/UCDenver-ccp/Concept-Recognition-as-Translation-ConceptMapper-Baseline.git ./baseline-repo.git
```

 2. Build the Docker container
Execute the following from inside the `./baseline-repo.git` directory created while cloning the project. `[BASELINE-REPO]` represents the absolute path to this directory in the commands below.

```bash
> cd [BASELINE-REPO]
> docker build -t cm-baseline:0.1 .
```

 3. Generate the baseline files
 
 ```bash
 docker run --rm -v [BASELINE-REPO]/data:/home/baseline/data cm-baseline:0.1 
 ```
 
   Alternatively, if you want to cache the ConceptMapper directories to the host machine, run the following where `[DICTIONARY-DIR]` is the absolute path to a directory on the host machine where the ConceptMapper dictionaries will be stored:
 
   ```bash
 docker run --rm -v [BASELINE-REPO]/data:/home/baseline/data -v [DICTIONARY-DIR]:/home/baseline/dictionaries cm-baseline:0.1 
   ```
 
