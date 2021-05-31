# Transformer Neural Machine Translation (NMT)

#TODO: FIX DEPLOY.PY MODEL LOAD
For easier execution of this code, run on google colab (enable GPU):

https://colab.research.google.com/github/ldselvera/Transformer-NMT/blob/main/Transformer_NMT.ipynb


1. IMPORTANT: The model can translate from english to german or english to german. 
To make it easier for the user, I have a variable called "trans". 
The program will know what is the translation to use according to this variable:
trans = "en_de" is for english to german translation (default)
trans = "de_en" for german to english translation
When running "Sample Runs", only run the ones according to your current translation. 
DO NOT run "Sample Runs: German to English" if all previous cells were executed with trans = "en_de" 
DO NOT run "Sample Runs: English to German" if all previous cells were executed with trans = "de_en"
To run "Sample Runs: German to English" you must first change trans variable to trans = "de_en", and the re-execute all cells.

2. IMPORTANT: If you want to use trained models, make sure the model path is modified in the notebook.
2 models are provided (in the previous Google Drive link), the "en_de" is for english to german and "de_en" for german to english translation.

3. IMPORTANT: For faster training change "num_epoch" variable to 5 (2 minutes training) or 10 (4 minutes training).
Trained models provided were trained on 100 epochs (1 hour training) therefore they provide better results.

