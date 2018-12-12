# Knowledge_Transfer
Attempting to replicate results from Knowledge Transfer paper. 

Using the following directory structure:


├── {source}

─── ├── audio

─── ─── ├── {sound files}

─── ├── meta

─── ─── ├── esc50.csv



├── {sorted}

─── ├── esc50-{fold}

─── ─── ├── train

─── ─── ─── ├── {classes}

─── ─── ─── ─── ├── {sound files}

─── ─── ├── val

─── ─── ─── ├── {classes}

─── ─── ─── ─── ├── {sound files}

The "sorted" directory structure is created by running prepare_esc50.py with "source" and "sorted" as parameters. 

e.g. python prepare_esc50.py --source {source} --destination {sorted}

Training is carried out using feat_extractor.py with the appropriate parameters. 

e.g. python feat_extractor.py --mode train --testfold 2 --source {source} --sorted {sorted}

This command will do transfer learning from the audioset model to the N1 model in the paper at: http://www.cs.cmu.edu/~alnu/tlwled/tlwled.pdf

Weights are updated for F1 and F2 is replaced with an appropriately sized layer. 

The model weights are in the 'Results' folder as 'trained_model_{}.dat' where the number is the fold used for testing. 

The SVM is trained by saving the F1 and/or F2 layer for each input using 'get_svm_data.py'.

e.g. python get_svm_data.py --model {path to fold-specific pytorch model} --sorted {ESC50 sorted folder} --testfold {fold}

This will output a fold-specific SVM datafile.

To use these together, run test_model.py which will load both the NN and SVM as appropriate. 

e.g. python test_model.py --nn {path to NN model file} --svm {path to SVM model file} --sample {folder with samples sorted by class}

Accuracy is reported in 'Results' folder's README. Best accuracy is 82.25% 