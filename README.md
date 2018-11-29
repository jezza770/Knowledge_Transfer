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
The model weights and confusion matrices for each fold are in the 'results' folder.
The combined accuracy is 73%, far below what was achieved in the paper. 