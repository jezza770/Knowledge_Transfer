# Knowledge_Transfer
Attempting to replicate results from Knowledge Transfer paper. 

Using the following directory structure:
.
+-- _{source}
|   +-- audio
|   |   +-- {sound files}
|   +-- meta
|   |   +-- esc50.csv

. 
+-- _{sorted}
|   +-- esc50-{fold}
|   |   +-- train
|   |   |   +-- {classes}
|   |   |   |   +-- {sound files}
|   |   +-- val
|   |   |   +-- {classes}
|   |   |   |   +-- {sound files}

Currently, results are only computed for the first transfer method - weights are updated for F1 and F2 is replaced with an appropriately sized layer. 
The model weights and confusion matrices for each fold are in the 'results' folder.
The combined accuracy is 73%, far below what was achieved in the paper. 