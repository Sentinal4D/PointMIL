# ATLAS-1 split

`split0.csv` is the single train/val/test split used for every ATLAS-1 result in the paper
(the supplementary material describes 5-fold cross-validation; the runs used this one split,
and selected checkpoints on its test part).

Cells: 1500 in total, 1032 train, 160 val and 308 test. One cell of the original split has no
public serial number and is not part of the release.

The public serials Blebbistatin_Plate3_037 and Blebbistatin_Plate3_224 are the same cell
released twice under two names; both are kept in the training split.
