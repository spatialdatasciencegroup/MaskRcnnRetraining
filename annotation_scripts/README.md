# Annotation Scripts

This folder contains scripts to generate the correct annotations for a given dataset. For information on the format that the annotations should be in, see [images/README.md](../images/README.md).

## labelBox_to_vgg.py

This script converts annotations created using labelBox into the appropriate format used in this project.

It takes two arguments; each argument is a path to a file:

1. **-i** or **--ifile**: The input annotations file... this should be the file created by labelBox
2. **-o** or **--ofile**: The output annotations file... this should be the new annotations file that will be used in the project.

An example flow of using this script would be to...
1. Generate annotations for a dataset using labelBox
2. Convert the labelBox annotations to the desired format using this script
3. Copy the new annotations to both the training and validation directories
4. Retrain using MaskRCNN using the new annotations

### Example Usage

`python labelBox_to_vgg.py -i ./old_annotations.json -o ./new_annotations.json`
