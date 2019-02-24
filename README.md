# neuron-segmentation

An automated, end-to-end, network for neuron segmentation, trained on fruit fly and mouse EM datasets. 

Note that the .train.py.swp should not be there. 

Training was for 10 epochs (using the train.py script) and the NeuronSegNet model which is somehting akin to the framework proposed by: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

The maxmimum validation accuracy over the 10 epochs was 90% while training on only fruit fly data, and 83% training on both mouse and fruit fly data. 

The predict.py allows passing in of an image path and an output of segmented prediction. 

The data_loader.py contains the helper functions needed for dataset processing. 
