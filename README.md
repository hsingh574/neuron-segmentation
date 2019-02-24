# neuron-segmentation

An automated, end-to-end, network for neuron segmentation, trained on fruit fly and mouse EM datasets. 

Note that the .train.py.swp should not be there. 

Training was for 10 epochs (using the train.py script) and the NeuronSegNet model which is somehting akin to the framework proposed by: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

The maxmimum validation accuracy over the 10 epochs was 90% while training on only fruit fly data, and 83% training on both mouse and fruit fly data. 

The predict.py allows passing in of an image path and an output of segmented prediction. 

The data_loader.py contains the helper functions needed for dataset processing. 


Sample Command to run training: 

python train.py \
 --save_weights_path="/home/hsuri/Datathon/final_weights.h5" \
 --train_images="/home/hsuri/Datathon/fruit_fly_volumes.npz" \
 --train_images2 ="/home/hsuri/Datathon/mouse_volumes.npz" \
 --n_classes=1 \
 --input_height=1248 \
 --input_width=1248 \
 --mouse_height = 384 \
 --mouse_width = 384 \
 --epochs = 10 \
 --batch_size = 1 \
 --checkpoints = "/home/hsuri/Datathon/weights/seg_"
