# loss-visualization
This project is to visualize the loss landscape of neural networks. This project uses a CAFFE saved model for visualizing the loss landscape. The project is an experimental implementation of the paper https://arxiv.org/abs/1712.09913

Steps for reproducing the experiment:
1. Install CAFFE and make pycaffe according to the link http://caffe.berkeleyvision.org/installation.html
2. Verify pycaffe is working by importing caffe in Python
3. Download the cifar10 dataset using the caffe script create_cifar10.sh from CAFFE_ROOT 
4. Select a pretrained model to run(Quick_train, Full_train, Sigmoid) by directory path in the code.
5. Set the DB Path to the path of the CIFAR_TEST_LMDB in the code
6. With the working directory as CAFFE_HOME execute the file loss_visualization.py
7. The code visualizes the loss surface and writes the different files to disk in the selected model folder.
8. Visualize the csv file using Python or Octave.
