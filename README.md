# efficient-transfer-learning-for-binary-image-classification
An efficient transfer learning implementation using the pretrained VGG16 network for the binary classification of the Food-5K dataset. Instead of running multiple (expensive) forward passes of the VGG network, we just do it once and use the output as the input for the binary classifier.
