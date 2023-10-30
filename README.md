# Denoising-Autoencoder-Shallow-Autoencoder

In this project, I will show how to work with dimensionality reduction and learn how to develop a simple Autoencoder. In the first part, I will explore how to develop a simple shallow autoencoder, then I will develop a deep version. Finally, I will experiment with the application of autoencoder on denoising data task (denoising-autoencoder).

# Dataset
I load the CIFAR-10 dataset, available from torchvision.datasets. This dataset is one of the most popular benckmark in the filed of Computer Vision. It consits of  10  different classes, that represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset contains  50,000  training images and  10,000  images for testing.

First, I will pre-process them with a PreProcessing fuction that works in the following way. The images are originally in RGB format, but I will convert them to grayscale for convenience. The value of each pixel is between  0  and  255 , and it represents a point of an image of size  32×32 . I will normalize all values between  0  and  1 , and I will flatten the  32×32  images into vectors of size  1024 . Moreover, since no validation set is defined, I split the train set in a validation set and a new test set. Finally, I will design a custom dataset class, derived from the standard Dataset class, that returns a PyTorch Dataset object, along with its noisy version. The Gaussian noise is tunable with the noise_factor parameter and will be used for the Denoising Autoencoder.

# Dimensionality reduction method

Similar to Principal component analysis (PCA), Singular Value Decomposition (SVD) is a standard linear dimensionality reduction method. They both linearly combine the features of the original high-dimensional dataset and project them into a lower-dimensional space, ideally retaing most of thier intrinsic properties.

# Comparing the results
I used linear and non-linear activation function (ReLU).

Loss: The shallow linear autoencoder achieved a lower training loss (0.00788673) compared to the shallow nonlinear autoencoder with ReLU activation (0.02394094). This indicates that the shallow linear autoencoder was able to reconstruct the input data more accurately during training.

Validation Loss: The shallow linear autoencoder also had a lower validation loss (0.01068598) compared to the shallow nonlinear autoencoder with ReLU activation (0.02381694). This suggests that the shallow linear autoencoder generalizes better to unseen data.

Training Time: The training times for both models are similar, with the shallow linear autoencoder taking 1027.71 seconds and the shallow nonlinear autoencoder taking 1040.40 seconds.

# Shallow Denoising Autoencoder
The model architecture consisted of two encoding layers with 128 and 64 units, respectively, followed by two decoding layers with 64 and 128 units. The activation function used was ReLU. The learning rate was set to 0.001.

For the noise levels tested, the chosen hyperparameter configuration worked well. The model achieved low validation losses and maintained good accuracy in denoising the data. However, it is worth noting that the performance of the model may vary with different noise levels.

The training time for the model was approximately 1026 seconds, indicating that the shallow autoencoder was efficient in training.
