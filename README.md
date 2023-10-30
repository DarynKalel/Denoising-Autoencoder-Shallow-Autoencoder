# Denoising-Autoencoder-Shallow-Autoencoder

In this project, I will show how to work with dimensionality reduction and learn how to develop a simple Autoencoder. In the first part, I will explore how to develop a simple shallow autoencoder, then I will develop a deep version. Finally, I will experiment with the application of autoencoder on denoising data task (denoising-autoencoder).

# Dataset
I load the CIFAR-10 dataset, available from torchvision.datasets. This dataset is one of the most popular benckmark in the filed of Computer Vision. It consits of  10  different classes, that represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset contains  50,000  training images and  10,000  images for testing.

First, I will pre-process them with a PreProcessing fuction that works in the following way. The images are originally in RGB format, but I will convert them to grayscale for convenience. The value of each pixel is between  0  and  255 , and it represents a point of an image of size  32×32 . I will normalize all values between  0  and  1 , and I will flatten the  32×32  images into vectors of size  1024 . Moreover, since no validation set is defined, I split the train set in a validation set and a new test set. Finally, I will design a custom dataset class, derived from the standard Dataset class, that returns a PyTorch Dataset object, along with its noisy version. The Gaussian noise is tunable with the noise_factor parameter and will be used for the Denoising Autoencoder.

# Dimensionality reduction method

Similar to Principal component analysis (PCA), Singular Value Decomposition (SVD) is a standard linear dimensionality reduction method. They both linearly combine the features of the original high-dimensional dataset and project them into a lower-dimensional space, ideally retaing most of thier intrinsic properties.

We will focus our attention on SVD decomposition and its performances. Given a matrix  X , the SVD decomposes it into the product of two unitary matrices,  V  and  U , and a rectangular diagonal matrix of singular values  S :

X=V⋅S⋅UT. 

The SVD is already implemented in PyTorch as torch.linalg.svd. In our case, the  X  matrix will represent the training set, where each row is a sample (therefore the number of columns will be the number of input features). However, notice that the  X  matrix has a huge number of rows (we have 50,000 input samples) and only 784 columns. If you are using the Colab free plan, the quantity of available RAM may not be sufficient to compute the SVD of  X . Therefore, to ease memory consumption and numerical stability, we resort to one property of the SVD and compute its equivalent version from the matrix  C=XT⋅X , that can be decomposed as:

C=U⋅S2⋅UT 

Since we need just the matrix  U  to compute the compressed version of our data, this trick turns out to be a quick and good solution.

