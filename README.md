# Deep Learning for Images
In this repository, I will implement the different convolutional neural networks by PyTorch. Reproduce the original model and then play around by using different datasets. Furthermore, we can use a pre-trained model and implement transfer learning.  
This repository will follow Prof Mu's [Introduce to Deep Learning](https://courses.d2l.ai/).
   
# LeNet
LeNet is a simple convolutional neural network. It used 2 convolutional layers, 2 pooling layers and 3 linear layer to deal with classification problem. This model works well on the MNIST dataset.  
![LeNet Structure](https://github.com/ZhipengHong0123/DL_image/blob/main/pictures/LeNet_structure.png "LeNet Structure")  
  
The Structure of LeNet:
- Input: (1 ,28 ,28) 
- Convolutional layer
- Activation layer
- Pooling layer
- Convolutional layer
- Activation layer
- Pooling layer
- Linear layer(16* 5* 5, 120)
- Activation layer
- Linear layer(120, 84)
- Activation layer
- Linear layer(84, 10)

### Reproducing
Using the same dataset and reproducing a similar model, it's easy to get a good result without spending lots of time in hyperparameter tuning. In the [notebook](https://github.com/ZhipengHong0123/DL_image/blob/main/LeNet/LeNet.ipynb), I train the model with two different dataset: MNIST and FashionMNIST. The LeNet predict well on both image classification task. The training loss and test loss decrease fast when using SGD optimizer. Without training lots of epochs. The model gets high accuracy in the test set.  
  
These are pictures of LeNet prediction by different datasets.
  
![LeNet MNIST](https://github.com/ZhipengHong0123/DL_image/blob/main/pictures/LeNet_MNIST.png)
![LeNet FasionMNIST](https://github.com/ZhipengHong0123/DL_image/blob/main/pictures/LeNet_FasionMNIST.png)


# AlexNet
# VGG
# ResNet
