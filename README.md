# Brain-Tumor-Detection
Building a detection model using a convolutional neural network in Tensorflow & Keras.
Used a brain MRI images data founded on Kaggle. You can find it [here](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) .

### About the data:
The dataset contains 2 folders: yes and no which contains 253 Brain MRI Images. The folder yes contains 155 Brain MRI Images that are tumorous and the folder no contains 98 Brain MRI Images that are non-tumorous.

## Getting Started
## Data Augmentation:
### Why did I use data augmentation?
Since this is a small dataset, There wasn't enough examples to train the neural network. Also, data augmentation was useful in taclking the data imbalance issue in the data.

Further explanations are found in the Data Augmentation notebook.

Before data augmentation, the dataset consisted of:
155 positive and 98 negative examples, resulting in 253 example images.

After data augmentation, now the dataset consists of:
1085 positive and 980 examples, resulting in 2065 example images.

Note: these 2065 examples contains also the 253 original images. They are found in folder named 'augmented data'.

## Data Preprocessing
For every image, the following preprocessing steps were applied:

* Crop the part of the image that contains only the brain (which is the most important part of the image).
* Resize the image to have a shape of (240, 240, 3)=(image_width, image_height, number of channels): because images in the dataset come in different sizes. So, all images should have the same shape to feed it as an input to the neural network.
* Apply normalization: to scale pixel values to the range 0-1.

## Data Split:
The data was split in the following way:

* 70% of the data for training.
* 15% of the data for validation.
* 15% of the data for testing.

## Neural Network Architecture
This is the architecture that I've built:
![image](https://user-images.githubusercontent.com/84140559/120081212-1608a180-c0da-11eb-866f-bbfedb61b7b6.png)

#### Understanding the architecture:

Each input x (image) has a shape of (240, 240, 3) and is fed into the neural network. And, it goes through the following layers:

* A Zero Padding layer with a pool size of (2, 2).
* A convolutional layer with 32 filters, with a filter size of (7, 7) and a stride equal to 1.
* A batch normalization layer to normalize pixel values to speed up computation.
* A ReLU activation layer.
* A Max Pooling layer with f=4 and s=4.
* A Max Pooling layer with f=4 and s=4, same as before.
* A flatten layer in order to flatten the 3-dimensional matrix into a one-dimensional vector.
* A Dense (output unit) fully connected layer with one neuron with a sigmoid activation (since this is a binary classification task).

## Why this architecture?
Firstly, I applied transfer learning using a ResNet50 and vgg-16, but these models were too complex to the data size and were overfitting. Of course, you may get good results applying transfer learning with these models using data augmentation. 
So why not try a simpler architecture and train it from scratch. And it worked :)

## Training the model
The model was trained for 24 epochs and these are the loss & accuracy plots:

![image](https://user-images.githubusercontent.com/84140559/120081295-9e874200-c0da-11eb-90a0-7b1b11959919.png)
![image](https://user-images.githubusercontent.com/84140559/120081298-a47d2300-c0da-11eb-941a-90f5420310a1.png)

The best validation accuracy was achieved on the 17th iteration.

## Results

Now, the best model (the one with the best validation accuracy) detects brain tumor with:
89.7% accuracy on the test set.
0.89 f1 score on the test set.
### Performance table for the best model:
|               |Validation set |  Test set |
| ------------- |:-------------:| ---------:|
| Accuracy      | 92%           | 89%       |
| F1 Score      | 0.92          | 0.89      |

## Final Notes
What's in the files?

* The code in the IPython notebooks.
* The weights for all the models. 
* The original data in the folders named 'yes' and 'no'. And, the augmented data in the folder named 'augmented'.
Thank you!
