# German Traffic Sign Classification
This repository provides codes for:
- downloading german traffic sign dataset
- loading and Visualization of a sample of the train dataset
- preprocess for all images before being used for training the model and testing.
- augmentation of dataset using keras ImageDataGenerator for training
- create, build and train a multiclass classification CNN model to recognize german traffic signs
- evaluation of model on the test dataset
- recognition on a test image using url

--------------------------------------
Dataset used for training, validation and testing
Link: https://bitbucket.org/jadslim/german-traffic-signs

- Dataset consists of 43 Traffic Signs. All the images are RGB images of (32,32) pixel resolution.
- Training Images: 34799
- Test Images: 12630
- Validation Images: 4410

Visualization of 5 images per traffic sign class
![Traffic Signs Training Samples](signnames_samples.jpg)

Distribution of training image samples over the training dataset
![Distribution of samples over training dataset](Dist_of_sample.jpg)

--------------------------------------
Preprocessing of images is done with
- conversion of image to grayscale image
- histogram equalization of grayscale image to improve contrast of the image
- normalization of pixel value to range of 0 to 1

--------------------------------------
Data augmentation is performed using ImageDataGenerator for increasing the number of training samples

![Sample images obtained using data augmentation technique on training dataset](Preprocessed_samples.jpg)

--------------------------------------
CNN model created for training
- Convolutional layer with 64 filters with kernel size (5,5)
- Relu activation layer
- Convolutional layer with 64 filters with kernel size (5,5)
- Relu activation layer
- Convolutional layer with 32 filters with kernel size (3,3)
- Relu activation layer
- Convolutional layer with 32 filters with kernel size (3,3)
- Relu activation layer
- Fully connected layer with 512 units
- Relu activation layer with dropout rate of 50%
- Adam optimizer with learning rate=0.001 is used for updating the model parameters.

![model summary](model_summary.png)

--------------------------------------
Model is trained for 5 epochs with 2000 iterations per epoch and 50 images per iteration

![Training history](training.png)
![Performance history](Performance_history.png)

Model's performance on test dataset is evaluated with loss: 0.0916 and accuracy: 97.28%

--------------------------------------
Result of recognition using a test image from a url

![Test Recognition](url_image_recognition.png)

