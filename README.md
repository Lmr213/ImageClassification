# ImageClassification
 
#### Dataset:The images of 5 differnert types of pokemon, the images of each type are under the corresponding folder named by that particular pokemon
#### After data preprocessing, we can load a bacth of images in he training dataset it looks like below(Using visdom to do visualization)

![a batch of training dataset](/Users/lmr/Desktop/Screen Shot 2023-02-06 at 2.42.37 PM.png)

#### Model: ResNet18,  Loss function: CrossEntropyLoss, Optimizer: Adam Optimizer
#### Training the ResNet model and apply validation and test dataset on it we can get the best accuracy on validation set is 0.9098 and the final test accuracy is 0.8841
![a batch of training dataset](/Users/lmr/Desktop/Screen Shot 2023-02-06 at 2.42.37 PM.png)

#### By using transfer learning and fine-tunning the model, we can get the best accuracy on validation set is 0.9571 and the final test accuracy is 0.9614
![a batch of training dataset](/Users/lmr/Desktop/Screen Shot 2023-02-06 at 2.42.37 PM.png)
