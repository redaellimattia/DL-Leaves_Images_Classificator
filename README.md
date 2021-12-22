# Leaves Images Classificator
## Description of the problem
This small project was performed as an assignment for the course of _Artificial Neural Networks & Deep Learning_ and it's part of the exam. 
The task was to develop a model for Image Classification and to train it to distinguis between 14 classes of leaves, as in the example below.
<p align=center>
  <img width="412" alt="Schermata 2021-11-29 alle 14 37 38" src="https://user-images.githubusercontent.com/79969755/143877722-613956ca-cdd1-4ce9-be17-075853540133.png">
</p>

## Dataset Description
The dataset we used was composed of 17728 images, but training with only those resulted in a poor accuracy on the private test set. We used the ImageDataGenerator class to perform data augmentation. 

## Final Model
The ultimate model we submitted reached an accuracy of 92.08% on the private test set and was obtained using the technique of _Transfer Learning_; the model was then fine-tuned for two times, with a progressively smaller learning rate. Find more accurate description on the development and the specific details of the models on the attached [document](https://github.com/GabrieleRivi/Leaves_Images_Classificator/blob/main/First_homework_AND2L.pdf).

# Group components:
Group Name:<b> Gamma </b>

- [Gabriele Rivi](https://github.com/GabrieleRivi)
- [Mattia Redaelli](https://github.com/redaellimattia)
- [Ariel Ratzonel](https://github.com/ArielRatzonel00)

