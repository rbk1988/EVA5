# S10-Assignment-Solution

Assignment: 

1. Pick your last code
2. Make sure to Add CutOut to your code. It should come from your transformations (albumentations)
3. Use this repo: https://github.com/davidtvs/pytorch-lr-finder
   1. Move LR Finder code to your modules
   2. Implement LR Finder (for SGD, not for ADAM)
   3. Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
4. Find best LR to train your model
5. Use SDG with Momentum
6. Train for 50 Epochs. 
7. Show Training and Test Accuracy curves
8. Target 88% Accuracy.
9. Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
10. Submit

 

S10-Assignment-Solution Questions:

1. Paste your S10 Assignment's GitHub Link - 500PTS
2. Paste the link or upload Training and Test Curves (there should only be 1 graph)- 100PTS
3. What is the test accuracy of your model? - 150PTS (If you have mentioned training accuracies, please comment on your assignment what is test)
4. Share the link or upload an image of 25 misclassified images with GradCam results on top of them- 250PTS

## Solution:

- LRFinder.py : has the logic to implement learning rate finder. The optimal learning rtae was found to be 8.42E-02.
- ReduceLROnPlateau is used while training the model
- The model is trained on CIFAR10 dataset for 50 epochs.
- For the 50th epoch training accuracy was 98.37% and validation accuracy was 91.45%.
- 25 misclassified images with gradcam outputs
- ![](C:\Users\rkamathb\Desktop\My work\EVA5\github-folder\EVA5\S10-Assignment-Solution\25-misclassified-images-gradcam.png)
- 

