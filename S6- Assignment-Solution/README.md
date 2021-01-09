# S6- Assignment Solution

1. **Your assignment 6** is to take your best 5th code, and run below versions for 25 epochs and report findings:
   1. with L1 + BN
   2. with L2 + BN
   3. with L1 and L2 with BN
   4. with GBN
   5. with L1 and L2 with GBN
2. **You cannot be running your code 5 times manually (-500 points for that). You need to be smarter and write a single loop or iterator to iterate through these conditions.**
3. draw **ONE** graph to show the validation accuracy curves for all 5 jobs above. This graph must have proper legends and it should be clear what we are looking at.
4. draw **ONE** graph to show the loss change curves for all 5 jobs above. This graph must have proper legends and it should be clear what we are looking at.
5. find any 25 misclassified images (combined into single image) for "with GBN" model. You should be using the saved model from the above jobs. You MUST show the actual and predicted class names.
6. the explanatory README file that explains what is your code all about, your findings, and your single image showing 25 misclassified images.
7. submit the Github link



P.S: I have used the version of Ghost Normalisation used in [Tabnet](https://gist.github.com/monk1337/731c2078203f01e0d486c4a74a470eb9#file-tabnet-py)



## S6-AssignmentSolution_training.ipynb

This notebook has the steps to train models with various combinations of L1,L2 regularizations and Batch and ghost batch norm layers. The model weights are saved in google drive.

A batch size of 4096 and virtual batch size of 32 are used.

Below versions of models are run for 25 epochs :

1. with L1 + BN
2. with L2 + BN
3. with L1 and L2 with BN
4. with GBN
5. with L1 and L2 with GBN

This notebook  outputs Validation Accuracy Change Graph (all 5 models combined) and Loss Change Graph (all 5 models combined).

![](https://github.com/rbk1988/EVA5/blob/main/S6-%20Assignment-Solution/validation_accuracy_plot.png)

![](https://github.com/rbk1988/EVA5/blob/main/S6-%20Assignment-Solution/validation_loss_plot.png)

Here are our findings at the end of 25 epochs:

1. L1+BN performs better than L2+BN.
2. L1+L2+BN performs better than L1+BN.
3. GBN performs better than L1+BN but less than L1+L2+BN .
4. L1+L2+GBN performs better GBN.
5. L1+L2+GBN gave the best performance.

Clearly L1 and L2 regularization helps to improve test accuracy by reducing overfitting.





## S6-AssignmentSolution_misclassified_images.ipynb

This notebook has the logic to find 25 misclassified images and plots them in a single plot along with the actual and predicted labels.