# EVA5

## Assignment for Session 5:

1. Assignment:

   1. Your new target is:
      1. 99.4% **(this must be consistently shown in your last few epochs, and not a one-time achievement)**
      2. Less than or equal to 15 Epochs
      3. Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
   2. Do this in exactly 4 steps
   3. Each File must have "target, result, analysis" TEXT block (either at the start or the end)
   4. You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 
   5. Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted. 
   6. Explain your 4 steps using these **target**, **results**, and **analysis** with **links** to your GitHub files (Colab files moved to GitHub). 
   7. Keep Receptive field calculations handy for each of your models. 
   8. If your GitHub folder structure or file_names are messy, -100. 
   9. When ready, attempt S5-Assignment Solution

2. 

   ## **S5_AssignmentSolutionStep1.ipynb:**

   **Target:**  

   * Get the basic structure in place.
   *  Just conv layers. No batch norm,dropouts or image augmentations.
   * [link](S5_AssignmentSolutionStep1.ipynb)

   **Result:** 

   * 6.37 M parameters.
   *  Best train accuracy = 99.92% at 13th epoch.
   * Best test accuracy = 99.32% at 13th epoch.

   **Analysis:**

   * The number of parameters is well above the required 10k parameters.
   * The training accuracy has been above 99.4 % consistently since epoch 4, but test accuracy has not crossed 99.4 % by the end of the 14th epoch.

   

   ## **S5_AssignmentSolutionStep2.ipynb:**

**Target:**  

* Use the structure from Step1 as the reference, and reduce the number of parameters below 10k.
*  Batch norm layers were added.
* Use GAP instead of conv layer towards the output , reduce the number of parameters.
* [link](S5_AssignmentSolutionStep2.ipynb)

**Result:** 

* 7.758 K parameters.
* Best train accuracy = 99.47% at 13th epoch.
* Best test accuracy = 99.24% at 14th epoch.

**Analysis:**

* The number of parameters is below the required 10k parameters.
* The training accuracy has been above 99.4 % consistently for the last 3 epochs, but test accuracy has not crossed 99.4 % by the end of the 14th epoch.





## **S5_AssignmentSolutionStep3.ipynb:**

**Target:**  

* Use the model from Step2.
*  Reduce the batch size from 128 to 64.
* Use RandomRotation for image augmentation with -7.0 and 7.0 degrees.
* [link](S5_AssignmentSolutionStep3.ipynb)

**Result:** 

* 7.758 K parameters. 
*  Best train accuracy = 99.25% at 13th and 14th epochs. Training accuracy is lower than the previous model due to the image augmentation used. 
*  Best test accuracy = 99.42% at 13th and 14th epochs.

**Analysis:**

* The reducing the batch size might have given a regularizing effect.
*  The image augmentation helped to increase the test accuracy.
*  The test accuracy has crossed 99.4 % by the end of the 13th epoch.



## **S5_AssignmentSolutionStep4.ipynb:**

**Target:**  

* Use the model from Step3.  
* Check if learning rate scheduling can help to improve test accuracy.
* [link](S5_AssignmentSolutionStep4.ipynb)

**Result:** 

* 7.758 K parameters. 
*  Best train accuracy = 99.26% at 13th epoch. Training accuracy is lower than the previous model due to the image augmentation used. 
*  Best test accuracy = 99.47% at 13th and 14th epochs.

**Analysis:**

* Learning rate scheduling helped to improve the test accuracy.
* The test accuracy has been greater than or equal to 99.4 % in the last 4 epochs.