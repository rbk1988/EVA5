# EVA5

## Assignment for Session 4:

MNIST Classifier: 

1. Refer to this code: [COLABLINK](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx) .

2. WRITE IT AGAIN SUCH THAT IT ACHIEVES

   1. 99.4% validation accuracy
   2. Less than 20k Parameters
   3. You can use anything from above you want. 
   4. Less than 20 Epochs
   5. No fully connected layer
   6. To learn how to add different things we covered in this session, you can refer to this code: [https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99 ](https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99)DONT COPY ARCHITECTURE, JUST LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC.

   ## **Solution:**

   ### S4-Assignment-Solution.ipynb:

   **Net1:**         

   ​         Total params: 17,194

   ​         Last epoch test accuracy : 99.3500%

   ​         Highest test accuracy : 99.4100% at epoch 13  

   ​         Number of times test accuracy exceeded 99.4% is 1.

   

   **Net2:**

   ​         Total params: 17,194

   ​         Last epoch test accuracy : 99.3700%

   ​         Highest test accuracy : 99.4400% at epoch 16

   ​        Number of times test accuracy exceeded 99.4% is 3.

   

   P.S : Both the above networks vary in the position  and value of dropout layers.

   

   

