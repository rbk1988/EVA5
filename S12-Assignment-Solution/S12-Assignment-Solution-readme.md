# S12 - Assignment Solution

1. Assignment A:
   1. Download this [TINY IMAGENET ](http://cs231n.stanford.edu/tiny-imagenet-200.zip)dataset. 
   2. Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 
   3. Submit Results. Of course, you are using your own package for everything. You can look at [this ](https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb) for reference. 
2. Assignment B:
   1. Download 50 (min) images each of people wearing hardhat, vest, mask and boots. 
      1. Use these labels (same spelling and small letters):
         1. hardhat
         2. vest
         3. mask
         4. boots
   2. Use [this ](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html)to annotate bounding boxes around the hardhat, vest, mask and boots.
   3. Download JSON file. 
   4. Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work). 
   5. Refer to this [tutorial ](https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203). Find out the best total numbers of clusters. Upload link to your Colab File uploaded to GitHub. 

 

Questions in S12-Assignment-Solution:

1. What is your final accuracy?
2. Share the Github link to your ResNet-Tiny-ImageNet code. All the logs must be visible. 
3. Describe the contents of the JSON file in detail. You need to explain each element in detail. 
4. Share the link to your Github file where you have calculated the best K clusters for your 50 dog dataset. 
5. Share the link to your hardhat, vest, mask and boots Images Folder on GitHub
6. Share the link to your JSON file on GitHub



## Assignment A

1. The tiny image net training data was split into two sets 70/30 to get datasets for train  and validation.

2. Resnet18 model was used.

3. After training for 20 epochs the model got a validation accuracy of 54.98%

4. Training logs:

   EPOCH: 1

   Train set: Average loss: 0.2926, Accuracy: 3587/70000 (5.12%) 

   Test set: Average loss: 0.2650, Accuracy: 3061/30000 (10.20%) 

   EPOCH: 2 Train set: Average loss: 0.2452, Accuracy: 9727/70000 (13.90%) 

   Test set: Average loss: 0.2279, Accuracy: 5671/30000 (18.90%) 

   EPOCH: 3 

   Train set: Average loss: 0.2156, Accuracy: 15037/70000 (21.48%) 

   Test set: Average loss: 0.1953, Accuracy: 8282/30000 (27.61%) 

   EPOCH: 4 

   Train set: Average loss: 0.1932, Accuracy: 19649/70000 (28.07%) 

   Test set: Average loss: 0.1844, Accuracy: 9533/30000 (31.78%) 

   EPOCH: 5 

   Train set: Average loss: 0.1767, Accuracy: 23164/70000 (33.09%) 

   Test set: Average loss: 0.1718, Accuracy: 10688/30000 (35.63%) 

   EPOCH: 6 

   Train set: Average loss: 0.1620, Accuracy: 26548/70000 (37.93%) 

   Test set: Average loss: 0.1585, Accuracy: 11904/30000 (39.68%) 

   EPOCH: 7 

   Train set: Average loss: 0.1499, Accuracy: 29295/70000 (41.85%)

   Test set: Average loss: 0.1517, Accuracy: 12612/30000 (42.04%) 

   EPOCH: 8 

   Train set: Average loss: 0.1393, Accuracy: 31663/70000 (45.23%) 

   Test set: Average loss: 0.1438, Accuracy: 13481/30000 (44.94%) 

   EPOCH: 9 

   Train set: Average loss: 0.1299, Accuracy: 34053/70000 (48.65%) 

   Test set: Average loss: 0.1410, Accuracy: 13965/30000 (46.55%) 

   EPOCH: 10 

   Train set: Average loss: 0.1212, Accuracy: 35982/70000 (51.40%) 

   Test set: Average loss: 0.1415, Accuracy: 14158/30000 (47.19%) 

   Saving the model for at /content/gdrive/MyDrive/EVA5/S12AssignmentSolution/tinyimagenet_10_epoch_10.pth. 

   EPOCH: 11 

   Train set: Average loss: 0.1128, Accuracy: 38124/70000 (54.46%) 

   Test set: Average loss: 0.1370, Accuracy: 14623/30000 (48.74%) 

   EPOCH: 12 

   Train set: Average loss: 0.1048, Accuracy: 39923/70000 (57.03%) 

   Test set: Average loss: 0.1353, Accuracy: 14813/30000 (49.38%) 

   EPOCH: 13 

   Train set: Average loss: 0.0970, Accuracy: 41903/70000 (59.86%) 

   Test set: Average loss: 0.1349, Accuracy: 14934/30000 (49.78%) 

   EPOCH: 14 

   Train set: Average loss: 0.0899, Accuracy: 43658/70000 (62.37%) 

   Test set: Average loss: 0.1360, Accuracy: 15216/30000 (50.72%) 

   EPOCH: 15 

   Train set: Average loss: 0.0831, Accuracy: 45483/70000 (64.98%) 

   Test set: Average loss: 0.1363, Accuracy: 15241/30000 (50.80%) 

   EPOCH: 16 

   Train set: Average loss: 0.0767, Accuracy: 47003/70000 (67.15%) 

   Test set: Average loss: 0.1398, Accuracy: 15283/30000 (50.94%) 

   EPOCH: 17 

   Train set: Average loss: 0.0697, Accuracy: 48742/70000 (69.63%) 

   Test set: Average loss: 0.1398, Accuracy: 15424/30000 (51.41%) 

   EPOCH: 18 

   Train set: Average loss: 0.0634, Accuracy: 50462/70000 (72.09%) 

   Test set: Average loss: 0.1423, Accuracy: 15338/30000 (51.13%) 

   EPOCH: 19 

   Train set: Average loss: 0.0575, Accuracy: 52012/70000 (74.30%) 

   Test set: Average loss: 0.1453, Accuracy: 15237/30000 (50.79%) 

   Epoch    19: reducing learning rate of group 0 to 1.0000e-03. 

   EPOCH: 20 

   Train set: Average loss: 0.0340, Accuracy: 59800/70000 (85.43%) 

   Test set: Average loss: 0.1331, Accuracy: 16494/30000 (54.98%)



## Assignment B

- Bounding annotations are prepared using VGG Image Annotator.
- A total of 122 images were used
- The classes used are hardhat, vest, mask and boots.
- K means clustering was performed on the scaled values of height and width of the bounding boxes to find the number of anchor boxes required.
- The optimal value of K was found to be 4 using Silhouette score method.

## VGG Image Annotation Tool Output Json:

1. The keys of the json are the names of the image files followed by a numerical id. This field has the following attributes:
   1. filename: This is the name of the image file.
   2. size : Size of the file in kB.
   3. regions : This is a list of regions annotated: Each element here have two attributes:
      1. shape_attributes
         1. name :  The name of the bounding box shape.
         2. x : x coordinate of the centroid of the bounding box.
         3. y : y coordinate of the centroid  of the bounding box.
         4. width : width of the bounding box.
         5. height : height of the bounding box.
      2. region_attributes
         1. name : This is the class label.
         2. type
         3. image_quality
            1. good
            2. frontal
            3. good_illumination
   4. file_attributes
      1. caption
      2. public_domain
      3. image_url

