# S15 - Assignment Solution

Assignment 15A

1. Look at this model: [https://github.com/intel-isl/MiDaS (Links to an external site.)](https://github.com/intel-isl/MiDaS)
2. Look at this model: https://github.com/NVlabs/planercnn
3. Now you have your helmet, mask, PPE, and boots dataset as well
4. Take your dataset and run it through Midas and get depth images.
5. Take your dataset and run it through the PlanerCNN model and get planer images (you'll not be using the depth images from PlanerCNN, so don't store them). 
6. Now your dataset contains depth map, surface planes, and bounding boxes for the classes
7. Upload to your google drive with a shareable link to everyone, and add a GitHub repo that describes the dataset properly.

Generating depth images using MiDaS : [S15_Assignment15A_MiDaS.ipynb ](S15_Assignment15A_MiDaS.ipynb) :

​	Follow the steps mentioned in the above colab notebook to get depth predictions of the custom dataset.

​    Zip the predictions(midas-output.zip) and store it in Google drive. 

​    The ouput consists of .png image as well .pfm files.

Generating planar images using PlaneRCNN : [S15_Assignment15A_PlanarCNN_v3.ipynb ](S15_Assignment15A_PlanarCNN_v3.ipynb) :

​		Follow the steps mentioned in the above colab notebook to get depth predictions of the custom dataset.

Zip the predictions (planercnn-inference.zip) and store in Google drive.

The folder structure is planercnn-inference/test/inference.

The output consists of the 9 files for every input file: depth files, plane masks and segmentation files.



The outputs are here : https://drive.google.com/drive/folders/1wgoPwoP0jQ5PmxWZrodpayf03otarUIv?usp=sharing







