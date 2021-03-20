# S13 - Assignment Solution

Assignment: 

1. OpenCV Yolo:

   SOURCE

   1. Run this above code on your laptop or Colab. 
   2. Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
   3. Run this image through the code above. 
   4. Upload the link to GitHub implementation of this
   5. Upload the annotated image by YOLO. 

2. Share your NEWLY annotated (same as 12, but annotated using new tool) images with Zoheb by Wednesday at midnight. Take the set back for training on Thursday.

3. Training Custom Dataset on Colab for YoloV3

   1. Refer to this Colab File: [LINK](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)

   2. Refer to this GitHub [Repo](https://github.com/theschoolofai/YoloV3)

   3. Collect a dataset from the last assignment and re-annotate them. Steps are explained in the readme.md file on GitHub.

   4. Once done:

      1. [Download ](https://www.y2mate.com/en19)a very small (~10-30sec) video from youtube which shows your classes. 
      2. Use [ffmpeg ](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence)to extract frames from the video. 
      3. Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
      4. Infer on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
         python detect.py --conf-thres 0.3 --output output_folder_name
      5. Use [ffmpeg ](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence) to convert the files in your output folder to video
      6. Upload the video to YouTube. 

   5. Share the link to your GitHub project with the steps as mentioned above

   6. Share the link to your YouTube video

   7. Share the link of your YouTube video on LinkedIn, Instagram, etc! You have no idea how much you'd love people complimenting you! 

      

   Questions on the submission page asked are:

   1. Upload the link to your YOLOv3OpenCV code on Github. - 100 pts
   2. Upload the link to the image annotated by OpenCV YOLO inference. - 100 pts
   3. Share the link to your GitHub project with the steps as mentioned above (for YoloV3 training on Colab). -1000 pts
   4. Share the link of your YouTube video (your object annotated by your YoloV3 trained model). - 800 pts.