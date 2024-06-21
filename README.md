# Real-time detection of crop rows in maize fields based on autonomous extraction of ROI
It is a crop rows detection method using the object detection model.
1. ⚡Super fast: It takes only 25 ms to process a single image (640*360 pixels) and the frame rate of the video stream exceeds 40FPS.
2. 👍Multiple periods: The model is trained on the dataset including various crop row periods.
3. 🤗High accuracy: The average error angle of the detection lines is 1.88◦, which can meet the accuracy requirements of field navigation.

# Architecture
<div align=center>
<img width="1113" alt="1702287039880" src="https://github.com/WoodratTradeCo/crop-rows-detection/assets/38500652/6f859450-be25-4491-8f4a-8ff0b7e41cee">
</div>

# Labeled images
<div align=center>
<img width="1040" alt="1702287116895" src="https://github.com/WoodratTradeCo/crop-rows-detection/assets/38500652/4e3a9b4c-a0f1-4df3-9d84-5398b92d3b17">
</div>

# Image processing

<div align=center>
<img width="1000" alt="1702292437924" src="https://github.com/WoodratTradeCo/crop-rows-detection/assets/38500652/15d2796c-7b3b-4374-b820-28833f42abc0">
</div>

# Results

<div align=center>
  
https://github.com/WoodratTradeCo/crop-rows-detection/assets/38500652/b091a076-273c-48ff-a5ab-d73d31c4d6f2

</div>

<div align=center>
  
https://github.com/WoodratTradeCo/crop-rows-detection/assets/38500652/53440847-a97e-406d-b26f-96e7a5d99cd9

</div>

# Usage (How to test our model)
Thanks to the contribution, the code is based on https://github.com/ultralytics/yolov5. 

    # 1. Download the trained weights and training log files.
    The trained model is uploaded on https://drive.google.com/file/d/1uca8t8SYReriOtuzo5_RZsCJqb2ggmte/view?usp=sharing. Model and training logs can be obtained after unzipping.
    # 2. Install the requirment.txt
    # 3. Run detect.py. 
    We have prepared 5 videos for testing, the root is test_video/*.mp4(avi). Images format has not been accepted yet.
    # 4. Change the test video.
    If you want to change the test video, you have to revise line 300 in detect.py


# NOTE:
1. We shared the part of our datasets. In this project, we trained 1500 images. It is sorry that we cannot share all the datasets. But you can still check some typical images in the folder "mydata/images/train". The training log is shown in "runs/train/exp1".
2. It is noted that we just provide a solution for crop row detection if you want to run the code in your own data. We strongly suggest you to make some datasets to train your own data to ensure the performance of the model.

If you find this code useful to your research, please cite our paper in the following BibTeX:

    @article{yang2023real,
      title={Real-time detection of crop rows in maize fields based on autonomous extraction of ROI},
      author={Yang, Yang and Zhou, Yang and Yue, Xuan and Zhang, Gang and Wen, Xing and Ma, Biao and Xu, Liangyuan and Chen, Liqing},
      journal={Expert Systems with Applications},
      volume={213},
      pages={118826},
      year={2023},
      publisher={Elsevier}
    }


