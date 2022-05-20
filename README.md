Real-time detection of crop rows in maize fields based on autonomous extraction of ROI

It is a crop rows detection algorithm.
If you want to test the code, you need to follow the following step:
1. Install the requirment.txt
2. run detect.py. We have prepared 5 videos for testing, the root is test_video/*.avi. 
3. If you want to change the video, you have to rivise the line 300 in detect.py

NOTE:
1. We shared the part of our datasets. In this project, we trained 1600 images. It is sorry that we cannot share all the dataset due to the human cost. But you can still check 300 typical images in folder "mydata/images/train". The validation data is available. 
2. The traning log is shown in "runs/train/exp1".
3. It is noted that we just provide a solution for crop rows detection, if you want to run the code in your own data. We strongly recommend you to make some datasets to train your own data to ensure the performance of the model, and of course you can also train dataset based on our trained model to increase the generalization ability of the model. 
