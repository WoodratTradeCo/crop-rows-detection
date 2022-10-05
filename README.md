Real-time detection of crop rows in maize fields based on autonomous extraction of ROI

It is a crop rows detection algorithm using YOLOv5 object detection model.
If you want to test the code, you need to follow the following steps:
1. Install the requirment.txt
2. Due to the size limitation of Github upload file, the trained model is stored in runs.zip. Model and training log can be obtained after unzipping. You MUST unzip one of the runs.zip to get the model! Otherwise the error will be reported when you run detect.py, because the corresponding trained model cannot be found!!
3. run detect.py. We have prepared 5 videos for testing, the root is test_video/*.mp4(avi). Images format is not accepected!
4. If you want to change the video, you have to rivise the line 300 in detect.py

NOTE:
1. We shared the part of our datasets. In this project, we trained 1500 images. It is sorry that we cannot share all the dataset due to the limitation of github. But you can still check some typical images in folder "mydata/images/train". The traning log is shown in "runs/train/exp1".
3. It is noted that we just provide a solution for crop rows detection, if you want to run the code in your own data. We strongly suggest you to make some datasets to train your own data to ensure the performance of the model, and of course you can also train dataset based on our trained model to increase the generalization ability of the model. 
4. The labeled image and detailed process are shown as follows:
![`3ZRB CTPV~2QXC5T@V}3@J](https://user-images.githubusercontent.com/38500652/169472351-d4743039-015f-4795-a2da-81e757eb460f.png)
![image](https://user-images.githubusercontent.com/38500652/194058780-feec709c-74d1-4eeb-9c3e-0468412b2401.png)
![image](https://user-images.githubusercontent.com/38500652/194058802-02308204-ff79-407e-a73d-c93d3b6816ee.png)
![image](https://user-images.githubusercontent.com/38500652/194058860-7f533038-9001-4cde-9c69-d63bce5aa738.png)

