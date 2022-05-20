import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import math
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def takeSecond(elem):
    return elem[1]

def to_sublists(lst, length = 2):
    return [lst[i:i+length] for i in range(0,(len(lst)+1-length),2)]

def rgb(img):

    img_1 = np.array(img, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(img_1)
    gray = 2 * g - 1 * b - 1 * r
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, img_2) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    element_1 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
    img_3 = cv2.erode(img_2, element_1, iterations=2)
    img_4 = cv2.dilate(img_3, element_1, iterations=4)

    return img_4


def detect(save_img=False):

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                start = time.time()
                xyxy_list = []
                im1 = im0.copy()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        # f'{names[int(cls)]} {conf:.2f}' you can choose to show the text on labels
                        label = f''
                        # You can choose to show the prediction boxes or not
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    xyxy_list.extend(xyxy)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            def fast(img, x, y, draw=True):
                flag = True
                fast = cv2.FastFeatureDetector_create(30)
                fast.setNonmaxSuppression(False) #NMS
                kp = fast.detect(img, None)
                left = []
                right = []
                kp = cv2.KeyPoint_convert(kp)
                for i in kp:
                    if i[0] > x:
                        right.append(i)
                    elif i[0] < x:
                        left.append(i)
                    else:
                        pass

                left = np.array(left)
                right = np.array(right)

                try:
                    [vx_middle_line, vy_middle_line, x_middle_line, y_middle_line] = cv2.fitLine(left, cv2.DIST_HUBER, 0,
                                                                                                     1e-2, 1e-2)
                    k = vy_middle_line / vx_middle_line
                    if k == 0:
                        k = k + 0.0001
                    b = y_middle_line - k * x_middle_line


                except:
                    pass
                try:
                    [vx_middle_line1, vy_middle_line1, x_middle_line1, y_middle_line1] = cv2.fitLine(right,
                                        cv2.DIST_HUBER, 0, 1e-2, 1e-2)
                    k1 = vy_middle_line1 / vx_middle_line1
                    if k1 == 0:
                        k1 = k1 + 0.0001
                    b1 = y_middle_line1 - k1 * x_middle_line1


                except:
                    pass

                try:
                    x1 = (int(((y - b) / k + (y - b1) / k1) / 2))
                    y1 = y
                    x2 = (int(((360 - b) / k + (360 - b1) / k1) / 2))
                    y2 = h
                    PI = math.pi
                    if (x2 - x1) != 0:
                        k_mid = (y2 - y1) / (x2 - x1)
                    else:
                        k_mid = (y2 - y1) / 0.01
                    ceita = (PI / 2 + math.atan(k_mid)) * (180 / PI)
                    if ceita > 100:
                        ceita = abs(ceita - 180)
                    if ceita > 10:
                        flag = False
                    x11 = int((new_lst[0][0] + xyxy_list_sort[0][0]) / 6)
                    y11 = int(new_lst[0][1] / 3)
                    x22 = int((xyxy_list_sort[-1][0] + new_lst[-1][0]) / 6)
                    y22 = int(xyxy_list_sort[-1][1] / 3)
                    PI = math.pi
                    if (x22 - x11) != 0:
                        k_m = (y22 - y11) / (x22 - x11)
                    else:
                        k_m = (y22 - y11) / 0.01
                    ceita_m = (PI / 2 + math.atan(k_m)) * (180 / PI)
                    if ceita_m > 100:
                        ceita_m = abs(ceita_m - 180)
                    if ceita_m > 10:
                        flag = False
                    if flag == True:
                        cv2.line(im1, (int((y - b) / k), y), (int((360 - b) / k), 360), (0, 0, 255), 3)
                        cv2.line(im1, (int((y - b1) / k1), y), (int((360 - b1) / k1), 360), (0, 0, 255), 3)
                        cv2.line(im1, (int(((y - b) / k + (y - b1) / k1) / 2), y),
                                           (int(((360 - b) / k + (360 - b1) / k1) / 2), 360),
                                           (0, 255, 255), 3)

                        cv2.polylines(im1, [c], True, (255, 0, 0), 3)
                except:
                    pass
                return im0
            # Stream results

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                    h0 = im0.shape[0]
                    w0 = im0.shape[1]

                    im0 = cv2.resize(im0, (640, 360))
                    im1 = cv2.resize(im1, (640, 360))
                    xyxy_list_sort = to_sublists(xyxy_list)

                    new_lst = []
                    for i in range(len(xyxy_list_sort)):
                        if i % 2 == 0:
                            new_lst.append(xyxy_list_sort[i])
                    new_lst.sort(key=takeSecond)
                    for i in range(len(xyxy_list_sort)):
                        for j in xyxy_list_sort:
                            if j in new_lst:
                                xyxy_list_sort.remove(j)
                    xyxy_list_sort.sort(key=takeSecond)

                    c = np.array(
                        [[int(new_lst[0][0] / (h0 / 360)), int(new_lst[0][1]) / (h0 / 360)],
                         [int(xyxy_list_sort[0][0] / (h0 / 360)),
                          int(new_lst[0][1]) / (h0 / 360)],
                         [int(xyxy_list_sort[-1][0] / (w0 / 640)), int(xyxy_list_sort[-1][1] / (w0 / 640))],
                         [int(new_lst[-1][0] / (w0 / 640)), int(xyxy_list_sort[-1][1] / (w0 / 640))]], np.int32)

                    roi_mask = np.zeros((360, 640), dtype=np.uint8)
                    cv2.fillPoly(roi_mask, [c], 255)
                    roi = cv2.bitwise_and(im0, im0, mask=roi_mask)
                    roi = rgb(roi)

                    fast(roi, int((new_lst[0][0] + xyxy_list_sort[0][0]) / (w0 / 320)), int(new_lst[0][1] / (h0 / 360)))


                    im1 = cv2.resize(im1, (640, 360))
                    print('###################')
                    end = time.time()
                    cv2.imshow(str(p), im1)
                    cv2.imshow('roi',roi)

                    if cv2.waitKey(1) == ord('q'):
                        break

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({end - start}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp1/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test_video/5.avi', help='source')  # only the video format is allowed!!
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    #print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
