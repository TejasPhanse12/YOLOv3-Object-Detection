import keyboard
import datetime
import cv2
from matplotlib import colors
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'darknet\cfg\yolov3.cfg')

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

#print("Number of objects that can be detected:\n", classes)


def imgdectection(img):

    height, width, _ = img.shape
    cv2.imshow('Sample Image', img)
    cv2.waitKey(0)

    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    for b in blob:
        for n, imgblob in enumerate(b):
            cv2.imshow(str(n), imgblob)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    print("Number of objects being detected:", len(boxes), '\n')
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    print("Number of bexes redundant:", indexes.flatten(), "\n")

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence,
                        (x, y+20), font, 2, (0, 255, 0), 2)
    else:
        window_name = "image"
        img = np.zeros([480, 640, 1])
        cv2.imshow(window_name, img)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)

    cv2.imshow('Sample Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vdodectection(vdo):
    frames = vdo.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(vdo.get(cv2.CAP_PROP_FPS))

    # calculate dusration of the video
    seconds = int(frames / fps)
    video_time = str(datetime.timedelta(seconds=seconds))
    print("Playing time: ", video_time)
    while True:
        _, img = vdo.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(
            img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence,
                            (x, y+20), font, 2, (0, 255, 0), 2)
        else:
            window_name = "image"
            img = np.zeros([480, 640, 1])
            cv2.imshow(window_name, img)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(0)

        #cv2.imshow('Video', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            vdo.realse()
            cv2.destroyAllWindows()


def facecam():
    fcam = cv2.VideoCapture(0)
    while True:
        _, img = fcam.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(
            img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence,
                            (x, y+20), font, 2, (0, 255, 0), 2)

            cv2.imshow('FaceCam', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("press:\ni-image detection\nv-video dectection\nc-face cam\nq-quite")
    while True:
        print("Enter your choice: ")
        if keyboard.read_key() == "v":
            print("Running Video Dectection...")
            svideo = cv2.VideoCapture('samples\\traffic.mp4')
            vdodectection(svideo)
        elif keyboard.read_key() == "i":
            print("Running Image Dectection...")
            simg = cv2.imread('samples\\phone.png')
            imgdectection(simg)
        elif keyboard.read_key() == "c":
            print("Running FaceCam Dectection...")
            facecam()
        elif keyboard.read_key() == "q":
            break
