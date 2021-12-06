from numpy.core.shape_base import block
from config import *
from human_pose_estimation import extract_key_points_from_image, create_blackie_from_image
import cv2
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

mpPose = mp.solutions.pose


def code_image(img):
    # Extract key points
    key_points = extract_key_points_from_image(img)
    if len(key_points) == 0:
        return np.array([])
    aux = []
    for point in key_points:
        x = point[0]
        y = point[1]
        aux.append(np.array([x,y]))
    key_points = np.array(aux)

    # Get coordinates of key points
    xs = []
    ys = []
    for point in key_points:
        xs.append(point[0])
        ys.append(point[1])

    # Calculate horizontality
    max_horizontal_distance = max(xs)-min(xs)
    max_vertical_distance = max(ys)-min(ys)
    horizontality = max_horizontal_distance/max_vertical_distance

    # Calculate hands distance - feet distance ratio
    hands_distance = np.linalg.norm(key_points[mpPose.PoseLandmark.LEFT_WRIST]-key_points[mpPose.PoseLandmark.RIGHT_WRIST])
    feet_distance = np.linalg.norm(key_points[mpPose.PoseLandmark.LEFT_HEEL]-key_points[mpPose.PoseLandmark.RIGHT_HEEL])
    hands_feet_ratio = hands_distance/feet_distance

    # Check if the hands are under the body (origin is in the upper-left corner)
    hands_under_body = False
    if ys[mpPose.PoseLandmark.LEFT_WRIST] > ys[mpPose.PoseLandmark.RIGHT_HEEL] and \
       ys[mpPose.PoseLandmark.LEFT_WRIST] > ys[mpPose.PoseLandmark.LEFT_HEEL] and \
       ys[mpPose.PoseLandmark.RIGHT_WRIST] > ys[mpPose.PoseLandmark.RIGHT_HEEL] and \
       ys[mpPose.PoseLandmark.RIGHT_WRIST] > ys[mpPose.PoseLandmark.LEFT_HEEL]:
        hands_under_body = True

    # Check if the hips are above the hands and the feet (origin is in the upper-left corner)
    hips_above_hands_and_foot = False
    if ys[mpPose.PoseLandmark.LEFT_WRIST] > ys[mpPose.PoseLandmark.LEFT_HIP] and \
       ys[mpPose.PoseLandmark.RIGHT_WRIST] > ys[mpPose.PoseLandmark.RIGHT_HIP] and \
       ys[mpPose.PoseLandmark.LEFT_HEEL] > ys[mpPose.PoseLandmark.LEFT_HIP] and \
       ys[mpPose.PoseLandmark.RIGHT_HEEL] > ys[mpPose.PoseLandmark.RIGHT_HIP]:
        hips_above_hands_and_foot = True
    
    # Consider main key_points
    main_key_points = [
        key_points[mpPose.PoseLandmark.NOSE],
        key_points[mpPose.PoseLandmark.RIGHT_SHOULDER],
        key_points[mpPose.PoseLandmark.LEFT_SHOULDER],
        key_points[mpPose.PoseLandmark.RIGHT_ELBOW],
        key_points[mpPose.PoseLandmark.LEFT_ELBOW],
        key_points[mpPose.PoseLandmark.RIGHT_WRIST],
        key_points[mpPose.PoseLandmark.LEFT_WRIST],
        key_points[mpPose.PoseLandmark.RIGHT_HIP],
        key_points[mpPose.PoseLandmark.LEFT_HIP],
        key_points[mpPose.PoseLandmark.RIGHT_KNEE],
        key_points[mpPose.PoseLandmark.LEFT_KNEE],
        key_points[mpPose.PoseLandmark.RIGHT_ANKLE],
        key_points[mpPose.PoseLandmark.LEFT_ANKLE],
    ]

    # Get coordinates of main key points
    xs = []
    ys = []
    for point in main_key_points:
        xs.append(point[0])
        ys.append(point[1])

    # Calculate relative vertical and horizontal distances between main key points
    distaces = []
    max_horizontal_distance = max(xs)-min(xs)
    max_vertical_distance = max(ys)-min(ys)
    for i in range(len(main_key_points)):
        for j in range(i+1, len(main_key_points)):
            horizontal_distance = abs(main_key_points[i][0]-main_key_points[j][0]) / max_horizontal_distance
            vertical_distance = abs(main_key_points[i][1]-main_key_points[j][1]) / max_vertical_distance
            distaces.append(horizontal_distance)
            distaces.append(vertical_distance)

    return np.array(distaces + [horizontality, hands_feet_ratio, hands_under_body, hips_above_hands_and_foot])


def create_input():
    X = []
    y = []
    df = pd.read_csv(train_csv_path)
    for i,filename in enumerate(os.listdir(train_images_path)):
        print(i,filename)
        img = cv2.imread(train_images_path + "/" + filename)
        coded_image = code_image(img)
        if len(coded_image) == 0: continue
        class_6 = int(df.loc[df['image_id'] == filename]['class_6'])
        X.append(coded_image)
        y.append(class_6)

    return np.array(X),np.array(y)


def create_and_save_input():
    X,y = create_input()
    file_name = os.path.join(data_path, 'matrix.pickle')
    pickle.dump((X,y), open(file_name, "wb"))


def load_input():
    file_name = os.path.join(data_path, 'matrix.pickle')
    return pickle.load(open(file_name, "rb"))


def show_input_sample():
    X,y = load_input()
    print(X[0], y[0])


def explore():
    df = pd.read_csv(train_csv_path)
    for filename in os.listdir(train_images_path)[:100]:
        img = cv2.imread(train_images_path + "/" + filename)
        coded_image = code_image(img)
        if len(coded_image) == 0: continue
        class_6 = int(df.loc[df['image_id'] == filename]['class_6'])
        if class_6 != 4: continue
        print(filename)
        print(coded_image, '->', class_6)
        plt.imshow(img)
        plt.show(block=True)


if __name__ == "__main__":
    create_and_save_input()