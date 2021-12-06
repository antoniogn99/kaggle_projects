import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os
from skimage.io import imread
import matplotlib.pyplot as plt

mpPose = mp.solutions.pose
mpDrawingUtils = mp.solutions.drawing_utils


def create_blackie_from_image(img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blackie = np.zeros(img.shape)
        with mpPose.Pose() as pose:
                results = mpPose.Pose().process(imgRGB)
                if results.pose_landmarks:
                        mpDrawingUtils.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # draw landmarks on blackie
                else:
                        print('No key points found. Returning blank image')
        return blackie


def extract_key_points_from_image(img):
        temp = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mpPose.Pose() as pose:
                results = pose.process(imgRGB)
                if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        for i,j in zip(mpPose.PoseLandmark,landmarks):
                                temp.append([j.x, j.y, j.z])
                else:
                        print('No key points found. Returning empty list')
        return temp


def example():
        image_path = 'C:\\Users\\anton\\kaggle\\yoga\\data\\images\\train_images\\0a5f8841c8dfb2d67d844a66e39658b8.jpg'
        image = imread(image_path)
        plt.imshow(image)
        plt.show(block=True)

        blackie = create_blackie_from_image(image)
        plt.imshow(blackie)
        plt.show(block=True)

        points = extract_key_points_from_image(image)
        print(points)


if __name__ == "__main__":
    example()