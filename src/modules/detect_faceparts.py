import cv2
import dlib
import sys
import os
import datetime
import glob


def faceparts_detector(dir):
    img_list_path = dir+ "*.jpg"
    img_list = glob.glob(img_list_path)
    #img_list = glob.glob("imgdata/face_img/*.jpg") 直書きするとこんな感じ
    detector = dlib.get_frontal_face_detector()
    predictor_file_path = "models/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_file_path)

    for img in img_list:
        frame = cv2.imread(img)
        cv2.imshow("output",frame)
        '''
        #cap = cv2.VideoCapture(0) リアルタイムの場合はこちらを採用
        while True:
            #ret, frame = cap.read() 動きの時
            cv2.imshow("output",frame)

            if cv2.waitKey(1) == 27:
                break
        #cap.release() 動画のストップ
        '''
        cv2.waitKey(0) #画像の場合の処理
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #img_list = glob.glob("imgdata/face_img/*.jpg")
    dir_path = "imgdata/face_img/"
    faceparts_detector(dir_path)
