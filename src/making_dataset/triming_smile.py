"""
youtube等の動画から笑顔のタイミングを取得する
笑顔のタイミング1.5秒前からの動画をトリミングしていくスクリプトを書く
smileかどうかは何で判断しようか...? => opencvのsmileカスケードでいいかな？
"""

import cv2
import os

def smile_capture():
    #file = "video_path"
    show_window_name = 'now_frame'
    #表示するwindow nameを統一するための変数
    face_detect_cascade_path = "../modules/haarcascades/haarcascade_frontalface_default.xml"
    face_detect_cascade = cv2.CascadeClassifier(face_detect_cascade_path)
    #顔の検出のためのカスケード

    smile_cascade_path = "../modules/haarcascades/haarcascade_smile.xml"
    smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
    #笑顔検出のためのカスケード


    #cap = cv2.VideoCapture(file) #Videoからのキャプチャーをするときはこの引数に指定する
    cap = cv2.VideoCapture(0) #webカメラなどデバイスから動画を検出する場合は引数に数値にしていする

    while cap.isOpened():
        ret, frame = cap.read()
        #カメラ画像の取得開始
        if not ret:
            break #frame数がなくなったら終了する

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #grayスケールに変換してから判別を行う
        face_list =face_detect_cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 2, minSize=(50,50))
        #顔の判定

        if len(face_list) > 0: #顔を検出したら
            for x,y,w,h in face_list:
                face = frame[y:int((y+h)), x:int((x+w))] #顔の位置座標を取りに行く
                scale = 480 / h #大きさの調整
                face = cv2.resize(face, dsize=None, fx=scale, fy=scale) #比率をそのままに大きさを表示する
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                smile_detector =smile_cascade.detectMultiScale(face_gray,scaleFactor= 1.2, minNeighbors=50, minSize=(80, 50))
                #smileの判定
                if len(smile_detector) >0:
                    #笑顔検出したときだけ切り抜くってやりたい  => うまく検出できたので, 一定フレーム数が溜まったら切り抜く形にする
                    cv2.imshow("face",face) #顔だけ切り抜いて表示する
                    cv2.moveWindow(show_window_name, 1200, 0)
                else:
                    cv2.destroyWindow("face")



        cv2.imshow(show_window_name,frame) #webカメラからの画像を取得し続ける
        cv2.moveWindow(show_window_name, 0, 0)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    smile_capture()

"""
参考URL:
https://note.nkmk.me/python-opencv-video-to-still-image/ フレームの切り出し

"""
