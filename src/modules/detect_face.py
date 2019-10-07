#68点モデルの使用を考えていたが,192点モデルのものがみつかったので,明日学校でモデルの作成を行う 20191007
import cv2
import sys
import os
import datetime
'''
facedetectしてから, そこから特徴点抽出する(別モジュール)
'''

def face_detector(img):
    #dirの構造がない時は自動でdirの作成を行う
    if os.path.exists("./imgdata") ==False:
        os.mkdir("./imgdata/face_img")

    origin_img = cv2.imread(img)
    img = origin_img.copy()
    #imgの読み込みを行う

    cascade_path = "haarcascades/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    #顔判定を行うカスケードの設定(openCVについているデフォルトのものを使用)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #facedetectをするときはgrayスケールで読み込みを行う
    faces_list = cascade.detectMultiScale(gray_img,scaleFactor =1.1, minNeighbors =3,minSize=(100,100))

    print(faces_list)#検出した顔の個数をカウントするためのリスト
    if len(faces_list) == 0:
        print("Falled or Noface")
        quit()

    index = 1 #顔が複数あった場合の処理
    #検出した顔それぞれを保存する
    for (x,y,w,h) in faces_list:
        face_img = origin_img[y:int((y*0.8+h*1.2)), x:(x+w)]
        dt_now = datetime.datetime.now()
        timestamp=dt_now.strftime('%Y%m%d%H%M%S')
        filename = "imgdata/face_img/face_" + timestamp+"_"+str(index)+".jpg"
        cv2.imwrite(filename, face_img)
        index = index +1 #複数あった場合はここでindexを振る


if __name__ == '__main__':
    img_path = sys.argv[1] #第一引数に検出した画像のpathを指定する
    face_img = face_detector(img_path)
