# coding:utf-8
"""
脸部68个特征点检测
"""
import cv2
import dlib
import time

# 加载并初始化检测器
# 模型下载地址http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("cannot open camera")
    exit(0)

fps = 25
bzx_new = 0
bzx_old = 0
bzy_new = 0
bzy_old = 0
move=5
while True:
    
    ret, frame = camera.read()
    if not ret:
        continue
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    dets = detector(frame_new, 1)
    # 查找脸部位置
    for i, face in enumerate(dets):
        shape = predictor(frame_new, face)
        # 绘制特征点
        cv2.circle(frame, (shape.part(39).x, shape.part(39).y),
                   3, (0, 0, 255), 2)
        cv2.circle(frame, (shape.part(42).x, shape.part(42).y),
                   3, (0, 0, 255), 2)
        cv2.circle(frame, (shape.part(30).x, shape.part(30).y),
                   3, (0, 0, 255), 2)
        bzx_old = shape.part(30).x
        bzy_old = shape.part(30).y
        cv2.circle(frame, (shape.part(48).x, shape.part(48).y),
                   3, (0, 0, 255), 2)
        cv2.circle(frame, (shape.part(54).x, shape.part(54).y),
                   3, (0, 0, 255), 2)

    cv2.imshow("Camera", frame)
    
    abs_bzx = abs(bzx_new-bzx_old)
    abs_bzy = abs(bzy_new-bzy_old)
    
    if bzx_new | bzy_new is None:
        bzy_new = bzx_old
        bzy_new = bzy_old
    else:
        if ((abs_bzx) > move) | ((abs_bzy) > move):
            print('鼻子动了')
    bzx_new = bzx_old
    bzy_new = bzy_old

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
