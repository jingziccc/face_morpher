import numpy as np
import cv2
import dlib
# Load the landmarks from the .npy file
detector = dlib.get_frontal_face_detector() # 获取人脸分类器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 获取人脸特征点分类器
img = cv2.imread('.\\input\\arya.jpg')
dets = detector(img, 1)
# landmarks = np.zeros((68, 2)) 
    
# for k, d in enumerate(dets):
#     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#         k, d.left(), d.top(), d.right(), d.bottom()))
#     # 获取特征点
#     shape = predictor(img, d) 
#     for i, pt in enumerate(shape.parts()):
#         # 保存特征点坐标
#         landmarks[i] = (pt.x, pt.y) 
#         # 绘制特征点，并用cv2.circle给每个特征点画一个圈，共68个
#         pt_pos = (pt.x, pt.y)
#         cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)

landmarks = np.load('.\\output\\arya_landmarks.npy')
# # Print the landmarks

subdiv = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
for p in landmarks:
    subdiv.insert((p[0], p[1]))
