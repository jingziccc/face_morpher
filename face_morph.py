from utils import get_landmarks_points, draw_delaunay, face_morph,add_landmarks,output_path
import cv2,numpy as np




input_path = ".\\input\\"

input_images = [input_path+"cat.jpg",input_path+ "tiger.jpg"]
alpha = 0.5
face_morph(input_images, alpha,True)
