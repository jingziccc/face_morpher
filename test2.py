from utils import get_landmarks_points, draw_delaunay, face_morph

input_path = ".\\input\\"

input_images = [input_path+"arya.jpg",input_path+ "claire.jpg"]
alpha = 0.5
face_morph(input_images, alpha)
