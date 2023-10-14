import dlib
import cv2
import numpy as np
import moviepy.editor as mpy
import os
from scipy.spatial import Delaunay

detector = dlib.get_frontal_face_detector() # 获取人脸分类器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 获取人脸特征点分类器
output_path = ".\\output\\"
image_name = ""

def get_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

# 获取面部特征点，并将点保留在npy文件中

def get_landmarks_points(image):
    img = image.copy()
    # 检测人脸
    dets = detector(img, 1)
    points = []
    # d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()是人脸的左上角和右下角坐标
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # 获取特征点
        shape = predictor(img, d) 
        for i, pt in enumerate(shape.parts()):
            # 保存特征点坐标
            points.append([pt.x, pt.y])
            # 绘制特征点，并用cv2.circle给每个特征点画一个圈，共68个
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
            cv2.putText(img, str(i), pt_pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))

    size = img.shape # 获取图片大小
    size = (size[1]-1, size[0]-1)
    for i, p in enumerate([(0,0), (0,size[1]), (size[0],0), size, (size[0]/2,0), (size[0]/2,size[1]), (0,size[1]/2), (size[0]/2,size[1]/2)]):
        points.append(p)
    # 保存特征点坐标，名称为img+landmarks.npy，路径为output
    np.save(output_path + image_name + '_landmarks.npy', np.array(points))
    # 保存图片，名称为img_landmarks.jpg，路径为output
    cv2.imwrite(output_path + image_name + '_landmarks.jpg', img)
    
    # return points
    

def add_landmarks(image_path):
    """_summary_
    给图片添加特征点，通过点击鼠标左键添加特征点，右键删除最近的特征点
    """
    global image_name
    image_name = get_name(image_path)
    img = cv2.imread(image_path)
    points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            # 添加特征点
            points.append((x, y))
            # 绘制特征点
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(img, str(len(points)-1), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Image", img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 删除最近的特征点
            if len(points) > 0:
                dists = [((x - p[0]) ** 2 + (y - p[1]) ** 2) for p in points]
                idx = np.argmin(dists)
                points.pop(idx)
                # 重新绘制特征点
                img_copy = img.copy()
                for i, p in enumerate(points):
                    cv2.circle(img_copy, p, 2, (0, 0, 255), -1)
                    cv2.putText(img_copy, str(i), (p[0]+5, p[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Image", img_copy)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            size = img.shape # 获取图片大小
            size = (size[1]-1, size[0]-1)
            for i, p in enumerate([(0,0), (0,size[1]), (size[0],0), size, (size[0]/2,0), (size[0]/2,size[1]), (0,size[1]/2), (size[0]/2,size[1]/2)]):
                points.append(p)
            # 保存特征点坐标，名称为img+landmarks.npy，路径为output
            np.save(output_path + image_name + '_landmarks.npy', np.array(points))
            # 保存图片，名称为img_landmarks.jpg，路径为output
            cv2.imwrite(output_path + image_name + '_landmarks.jpg', img)
            break

    cv2.destroyAllWindows()

# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# 为传入的图片进行处理并画delaunay triangles
#steps:
#1.获取landmark detection的点位
#2.使用算法画出delaunay triangles

def draw_delaunay(image_path):
    # 读取图片
    img = cv2.imread(image_path)
    # 创建subdiv
    subdiv = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
    points = np.load(output_path + os.path.splitext(os.path.basename(image_path))[0] + '_landmarks.npy')
    # 加入特征点
    for p in points:
        subdiv.insert((p[0], p[1]))
    triangleList = subdiv.getTriangleList()#获取三角形列表
    size = img.shape
    r = (0, 0, size[1], size[0])
    delaunay_color = (255, 255, 255)
    # 画出三角形
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))  # Convert to integers
        pt2 = (int(t[2]), int(t[3]))  # Convert to integers
        pt3 = (int(t[4]), int(t[5]))  # Convert to integers

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
    # 保存图片
    cv2.imwrite(output_path + os.path.splitext(os.path.basename(image_path))[0] + '_delaunay.jpg', img)

# 仿射变换
def affine_transform(input_image, input_triangle, output_triangle, size):
    """
    仿射变换
    """
    warp_matrix = cv2.getAffineTransform(
        np.float32(input_triangle), np.float32(output_triangle))
    output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return output_image
    
def morph_triangle(img1, img2, img, tri1, tri2, tri, alpha):
    """_summary_
    三角形变形，Alpha 混合
    """
    # 计算三角形的边界框
    rect1 = cv2.boundingRect(np.float32([tri1]))# [array([227., 157.]), array([229., 133.]), array([299., 299.])] => (227, 133, 73, 167)
    rect2 = cv2.boundingRect(np.float32([tri2]))
    rect = cv2.boundingRect(np.float32([tri]))

    tri_rect1 = []
    tri_rect2 = []
    tri_rect_warped = []
    
    for i in range(0, 3):
        tri_rect_warped.append(
            ((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
        tri_rect1.append(
            ((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        tri_rect2.append(
            ((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

    # 在边界框内进行仿射变换
    img1_rect = img1[rect1[1]:rect1[1] +
                     rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_rect = img2[rect2[1]:rect2[1] +
                        rect2[3], rect2[0]:rect2[0] + rect2[2]]
    
    size = (rect[2], rect[3])
    warped_img1 = affine_transform(
        img1_rect, tri_rect1, tri_rect_warped, size)
    warped_img2 = affine_transform(
        img2_rect, tri_rect2, tri_rect_warped, size)

    # 加权求和
    img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2
    
    # 将变形后的三角形放回原始图像中
    mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)
    
    img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
        img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] +
            rect[2]] * (1 - mask) + img_rect * mask
    




def face_morph(image_path, alpha=0.5,is_animal=False):
    """_summary_
    融合两张图片
    """
    global image_name
    image_name = get_name(image_path[0])
    img1 = cv2.imread(image_path[0])
    if(not is_animal):
        get_landmarks_points(img1)
    points1 = np.load(output_path + image_name + '_landmarks.npy')
    
    image_name = get_name(image_path[1])  
    img2 = cv2.imread(image_path[1])
    if(not is_animal):
        get_landmarks_points(img2)
    points2 = np.load(output_path + image_name + '_landmarks.npy')
    
    points = (1 - alpha) * np.array(points1) + alpha * np.array(points2) # [[0,0][2,3]...]
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)
    triangles = Delaunay(points).simplices # 获取三角形索引列表[[71 16 70] [15 16 71]...]
    
    for i in triangles:
        x = i[0]
        y = i[1]
        z = i[2]
        tri1 = [points1[x], points1[y], points1[z]]
        tri2 = [points2[x], points2[y], points2[z]]
        tri = [points[x], points[y], points[z]]
        # 计算仿射变换
        morph_triangle(img1, img2, img_morphed, tri1, tri2, tri, alpha)
        # cv2.imshow("Morphed Image", img_morphed)
        # # 等0.1s
        # cv2.waitKey(20)
    # 保存图片
    cv2.imwrite(output_path + get_name(image_path[0]) + '_' + get_name(image_path[1]) + '_morphed.jpg', img_morphed)
    