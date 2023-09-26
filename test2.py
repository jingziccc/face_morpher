import dlib
import cv2

# Path to the facial landmark predictor model
predictor_path = "./shape_predictor_68_face_landmarks.dat"

# Path to the input image
faces_path = "./examples/p1.jpg"

# Load the frontal face detector from Dlib
detector = dlib.get_frontal_face_detector()

# Load the facial landmark predictor model
predictor = dlib.shape_predictor(predictor_path)

# Read the input image
img = cv2.imread(faces_path)

# Detect faces in the image using the frontal face detector
dets = detector(img, 1)

# Loop through each detected face
for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))

    # Use the predictor to detect facial landmarks for the current face
    shape = predictor(img, d)

    # Save the detected facial landmarks to a text file ("points.txt")
    with open("points.txt", "w") as f:
        for pt in shape.parts():
            f.write("{} {}\n".format(pt.x, pt.y))

    # Draw the facial landmarks on the image
    for index, pt in enumerate(shape.parts()):
        print('Part {}: {}'.format(index, pt))
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)

# Display the image with facial landmarks
cv2.imshow('test2', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the processed image with landmarks as "image.jpg"
cv2.imwrite("image.jpg", img)
