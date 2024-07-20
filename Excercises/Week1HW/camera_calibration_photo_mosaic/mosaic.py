import cv2
import numpy as np
import glob

# Load Calibration Parameters
def load_calibration(calibration_file):
    with np.load(calibration_file) as data:
        camera_matrix = data['mtx']
        dist_coeffs = data['dist']
    return camera_matrix, dist_coeffs

# Undistort Image
def undistort_image(image, camera_matrix, dist_coeffs):
    image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    return undistorted_img

# Harris Corner Detection
def harris_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
    return image, corners

# Match Features Between Images
def match_features(image1, image2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    #matches = sorted(matches, key=lambda x: x.distance)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Build src and dst arrays
    src = []
    dst = []
    distances = []
    for match in good:
        srcpt = kp1[match.queryIdx].pt
        dstpt = kp2[match.trainIdx].pt

        src.append(srcpt)
        dst.append(dstpt)

        distances.append(abs(dstpt[1] - srcpt[1]))

    
    least = np.array(distances) < 30
    least = np.tile(least, (2, 1)).T
    format_pts = lambda x: np.float32(x)[least].reshape(-1, 1, 2)

    return format_pts(src), format_pts(dst)

# Create Mosaic
def create_mosaic(images, camera_matrix, dist_coeffs):
    undistorted_images = [undistort_image(img, camera_matrix, dist_coeffs) for img in images]
    mosaic = undistorted_images[0]
    
    for i in range(1, len(undistorted_images)):
        img1 = undistorted_images[i]
        img2 = mosaic
        img1_corners = harris_corner_detection(img1)
        img2_corners = harris_corner_detection(img2)
        pts1, pts2 = match_features(img1_corners, img2_corners)
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        # Warp the current mosaic image
        dst = cv2.warpPerspective(img1, H, (img1.shape[1] + mosaic.shape[1], img1.shape[0]))
        
        # Blend current image into mosaic
        z = np.zeros((dst.shape[0], dst.shape[1] - mosaic.shape[1], 3))
        mosaic2 = np.concatenate((mosaic, z), axis=1)

        tmp = np.where(mosaic2 > 0, mosaic2, dst)
        mosaic = np.uint8(tmp)

    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    mosaic = mosaic[y:y + h, x:x + w]

    return mosaic

calibration_file = 'camera_calibration.npz'
images_path = 'International Village - 50 Percent Overlap/*.jpg'
camera_matrix, dist_coeffs = load_calibration(calibration_file)
images = [cv2.imread(img_path) for img_path in glob.glob(images_path)]

if images:
    mosaic_image = create_mosaic(images, camera_matrix, dist_coeffs)
    cv2.imwrite('mosaic_image.jpg', mosaic_image)
    mosaic_image = cv2.resize(mosaic_image, (4032//4,3024//4))
    cv2.imshow('Mosaic', mosaic_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No images found in the specified directory.")


# 1. The 50% overlap has best feature extraction
# 2. More overlap = better stitching since opencv has more points to match
# 3. The brick pics mosaic is trash compared to the others because of the extreme similarities between
# all the pictures messing with the feature matcher.