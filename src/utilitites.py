import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def frame_board_pose_estimation(mtx,dist,img, board, dict = None):
    """
    Estimates the rotation and translation of the origin of the frame of the board
    mtx: camera coefficient matrix to account for projections
    dist: camera distortion coefficint matrix to account for lens distortions
    img: image with board on frame 
    board: board object with the real parameters of the board used
    dict (optional):aruco dictionary used during board generation
    ---
    r: rotation from camera space to board space
    t: translation from camera space to board space
    """
    # Change if no reprojection error has to be computed
    val = True
    # Detect board
    charuco_detector = cv2.aruco.CharucoDetector(board)
    #aruco_detector = cv2.aruco.ArucoDetector(dict)
    res = charuco_detector.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    # Detect pose and return rotation matrix and translation
    _, r, t = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners = res[0],
                charucoIds = res[1],
                board = board,
                cameraMatrix =mtx,
                distCoeffs = dist,
                rvec = np.eye(3),
                tvec = np.zeros(3),
                useExtrinsicGuess = False)
    
    return r, t

def load_params(path):
        mtx = np.eye(3)
        dist = np.zeros(14)
        with open(path,'rb') as file:
            data = np.loadtxt(file)
        mtx[0,:], mtx[1,:], mtx[2,:] = data[0:3] , data[3:6], data[6:9]
        dist = data[9:].T
        return mtx, dist

def save_params(pth,params):
    """
    Given a certain destination path, save the given parameters to a txt file
    """
    np.savetxt(pth, params)
    return

def calibrate_camera(pth: str):
    """
    Calibrates the camera using the dected corners.
    path: path to calibration video
    board: board object from which the theoretical points are extracted
    phone_to_calibrate: phone_# to select the appropriate folder to save and load parameters
    ---
    most important outputs: camera_matrix, distortion_coefficients0 

    """
    print("CAMERA CALIBRATION")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((6, 8), 0.037, 0.029, aruco_dict)
    allCorners, allIds, imsize = get_calibration_keypoitns(os.path.split(pth)[0], board)
    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)

    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors, _, _, errors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    
    paramerters = np.hstack((camera_matrix.ravel(),distortion_coefficients0.ravel()))
    
    fig, ax = plt.subplots(figsize = (5, 3) )
    ax.scatter(np.linspace(1, len(errors) + 1, len(errors)),errors)
    ax.plot(np.linspace(1, len(errors) + 1, len(errors)),errors)
    ax.set_xlabel('Image n')
    ax.set_ylabel('Error (pixels)')
    fig.tight_layout()
    plt.show()
    save_params(pth, paramerters)

    return  ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors, errors


def get_calibration_keypoitns(path, board):
    """
    Gets the desired points of interests for each board in frame
    path: path to calbration video
    board: board with real parameters
    ---
    Corners: Corners detected by the charuco detector
    Ids: Ids detected by the charuco detector
    imsize: image shape after reshaping
    """
    allCorners = []
    allIds = []
    decimator = 0
    off = 0
    # DETECTOR INSTANTIATION FOR CHARUCO
    charuco_detector = cv2.aruco.CharucoDetector(board)
    files = os.listdir(path)
    for idx, im in enumerate(files):
        if ('.png' in im) or ('.jpg' in im):
            print("=> Processing image {0}".format(idx - off))
            frame = cv2.imread(path + r'\\' + im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)              
            res2 = charuco_detector.detectBoard(gray)

            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[0])
                allIds.append(res2[1])

            decimator+=1
            imsize = gray.shape
        else:
            off +=1

    return allCorners, allIds, imsize

def check_timestamp(pth):

    vidcap = cv2.VideoCapture(pth)
    N = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamp = np.zeros(N)

    for i in range(N):
        _,_ = vidcap.read()
        timestamp[i] = vidcap.get(cv2.CAP_PROP_POS_MSEC)
    
    plt.plot(np.diff(timestamp).astype(int))
    plt.show()
    return