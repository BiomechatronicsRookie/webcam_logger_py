import numpy as np
import cv2
import argparse
import threading
import time
import os
from utilitites import *

subject_number = 3
save_path = r"C:\Users\NakamaLab\Videos\\" #Save format file path included extension


class webcam(threading.Thread):

    def __init__(self, web_id :str = None, source: int = None, fps: float = None, flag = False):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.flag = flag
        self.camera = camera(web_id, source, fps, flag)
        return
    
    def run(self):
        if self.flag == 'Stream':
            self.camera.stream()
        if self.flag == 'Record':
            self.camera.record()
            
    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
    
class camera():
    def __init__(self, web_id :str = None, source: int = None, fps: float = None, flag = False):
        self.web_id = web_id
        if '1' in self.web_id:
            self.phone_n = 1
        elif '2' in self.web_id:
            self.phone_n = 2
        self.source = source
        self.fps = fps
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        self.capture.set(3, 1920)
        self.capture.set(4, 1080)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        self.frame_size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_size_rotated = (int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.calibration_path = None# Path to save calibration pictures to
        self.parameters_path = None # Path to save the calibration parameters to
        self.must_calibrate = False
        self.est_pose = False

        if self.must_calibrate:
            if os.path.isfile(self.parameters_path):
                self.mtx, self.dst = load_params(self.parameters_path)
                self.est_pose = True
            else:
                print('Get images and calibrate the camera')
                if int(len(os.listdir(self.calibration_path))) > 20:
                    self.calibrate()
                else:
                    self.get_calibration_imgs()
                    self.calibrate()
                    self.mtx, self.dst = load_params(self.parameters_path)
                    self.est_pose = True
        return
    
    def stream(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        #board = cv.aruco.CharucoBoard((6, 8), 0.035, 0.029, aruco_dict)
        board = cv2.aruco.CharucoBoard((3, 4), 0.136, 0.0865, aruco_dict)
        prev_time = 0
        fps = 0
        while True:
            ret, img = self.capture.read()
            if ret:
                new_time = time.time()
                fps = 1/(new_time - prev_time)
                prev_time = new_time
                #img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                if self.est_pose:
                    r, t = frame_board_pose_estimation(self.mtx, self.dst, img, board)
                    cv2.drawFrameAxes(img, self.mtx, self.dst, r, t, 0.3, 4)
                cv2.putText(img, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow(self.web_id ,cv2.resize(img,(0,0), fx = 0.4, fy = 0.4))
                val = cv2.waitKey(1)
                if val == ord('q'):
                        cv2.destroyWindow(self.web_id )
                        self.capture.release()
                        break
                elif val == ord(' '):
                    n = len(os.listdir(self.calibration_path))
                    cv2.imwrite(self.calibration_path+'\img_{0}.png'.format(str(n).zfill(3)),img)
                    print('saved img {0}'.format(n))
                
        return True
    
    def record(self): # Do not estimate pose while recording, as it takes resources from the PC
        file_n = len(os.listdir(os.path.split(save_path)[0].format(subject_number, self.phone_n)))
        self.filename = save_path + str(file_n)+ '.mp4'
        
        self.file_writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps , self.frame_size)

        while True:
            _, img = self.capture.read()
            #img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow(self.web_id ,cv2.resize(img,(0,0), fx = 0.4, fy = 0.4))
            self.file_writer.write(img)
            val = cv2.waitKey(1)
            if val == ord('q'):
                    cv2.destroyWindow(self.web_id)
                    self.file_writer.release()
                    self.capture.release()
                    break
        return True
    
    def calibrate(self):
        calibrate_camera(self.parameters_path)
        return
    
    def get_calibration_imgs(self):
        n_imgs = 30
        n = 0
        print('Acquiring calibration images for {0}, press space to save image'.format(self.web_id))
        while n < n_imgs:
            _, img = self.capture.read()
            #img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow(self.web_id ,cv2.resize(img,(0,0), fx = 0.4, fy = 0.4))
            val = cv2.waitKey(1)
            if val == ord('q'):
                    cv2.destroyWindow(self.web_id )
                    self.capture.release()
                    break
            elif val == ord(' '):
                n = len(os.listdir(self.calibration_path))
                cv2.imwrite(self.calibration_path+'\img_{0}.png'.format(str(n).zfill(3)),img)
                print('saved img {0} of 30 for {1}'.format(n, self.web_id))
        cv2.destroyWindow(self.web_id )
        self.capture.release()

        return

    
def main():
    flag = 'Record'
    thread_cam1 = webcam('Webcam 1', 0, 30.0, flag = flag)
    #thread_cam2 = webcam('Webcam 2', 1, 60.0, flag = flag)
    thread_cam1.start()
    #thread_cam2.start()
    thread_cam1.join()
    #thread_cam2.join()

    return

if __name__=='__main__':
    main()