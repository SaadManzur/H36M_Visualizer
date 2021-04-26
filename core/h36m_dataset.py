import os
import h5py
import numpy as np
import cv2 as opencv

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.pose import draw_skeleton

SUBJECTS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
INDICES = [0, 1, 2, 3, 6, 7, 8, 13, 17, 18, 19, 25, 26, 27]

class H36MDataset(object):

    def __init__(self, base_path):
        super().__init__()
        self._base_path = base_path

    def load_file(self, subj, action, cam_id):

        file_path = os.path.join(self._base_path, "processed", subj, action, "annot.h5")

        with h5py.File(file_path, 'r') as file_:

            pose2d = file_['pose/2d'][:, INDICES, :]
            max_frame = np.max(file_['frame'][:])

            video_action = action.replace("-", " ")

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)

            start_frame = 0

            for i in range(0, pose2d.shape[0], max_frame):

                if cam_id == file_['camera'][i]:
                    start_frame = i
                    break
            
            vid_path = os.path.join(self._base_path, "extracted", subj, "Videos", f"{video_action}.{cam_id}.mp4")

            capture = opencv.VideoCapture(vid_path)

            total_frames = int(capture.get(opencv.CAP_PROP_FRAME_COUNT))

            for i_frame in range(start_frame, start_frame+total_frames):

                ret, frame = capture.read()
                frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)

                frame_pose = pose2d[i_frame, :, :]
                draw_skeleton(frame_pose, ax)

                ax.imshow(frame)
                plt.draw()
                plt.pause(1e-8)
                ax.clear()

            file_.close()
