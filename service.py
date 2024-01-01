# from nebullvm.inference_learners.base import LearnerMetadata
import os
import yaml
import sys
import numpy as np
import onnxruntime as ort
import torch
import traceback
import time
# embedding_model_path_a = os.path.join("Services", "face_recognition", "resources", "face_recoginition", "weights", "facenet_light")
session_reco = ort.InferenceSession("Services/face_recognition/resources/face_recoginition/weights/facenet/model.onnx", providers=["TensorrtExecutionProvider"])

from ultralytics import YOLO
face_model = YOLO("Services/face_recognition/resources/face_detection/weights/wider_face_best.pt") 

# optimized_model_reco = LearnerMetadata.read(embedding_model_path_a).load_model(embedding_model_path_a)
def optimized_model_reco(image):
    outputs = session_reco.run(None, {session_reco.get_inputs()[0].name: image.astype(np.float32)})[0][0]
    return outputs




input_shape = (160, 160)
threshold = 0.3
distance_metric = "cosine"
# ---------------------------------------------------------------------------------------
sys.path.append("Services/face_recognition/resources/face_align")
from TDDFA import TDDFA
from utils.functions import draw_landmarks, get_suffix
from utils.pose import viz_pose, viz_pose_custom_view,viz_pose_custom

# config_file_path = "Services/face_recognition/resources/face_align/configs/mb1_120x120.yml"
config = "Services/face_recognition/resources/face_align/configs/mb05_120x120.yml"

onnx = True
mode = "cpu"
opt = "pose"
show_flag = True
dense_flag = False
new_suffix = f".{opt}" if opt in {"ply", "obj"} else ".jpg"
cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
if onnx:
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "4"

    # from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX

    tddfa = TDDFA_ONNX(**cfg)
else:
    gpu_mode = mode == "gpu"
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)

# ----------------------------------------------------------------------------------------
working_dir = os.getcwd()
change_dir_path = os.path.join(working_dir, "Services/face_recognition/resources/face_detection")
sys.path.append(change_dir_path)
os.chdir(change_dir_path)


from vision.ssd.config.fd_config import define_img_size


input_img_size = 320
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "models/voc-model-labels.txt"


class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = "cpu"

candidate_size = 1000
threshold1 = 0.9
# print(1)
path_det = os.path.join("weights", "small_detector_slim_320","model.onnx")
# Services/face_recognition/resources/face_detection/weights/small_detector_RFB_640
# path_det = os.path.join('weights','1mb_optimize_withoutacc_hit')
# net = LearnerMetadata.read(path_det).load_model(path_det)
# print(2)

session_det = ort.InferenceSession(path_det, providers=["TensorrtExecutionProvider"])
# print(3)

# optimized_model_reco = LearnerMetadata.read(embedding_model_path_a).load_model(embedding_model_path_a)
def net(image):
    image = image.numpy()
    outputs = session_det.run(None, {session_det.get_inputs()[0].name: image.astype(np.float32)})
    # outputs = torch.from_numpy(outputs)
    return torch.from_numpy(outputs[0]),torch.from_numpy(outputs[1])


# print(4)



predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)

# print(5)

def face_aligner(frame, detected_faces):
    try:
        # s1 = time.time()
        param_lst, roi_box_lst = tddfa(frame, detected_faces)
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        roll = viz_pose_custom_view(frame, param_lst, ver_lst, show_flag=show_flag)
        # print("align",time.time()-s1)
        # post processing of aligned image

    except Exception as e:
        roll = []
        print(f"face_aligner error due to {e}")
        print(traceback.print_exc())

    return frame, roll


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


# change_dir_path = os.path.join(os.getcwd(), "Services/face_recognition/resources/face_detection")
print(working_dir)
os.chdir(working_dir)
