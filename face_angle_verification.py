# from ai_models.face_align.face_align_model import tddfa
# from ai_models.face_align.utils.pose import viz_pose_custom
# import cv2
# from vision.ssd.config.fd_config import define_img_size
# define_img_size(640)
# from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
# import os

# print(os.getcwd())

from service import (
    predictor,
    candidate_size,
    threshold1,
    tddfa,
    viz_pose_custom,
    face_model
    
)

import cv2
# label_path = "./models/voc-model-labels.txt"
# net_type = "RFB"
# class_names = [name.strip() for name in open(label_path).readlines()]
# num_classes = len(class_names)
# test_device = "cpu"
# candidate_size =10
# threshold1 = 0.85
# model_path = "models/pretrained/version-RFB-640.pth"
# net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
# predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
# net.load(model_path)
print("jjjjjj")

def face_angle(img,face_view):
    """
	This function checks if face is present or not and check the give face view is correct or not
    
	Parameters:
		img (string): image path 
		face_view (string): face_view check that the given view is true or not 

	Returns:
		if face is not present returns no face
        if detected face is blurry returns blurry face
        if face is detected and clear checks view and returns given view is true or not
	"""
    print("jjjjjj")
    # boxes, _, _ = predictor.predict(img, candidate_size / 2, threshold1)
    boxes = face_model.predict(img,classes=0,verbose=False, imgsz=[480],conf = 0.7)[0].boxes.cpu().numpy().data
    
    boxes = boxes.tolist()
    if len(boxes) is 0:
        return "No face detected", False
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    x1,y1,x2,y2,_,_ = boxes[0]
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    y1 = int((y2-y1)*0.15)+y1

    
    if (y2-y1)>150:
        pass
    else:
        return f"Too Far {(y2-y1)}", False
        
    corp_img = img[y1:y2, x1:x2]
    blurr_val = cv2.Laplacian(corp_img, cv2.CV_64F).var()
    if blurr_val < 70:
        return f"Face is blurry", False
    # yaw = viz_pose_custom_view(corp_img, param_lst, ver_lst, show_flag='true')
    yaw = viz_pose_custom(corp_img, param_lst, ver_lst, show_flag='true')[0]

    if face_view == "right":
        if (40 > yaw > 23):
            return "right face is verified", True
        else : 
            return "right face is not verified", False

    elif face_view == "left":
        if (-23 > yaw > -40):
            return "left face is verified", True
        else : 
            return "left face is not verified", False

    elif face_view == "front":
        if (23 > yaw > -23):
            return "Front face is verified", True
        else : 
            return "Front face is not verified", False
    
    return "Face is not verified", False

if __name__ == "__main__":
    print(face_angle(r"/home/dnw_omen/Pictures/Webcam/2023-02-22-131438.jpg","front"))