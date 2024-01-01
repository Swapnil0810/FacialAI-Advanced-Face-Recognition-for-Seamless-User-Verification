

from service import (
    optimized_model_reco,
    viz_pose_custom_view,
    tddfa,
    predictor,
    face_aligner,
    findCosineDistance,
    findEuclideanDistance,
    l2_normalize,
    candidate_size,
    threshold1,
    input_shape,
    threshold,
    distance_metric,
    face_model
)
import shutil
import traceback
print(optimized_model_reco)
print()
print(viz_pose_custom_view)
print()
print(tddfa)
print()
print(predictor)
print()
import datetime

import cv2
import numpy as np
import imutils
import pandas as pd
SCALE_X = {}
SCALE_Y = {}
# EMBEDDING_DICTIONARY = {}
# EMBEDDING_DICTIONARY[camera_id]["df"]
# ALL_DF_2 = pd.DataFrame([["default_person", np.random.rand(128), "1", distance_metric, "authorised", 0]], columns=["employee", "embedding", "unique_id", "distance_metric", "status", "distance"])

# width, height = 640, 480
fast_count_1, fast_count_2 = 0.75, 1

import time

from sys import path,argv
from pathlib import Path
from threading import Thread
# import sys
#relative imports are added here
from sort import *#changes
from service_skeleton.http_server import *#changes
from skeleton_updater import check_for_lib_updates  #changes
check_for_lib_updates(DJANGO_API_ENDPOINT) #changes

from face_angle_verification import face_angle

path.append(str(Path(__file__).resolve().parents[2]))

# from commons.codes.http_server import *

# from commons.codes.alerts import Alert

# from commons.codes.helper import check_heart_beat,get_config_json,send_version_and_classes_details
import json

from cv2 import imshow,waitKey
import os
import cv2

# from commons.codes.http_server import ALL_DF_2
service_name_dir = str(Path(__file__).resolve().parts[-2])

config_json_path = os.path.join('Services',service_name_dir,'config.json')

SERVICE_NAME = get_config_json(config_json_path)['service_name']
send_version_and_classes_details(config_json_path,DJANGO_API_ENDPOINT)#changes

# send_version_and_classes_details(config_json_path)
# width, height = 640, 480
width, height = 320, 240
import time
data_path = "/workspace/images"

def sig(number):
    return 0 if number < 0 else number

def face_detector(camera_id, frame):
    recognition_run = True
    try:
        # recognition_run = True
        global CAMERAS
        global CAMERAS_DATA
        # s1 = time.time()
        emty_numpy_array = face_model.predict(frame,classes=0,verbose=False, imgsz=[480],conf = 0.7)[0].boxes.cpu().numpy().data
        # print("det",time.time()-s1)
        track_bbs_ids = CAMERAS_DATA[camera_id]["sort_objects"].update(emty_numpy_array)
        # s1 = time.time()
        _, alignment_out = face_aligner(frame,track_bbs_ids)
        # print("alig",time.time()-s1)

        new_cam_path = f"{data_path}/{camera_id}"
        if not os.path.isdir(new_cam_path):
            os.mkdir(new_cam_path)

        for align_faces,coords in  zip(alignment_out,track_bbs_ids):
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            y1 = int((y2-y1)*0.15)+y1

            if (x2 - x1) < 200:
                continue

            x1, y1, x2, y2 = sig(int(coords[0])), sig(int(coords[1])), sig(int(coords[2])), sig(int(coords[3]))

            y1 = int((y2-y1)*0.15)+y1
            crop_frame = frame[int(y1):int(y2),int(x1):int(x2)]
            

            
            width = x2 - x1
            height = y2 - y1
            increase_width = 2 * width
            increase_height = 2 * height
            x1_new = int(x1 - increase_width)
            y1_new = int(y1 - increase_height)
            x2_new = int(x2 + increase_width)
            y2_new = int(y2 + increase_height)


            if int(coords[4]) in CAMERAS_DATA[camera_id]["face_track"]:
                x = np.array(CAMERAS_DATA[camera_id]["face_track"][int(coords[4])])
                y = np.array(align_faces)
                if int(np.linalg.norm(x - y)) < 3:
                    continue

            CAMERAS_DATA[camera_id]["face_track"][int(coords[4])] = align_faces

            yaw, pitch ,roll = align_faces 
            if -20 > pitch > 23:
                continue
            if abs(yaw) > 35:
                continue


            cam_dir_path = f"{data_path}/{camera_id}/{int(coords[4])}"
            



                
            if not os.path.isdir(cam_dir_path):
                os.mkdir(cam_dir_path)
                cam_dir_path_txt = f"{data_path}/{camera_id}/{int(coords[4])}.txt"

                Path(cam_dir_path_txt).touch()

                data = {'file_name': [],
                            'reco_name': [],
                            'best_distance': []}

                df = pd.DataFrame(data)
                df.to_csv(cam_dir_path+"/extracted.csv", index=False)
            # print("new_cam_path",new_cam_path)

            

            crop_event_save_path = f"{data_path}/{camera_id}/{int(coords[4])}/alert_image.jpg"

            if not os.path.exists(crop_event_save_path):
                x1_new, y1_new, x2_new, y2_new = sig(x1_new), sig(y1_new), sig(x2_new), sig(y2_new)
                crop_frame_evemt = frame[int(y1_new):int(y2_new),int(x1_new):int(x2_new)]

                x1_crop,y1_crop = (int(x1 - x1_new), int(y1 - y1_new))
                x2_crop,y2_crop = (int(x2 - x1_new), int(y2 - y1_new))

                x1_crop = int(x1_crop * (600 / crop_frame_evemt.shape[1]))
                y1_crop = int(y1_crop * (600 / crop_frame_evemt.shape[0]))
                x2_crop = int(x2_crop * (600 / crop_frame_evemt.shape[1]))
                y2_crop = int(y2_crop * (600 / crop_frame_evemt.shape[0]))

                file2write = open(f"{data_path}/{camera_id}/{int(coords[4])}/alert_image.txt",'w')

                file2write.write(str([x1_crop,y1_crop,x2_crop,y2_crop]))
                file2write.close()

                crop_frame_evemt = cv2.resize(crop_frame_evemt,(600,600))

                cv2.imwrite(crop_event_save_path, crop_frame_evemt)
                # cv2.imshow("crop_event",crop_frame_evemt)
                # cv2.waitKey(1) 

            # CAMERAS_DATA[camera_id]["face_track"][int(coords[4])] = align_faces

            
            recognition_run = False
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 190, 254), 9)
            # cv2.putText(frame, str(int(coords[4])), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 192, 203), 3)

            # x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            crop_frame = imutils.rotate(crop_frame, angle=-(roll))


            crop_frame = cv2.resize(crop_frame,(160,160))



            file_name_detect = "P"+str(time.time()).replace('.', '') +" "+str(align_faces)
            path_save =f"{data_path}/{camera_id}/{int(coords[4])}/{file_name_detect}.png"
            # s1 = time.time()

            if int(coords[4]) in CAMERAS_DATA[camera_id]["face_save_limit"] :
                CAMERAS_DATA[camera_id]["face_save_limit"][int(coords[4])] +=1
            
            else:
                CAMERAS_DATA[camera_id]["face_save_limit"][int(coords[4])] = 1

            if CAMERAS_DATA[camera_id]["face_save_limit"][int(coords[4])] < 60:
                cv2.imwrite(path_save, crop_frame)

            

            if len(CAMERAS_DATA[camera_id]["face_track"]) > 100:
                for _ in CAMERAS_DATA[camera_id]["face_track"]:
                    CAMERAS_DATA[camera_id]["face_track"].pop(_)
                    break
            
            if len(CAMERAS_DATA[camera_id]["face_save_limit"]) > 200:
                for _ in CAMERAS_DATA[camera_id]["face_save_limit"]:
                    CAMERAS_DATA[camera_id]["face_save_limit"].pop(_)
                    break

            # print(time.time()-s1)

        # cv2.imshow("test1",cv2.resize(frame,(1000,700)))
        # cv2.waitKey(1) 
        # print()
        # print(time.time()-s1)
    except Exception as e:
        print(traceback.print_exc())
        pass
    return recognition_run


def face_analysis(df):
    doubt_full_status = False
    min_index = df['best_distance'].idxmin()
    if df.loc[min_index]['best_distance'] > 0.24 :
        # print("unknown")
        # print(df.loc[min_index])
        return df.loc[min_index],doubt_full_status
    elif df.loc[min_index]['best_distance'] < 0.205:
        # print("known")
        # print(df.loc[min_index])
        return df.loc[min_index],doubt_full_status

    else:
        doubt_full_status = True
        # mask = (df['best_distance'] > 0.205) & (df['best_distance'] < 0.24)
        # result_df = df[mask]
        # unique_values = result_df['reco_name'].unique()


        # data = {'file_name': [],
        #     'reco_name': [],
        #     'best_distance': []}
        # df_data = pd.DataFrame(data)
        
        # for unique_names in unique_values:

        #     unique_values = result_df['reco_name'] == unique_names
        #     minimum_df = result_df[unique_values]

        #     min_index = minimum_df['best_distance'].idxmin()
        #     new_row = minimum_df.loc[min_index]


        #     df_data = df_data.append(new_row, ignore_index=True)


        # print("doubtfull")
        # print(df_data)
        return df.loc[min_index],doubt_full_status
    
def recognition_model(frame):
    def findDistance(row):
        distance_metric = row["distance_metric"]
        img2_representation = row["embedding"]
        distance = 1000  # initialize very large value
        if distance_metric == "cosine":
            distance = findCosineDistance(img1_representation, img2_representation)
        elif distance_metric == "euclidean":
            distance = findEuclideanDistance(img1_representation, img2_representation)
        elif distance_metric == "euclidean_l2":
            distance = findEuclideanDistance(l2_normalize(img1_representation), l2_normalize(img2_representation))

        return distance
    # print(cv2.Laplacian(frame, cv2.CV_64F).var())
    # if cv2.Laplacian(frame, cv2.CV_64F).var() < 200 :  # savio
    # if 0:  # savio
    #     print("skip")
    #     # pass
    #     return "blurr",100
    # else:
        # print("in reco")

    mean, std = frame.mean(), frame.std()
    frame = (frame - mean) / std

    batch_process_stack = np.expand_dims(frame, axis=0)
    label = "Unknown"
    best_distance = 100
    # print("in")
    # print("EMBEDDING_DICTIONARY",EMBEDDING_DICTIONARY[camera_id]["df"].shape[0])
    if ALL_DF_2["df"].shape[0] > 0:
        # print("in reco 2")

        # label = "unknown"
        img1_representation = optimized_model_reco(batch_process_stack)
        # df = EMBEDDING_DICTIONARY[camera_id]["df"].copy()
        df = ALL_DF_2["df"].copy()
        # status = "45Unauthorized"
        best_distance = 100
        # print(df)
        df["distance"] = df.apply(findDistance, axis=1)
        df = df.sort_values(by=["distance"])
        candidate = df.iloc[0]
        employee_name = candidate["employee"]
        best_distance = candidate["distance"]
        # status = candidate["status"]
        # print(best_distance)
        if best_distance <= 0.24:
            # print("in reco 3")

            label = employee_name.split("/")[-1].replace(".jpg", "")
            # print(label,best_distance)

        return label,best_distance
        

    else:
        return "Unknown",best_distance



def refresh_face_ids(camera_id):
    try:
        global CAMERAS_DATA

        if len(CAMERAS_DATA[camera_id]["dict_add_id"]) > 60:
            for i1, j1 in zip(CAMERAS_DATA[camera_id]["dict_add_id"], CAMERAS_DATA[camera_id]["dict_add_id_known"]):
                CAMERAS_DATA[camera_id]["dict_add_id"].pop(i1)
                CAMERAS_DATA[camera_id]["dict_add_id_known"].pop(j1)
                break

            if len(CAMERAS_DATA[camera_id]["events_list"]) > 60:
                for k1, l1 in zip(CAMERAS_DATA[camera_id]["events_list"], CAMERAS_DATA[camera_id]["filter"]):
                    CAMERAS_DATA[camera_id]["events_list"].pop(k1)
                    CAMERAS_DATA[camera_id]["filter"].pop(l1)
                    break

    except:
        print(traceback.print_exc())

stime = time.time()



#changes
########################             common API Endpoints  ######################################
@app.route('/update_pipeline_service_data/',methods=['POST'])
def update_pipeline_service_data():
    global CAMERAS_DATA
    try:
        
        SLEEP_TIME = 3
        data = request.get_json()
        # print(data)

        # CAMERAS_DATA = {}
        if type(data['path']) == str:
            data['path'] = [data['path']]

        for i, path in enumerate(data['path']):
            path_components = path.strip('[]').split('][')
            idx = int(path_components[0])
            if idx not in CAMERAS_DATA:
                CAMERAS_DATA[idx] = {}
                cam_data = fetch_camera_data(idx,DJANGO_API_ENDPOINT)

                CAMERAS_DATA[idx]["sort_objects"] = Sort(max_age=1, min_hits=0, iou_threshold=0.1)
                CAMERAS_DATA[idx]["dict_add_id"] = {}
                CAMERAS_DATA[idx]["dict_add_id_known"] = {}
                CAMERAS_DATA[idx]["events_list"] = {}
                CAMERAS_DATA[idx]["filter"] = {}
                CAMERAS_DATA[idx]["active_status"] = True
                CAMERAS_DATA[idx]["skip_counter"] = 0
                CAMERAS_DATA[idx]["classes"] = set()
                CAMERAS_DATA[idx]["false_check"] = 0
                CAMERAS_DATA[idx]["face_track"] = {}
                CAMERAS_DATA[idx]["face_save_limit"] = {}

                
                
                if cam_data:
                    CAMERAS[idx] = cam_data
                
            current_level = CAMERAS_DATA[idx]
            if len(path_components) > 1:
                if 'relation_fields' not in current_level:
                    current_level['relation_fields'] = {}
                current_level = current_level['relation_fields']
                for component in path_components[1:]:
                    if component not in current_level:
                        current_level[component] = {}
                    current_level = current_level[component]
                try:
                    current_level[data['object_id']].update(data['changed_fields'])
                except:
                    current_level[data['object_id']] = data['changed_fields']
                    
                print(f"if block")
            else:
                print(f"else block")
                CAMERAS_DATA[idx].update(data['changed_fields'])

        
            if "extra_fields" not in CAMERAS_DATA[idx]:
                CAMERAS_DATA[idx]["extra_fields"] = {}

            if "service_activation_schedule" not in CAMERAS_DATA[idx]:
                CAMERAS_DATA[idx]["service_activation_schedule"] = []
                SLEEP_TIME = 0

        res = ''
        for key in CAMERAS_DATA:
            res += f'{CAMERAS_DATA[key]}<br><br>'

        return f"""
            {'*'*25 + '  Service_Name = ' + SERVICE_NAME + '  ' + '*'*25}<br>
            CAMERAS = {CAMERAS} <br>
            CAMERAS_DATA = {CAMERAS_DATA} <br>
            SLEEP_TIME = {SLEEP_TIME} <br>
        
        """

    except Exception as e:
        print(f"update_pipeline_service_data Exception occured due to {e}")
        SLEEP_TIME = 0


def get_all_service_cameras(DJANGO_API_ENDPOINT,SERVICE_NAME):
    try:
        while True:
            try:
                SLEEP_TIME = 5
                global CAMERAS_DATA,CAMERAS
                data=post(f'{DJANGO_API_ENDPOINT}/service_app/get_data/',json={"class_name":"CameraServiceConfig","service_name":SERVICE_NAME},headers=get_api_headers()).json()
                print(f"data = {data} and SERVICE_NAME = {SERVICE_NAME}")
                for i in data:
                    try:
                        CAMERAS_DATA[i['camera_id']]=i

                        CAMERAS_DATA[i['camera_id']]["sort_objects"] = Sort(max_age=1, min_hits=0, iou_threshold=0.1)
                        CAMERAS_DATA[i['camera_id']]["dict_add_id"] = {}
                        CAMERAS_DATA[i['camera_id']]["dict_add_id_known"] = {}
                        CAMERAS_DATA[i['camera_id']]["events_list"] = {}
                        CAMERAS_DATA[i['camera_id']]["filter"] = {}
                        CAMERAS_DATA[i['camera_id']]["active_status"] = True
                        CAMERAS_DATA[i['camera_id']]["skip_counter"] = 0
                        CAMERAS_DATA[i['camera_id']]["classes"] = set()
                        CAMERAS_DATA[i['camera_id']]["false_check"] = 0
                        CAMERAS_DATA[i['camera_id']]["face_track"] = {}
                        CAMERAS_DATA[i['camera_id']]["face_save_limit"] = {}


                    


                        
                        cam_data = fetch_camera_data(i['camera_id'],DJANGO_API_ENDPOINT)
                        print(f"cam_data = {cam_data}")
                        if cam_data:
                            CAMERAS[i['camera_id']] = cam_data
                    except:
                        pass

                print(CAMERAS_DATA)
                break
            
            except:
                pass

        
    except Exception as e:
        print(f"get_all_service_cameras error sue to {e}")
    SLEEP_TIME = 0

########################             /common API Endpoints  #####################################

#################################### FR Custom API's #############################################
@app.route("/multiple_angle_face_validation/", methods=["POST"])
def multiple_angle_face_validation():
    try:
        SLEEP_TIME = 5
        data = request.get_json()
        byte_data = b64decode(data["capturedFace"])
        face = cv2.imdecode(frombuffer(byte_data, uint8), cv2.IMREAD_COLOR)
        response, valid_face = face_angle(face, data["faceAngle"])
        # response , valid_face = 'accepted', True
        if valid_face:
            SLEEP_TIME = 0

            return Response(f"{response}", status=200, mimetype="application/json")
        else:
            SLEEP_TIME = 0
            return Response(f"{response}", status=400, mimetype="application/json")
    except Exception as e:
        SLEEP_TIME = 0
        # pipeline_error_logger.error(f"function_name : face_angle_verification , Error : {e}")
        return Response(f"Error occured at the multiple_angle_face_validation {e}", status=400, mimetype="application/json")

#---------------------------------------------------------------------------
def create_image_encoding(img, encoding_version=1):
    input_shape_y, input_shape_x = 160, 160

    if encoding_version == 1:
        try:
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return face_recognition.face_encodings(img)[0]
        except:
            pass

    if encoding_version >= 2:
        try:
            # boxes, _, _ = predictor.predict(img, candidate_size / 2, threshold1)
            boxes = face_model.predict(img,classes=0,verbose=False, imgsz=[480],conf = 0.7)[0].boxes.cpu().numpy().data

            x1, y1, x2, y2,_,_ = boxes[0]
            y1 = int((y2-y1)*0.15)+y1
            img = img[int(y1) : int(y2), int(x1) : int(x2)]
            img = cv2.resize(img, (input_shape_y, input_shape_x))

            img = np.expand_dims(img, axis=0)

            mean, std = img.mean(), img.std()
            img = (img - mean) / std
            img_representation = optimized_model_reco(img)
            x = img_representation.tobytes()
            return x
        except Exception as e:
            print(f"Exception is {e}")
            return f"Exception is {e}"

#---------------------------------------------------------------------
@app.route("/create_face_embedding/", methods=["POST"])
def create_face_embedding():
    try:
        SLEEP_TIME = 5
        image_path = request.files["file"].read()
        # print(image_path)
        image = cv2.imdecode(frombuffer(image_path, uint8), cv2.IMREAD_COLOR)
        face_encode = str(create_image_encoding(image, encoding_version=2))
        # face_encode='functionality not implemented'
        return {"face_encode": face_encode}
    except:
        SLEEP_TIME = 0


#################################### /FR Custom API's ############################################
#changes




reco_wait_frames = 30

if __name__ == "__main__":
    try:
        shutil.rmtree(data_path)
        os.mkdir(data_path)


        #1. Initialzing the HTTP SERVER
        data = {'host':"0.0.0.0",'port':8080,'service_name':SERVICE_NAME}
        http_server = Thread(target=run_http_server,args=(app,data,))
        http_server.start()

        get_all_service_cameras(DJANGO_API_ENDPOINT,SERVICE_NAME)#changes

        # get_all_service_cameras()
        update_all_df_and_embedding_dict(DJANGO_API_ENDPOINT)


        SERVICE_MANAGER_PRESENT = True
        print("----------------------warm up start ------------------------")

        for _ in range(20):
            frame = np.random.rand(1290,2000,3)
            frame = cv2.resize(frame, (width, height))
            s1 = time.time()

            # predictor.predict(frame, candidate_size / 2, threshold1)
            print(time.time()-s1)

            custom_face = cv2.resize(frame, input_shape)

            batch_process_stack = np.expand_dims(custom_face, axis=0)
            s1 = time.time()
            img1_representation = optimized_model_reco(batch_process_stack)
            print(time.time()-s1)

            print(_)

        print("----------------------warm up done ------------------------")
        global SERVICE_CONFIG_DATA

        SERVICE_CONFIG_DATA  = set_parameter_with_servce_name(DJANGO_API_ENDPOINT)#changes


        #2. Service  Code Goes Here
        frame_counter = 0
        while True:
            try:
                if time.time()-stime > 200:
                    SERVICE_MANAGER_PRESENT = check_heart_beat(DJANGO_API_ENDPOINT = DJANGO_API_ENDPOINT)#changes

                    # SERVICE_MANAGER_PRESENT = check_heart_beat()
                    stime = time.time()
                if SERVICE_MANAGER_PRESENT:

                
                    delet_unsed_cam = set()
                    for camera_id in CAMERAS:
                        delet_unsed_cam.add(camera_id)

                        try:
                            if 'active_status' in CAMERAS_DATA[camera_id] and not CAMERAS_DATA[camera_id]['active_status']:
                                continue
                            
                            if CAMERAS_DATA[camera_id]["skip_counter"] % 1 == 0:
                                # s1 = time.time()
                                ret,frame = CAMERAS[camera_id]['cap'].read()
                                recognition_run = True


                                if ret :

                                    recognition_run = face_detector(camera_id, frame)
                                    if not recognition_run:
                                        CAMERAS_DATA[camera_id]["false_check"] = 0
                                    else:
                                        CAMERAS_DATA[camera_id]["false_check"] += 1 
                                        if CAMERAS_DATA[camera_id]["false_check"] >1000:
                                            CAMERAS_DATA[camera_id]["false_check"] = reco_wait_frames +5
                                

                                else :
                                    CAMERAS[camera_id]['cap'] = cv2.VideoCapture(CAMERAS[camera_id]['rtsp_url'])

                                if (CAMERAS_DATA[camera_id]["false_check"] > reco_wait_frames) and (CAMERAS_DATA[camera_id]["false_check"]%5==0):
                                    check_img = False

                                    # folder_path = f"/home/dnw/Desktop/savio/product_2.0/face-recognition/images/{0}"
                                    folder_path = f"{data_path}/{camera_id}"
                                    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                                    file_info_list = [(file_name[:-4], os.path.getctime(os.path.join(folder_path, file_name))) for file_name in files]
                                    sorted_files = sorted(file_info_list, key=lambda x: x[1])



                                    for i in sorted_files:
                                        crop_dir_path = folder_path+"/"+str(i[0])

                                    # for crop_dir_path in glob.glob(f"{data_path}/{camera_id}/*"):
                                        for crop_path in glob.glob(crop_dir_path+"/P*.png"):
                                            
                                            # print(crop_path)
                                            frame = cv2.imread(crop_path)
                                            label,best_distance = recognition_model(frame)
                                            # print(label,best_distance)
                                            csv_path = crop_dir_path+"/extracted.csv"
                                            # print("csv_path",csv_path)
                                            crop_df = pd.read_csv(csv_path)
                                            file_name = os.path.basename(crop_path)

                                            new_row = {'file_name': file_name[1:], 'reco_name': label, 'best_distance': best_distance}
                                            
                                            crop_df = crop_df.append(new_row, ignore_index=True)
                                            crop_df.to_csv(csv_path, index=False)
                                            # print(crop_path)
                                            # print("r",crop_dir_path+"/"+file_name[1:])
                                            os.rename(crop_path, crop_dir_path+"/"+file_name[1:])
                                            check_img = True
                                            break
                                        break
                                    # if CAMERAS_DATA[camera_id]["false_check"]%30==0:
                                    if check_img:

                                    # if CAMERAS_DATA[camera_id]["false_check"] > reco_wait_frames:

                                        service_id = CAMERAS_DATA[camera_id]['service_id']
                                        user_id = CAMERAS_DATA[camera_id]['user_id']
                                        camera_name = CAMERAS[camera_id]['camera_name']
                                        service_name = CAMERAS_DATA[camera_id]['service_name']

                                        folder_path = f"{data_path}/{camera_id}"
                                        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                                        file_info_list = [(file_name[:-4], os.path.getctime(os.path.join(folder_path, file_name))) for file_name in files]
                                        sorted_files = sorted(file_info_list, key=lambda x: x[1])

                                        for i in sorted_files:
                                        # for crop_dir_path in glob.glob(f"/home/dnw/Desktop/savio/product_2.0/face-recognition/images/{0}/*"):
                                            # print("crop_dir_path",crop_dir_path)
                                            crop_dir_path = folder_path+"/"+str(i[0])

                                        # for crop_dir_path in glob.glob(f"{data_path}/{camera_id}/*"):
                                            all_processed = True
                                            for crop_path in glob.glob(crop_dir_path+"/P*.png"):
                                                all_processed = False
                                                break
                                            if all_processed:
                                                data_frame_face = pd.read_csv(crop_dir_path + "/extracted.csv")
                                                if len(data_frame_face) < 4 :
                                                    if os.path.exists(crop_dir_path):
                                                        shutil.rmtree(crop_dir_path)
                                                        os.remove(crop_dir_path+".txt")
                                                    continue
                                                final_results,doubt_full_status = face_analysis(data_frame_face)
                                                # print(final_results)
                                                frame_send = cv2.imread(crop_dir_path +"/alert_image.jpg")

                                                modification_time = os.path.getmtime(crop_dir_path +"/alert_image.jpg")
                                                formatted_time = str(datetime.datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %I:%M:%S %p'))
                                                
                                                file2write=open(crop_dir_path +"/alert_image.txt",'r')
                                                bounding_box_send = json.loads(file2write.readline())
                                                xx1, yy1, xx2, yy2 = bounding_box_send
                                                file2write.close()


                                                if os.path.exists(crop_dir_path):
                                                    shutil.rmtree(crop_dir_path)
                                                    os.remove(crop_dir_path+".txt")
                                                name_person = final_results[1]
                                                status_person = "Unauthorized"
                                                # if final_results[1] == "unknown":

                                                # d2[d2["employee"] == "sam"].head(1)["status"][0]
                                                unique_id = "-"
                                                if len(EMBEDDING_DICTIONARY) and (EMBEDDING_DICTIONARY[camera_id]["df"].shape[0] > 0):
                                                    if (EMBEDDING_DICTIONARY[camera_id]["df"]["employee"].eq(name_person)).any():
                                                        user_row = EMBEDDING_DICTIONARY[camera_id]["df"][EMBEDDING_DICTIONARY[camera_id]["df"]["employee"] == name_person].head(1)
                                                        status_person = user_row["status"][0]
                                                        unique_id = user_row["unique_id"][0]
                                                    # status_person = event_df[event_df["employee"] == name_person]["status"].iloc[0]
                                                status = f"{name_person}({status_person}) Person Detected."
                                                if name_person == "Unknown":
                                                    status = f"{name_person} Person Detected."
                                                  
                                                alert = Alert(status,camera_id = camera_id,service_id = service_id,user_id = user_id,camera_name=camera_name,service_name=service_name,ALERT_SEND_URL = ALERT_SEND_URL)
                                                # alert.bbox = json.dumps([res_bbox])
                                                class_data = []
                                                doubtful_data = []


                                                alert.unique_id = json.dumps("-")
                                                # alert.unique_id(str(unique_id))



                                                if doubt_full_status:
                                                        doubtful_data.append({"class_name":name_person,"value":status_person, "value_type":"str"  ,"unique_id":str(unique_id) })
                                                        
                                                        doubtful_data.append({"class_name":"Unknown","value":"Unknown", "value_type":"str"  ,"unique_id":"-" })
                                                        class_data.append({"class_name":name_person,"value":status_person,"value_type":"str", "unique_id":str(unique_id)})
                                                else:

                                                    class_data.append({
                                                        "class_name":name_person,"value":status_person,"value_type":"str", "unique_id":str(unique_id)
                                                    })

                                                    # alert.unique_id(str(unique_id))
                                                    alert.unique_id = json.dumps("-")

                                                alert.message_details = json.dumps([name_person])



                                                alert.classes = json.dumps(class_data)

                                                alert.doubtful = json.dumps(doubtful_data)

                                                alert.edited = json.dumps(False)
                                                hh,ww,_ = frame_send.shape

                                                bbox_output = [{'top_left':(xx1, yy1), 'top_right':(xx2, yy1), 'bottom_left':(xx1, yy2), 'bottom_right':(xx2, yy2),'class':""}]
                                                res_bbox = {'bbox':bbox_output,'image_size':[hh,ww],'image_selection_bbox':[]}

                                                alert.bbox = json.dumps([res_bbox])
                                                alert.alert_date = formatted_time
                                                # alert.alert_date = json.dumps(formatted_time)

                                                alert.send(frame_send)



                                            break
                                

                                if CAMERAS[camera_id]['is_watch']:
                                    imshow(f"{camera_id}", frame) 

                                    waitKey(1)
                            else:
                                ret = CAMERAS[camera_id]['cap'].grab()

                            CAMERAS_DATA[camera_id]["skip_counter"] += 1
                            if CAMERAS_DATA[camera_id]["skip_counter"] >= 100000:
                                CAMERAS_DATA[camera_id]["skip_counter"] = 0

                        except Exception as e:
                            if not ret:
                                CAMERAS[camera_id]['cap'] = cv2.VideoCapture(CAMERAS[camera_id]['rtsp_url'])
                            print(traceback.print_exc())
                            print(f"unable to process camera of camera_id = {camera_id} due to {e}")

                        refresh_face_ids(camera_id)

                    for un_cam in glob.glob("{data_path}/*"):
                        if int(un_cam[0].split("/")[-1]) not in delet_unsed_cam:
                            shutil.rmtree(un_cam)


                    # for un_cam in delet_unsed_cam:

                    #     new_cam_path = f"{data_path}/{camera_id}"

                    #     if not os.path.isdir(new_cam_path):
                    #         os.mkdir(new_cam_path)
                # frame_counter += 1

                if len(CAMERAS) == 0 or not SERVICE_MANAGER_PRESENT:
                    sleep(3)

                # if frame_counter >= 100000:
                #     # SERVICE_MANAGER_PRESENT = check_heart_beat()
                #     frame_counter = 0

                sleep(SLEEP_TIME)

            except :
                print(traceback.print_exc())

    except Exception as e:
        print(f"unable to run the {SERVICE_NAME} program due to {e}")
