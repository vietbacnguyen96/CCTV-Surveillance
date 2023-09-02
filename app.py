from flask import Flask, render_template, Response, jsonify
import cv2
import requests
import base64
import json                    
import time
import threading

from utils.service.TFLiteFaceAlignment import * 
from utils.service.TFLiteFaceDetector import * 
from utils.functions import *

app = Flask(__name__)

# path = "/home/vkist1/frontend_facerec_VKIST/"
path = "./"

fd_0 = UltraLightFaceDetecion(path + "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fd_1 = UltraLightFaceDetecion(path + "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fa_0 = CoordinateAlignmentModel(path + "utils/service/weights/coor_2d106.tflite")
fa_1 = CoordinateAlignmentModel(path + "utils/service/weights/coor_2d106.tflite")

# url = 'http://192.168.0.100:5052/'
url = 'https://dohubapps.com/user/daovietanh190499/5000/'

api_list = [url + 'facerec', url + 'FaceRec_DREAM', url + 'FaceRec_3DFaceModeling', url + 'check_pickup']
api_index = 0
extend_pixel = 100
crop_image_size = 100

# hhsc hhsc2023
secret_key = "f37b4dd3-7496-40e9-aa6d-28ae1c4b6f70"

predict_labels = []

def face_recognize(frame):
    global api_index

    _, encimg = cv2.imencode(".jpg", frame)
    img_byte = encimg.tobytes()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    new_img_str = "data:image/jpeg;base64," + img_str
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}

    payload = json.dumps({"secret_key": secret_key, "img": new_img_str, 'local_register' : False})

    response = requests.post(api_list[api_index], data=payload, headers=headers, timeout=100)

    try:
        # for id, name, picker_name, profileID, picker_profile_face_id, timestamp in zip( 
        for id, name, profileID, timestamp in zip( 
                                                                                        response.json()['result']['id'],
                                                                                        response.json()['result']['identities'],
                                                                                        # response.json()['result']['picker_profile_names'],
                                                                                        response.json()['result']['profilefaceIDs'],
                                                                                        # response.json()['result']['pickerProfileFaceIds'],
                                                                                        response.json()['result']['timelines']
                                                                                        ):
            print('Server response', response.json()['result']['identities'])
            if id != -1:
                # response_time_s = time.time() - seconds
                # print("Server's response time: " + "%.2f" % (response_time_s) + " (s)")
                # print('picker_profile_face_id', picker_profile_face_id)
                cur_profile_face = None
                cur_picker_profile_face = None

                if profileID is not None:
                    cur_url = url + 'images/' + secret_key + '/' + profileID
                    cur_profile_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                    # cur_profile_face = cv2.resize(cur_profile_face, (crop_image_size, crop_image_size))
                    cur_profile_face = cv2.cvtColor(cur_profile_face, cv2.COLOR_BGR2RGB)

                    _, encimg = cv2.imencode(".jpg", cur_profile_face)
                    img_byte = encimg.tobytes()
                    img_str = base64.b64encode(img_byte).decode('utf-8')
                    cur_profile_face = "data:image/jpeg;base64," + img_str


                # if picker_profile_face_id is not None:
                #     cur_url = url + 'images/' + secret_key + '/' + picker_profile_face_id
                #     cur_picker_profile_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                #     # cur_picker_profile_face = cv2.resize(cur_picker_profile_face, (crop_image_size, crop_image_size))
                #     cur_picker_profile_face = cv2.cvtColor(cur_picker_profile_face, cv2.COLOR_BGR2RGB)

                    # _, encimg = cv2.imencode(".jpg", cur_picker_profile_face)
                    # img_byte = encimg.tobytes()
                    # img_str = base64.b64encode(img_byte).decode('utf-8')
                    # cur_picker_profile_face = "data:image/jpeg;base64," + img_str

                frame = cv2.resize(frame, (crop_image_size, crop_image_size))
                _, encimg = cv2.imencode(".jpg", frame)
                img_byte = encimg.tobytes()
                img_str = base64.b64encode(img_byte).decode('utf-8')
                new_img_str = "data:image/jpeg;base64," + img_str

                # predict_labels.append([id, name, picker_name, new_img_str, cur_profile_face, cur_picker_profile_face, timestamp])
                predict_labels.append([id, name, new_img_str, cur_profile_face, timestamp])

    except requests.exceptions.RequestException:
        print(response.text)

def get_frame_0():
    # Open the webcam stream
    webcam_0 = cv2.VideoCapture(0)

    frame_width = int(webcam_0.get(3))
    frame_height = int(webcam_0.get(4))

    prev_frame_time = 0
    new_frame_time = 0
    queue = []

    count = 0
    frequency = 4

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_0.read()
        # orig_image = cv2.flip(orig_image, 1)
        final_frame = orig_image.copy()
        if (count % frequency) == 0:
            count = 0

            temp_boxes, _ = fd_0.inference(orig_image)

            # Draw boundary boxes around faces
            draw_box(final_frame, temp_boxes, color=(125, 255, 125))

            # Find landmarks of each face
            temp_marks = fa_0.get_landmarks(orig_image, temp_boxes)

            # -------------------------------------- Draw landmarks of each face ---------------------------------------------
            # for bbox_I, landmark_I in zip(temp_boxes, temp_resized_marks):
            #     landmark_I = landmark_I * (1 / scale_ratio)
            #     draw_landmark(final_frame, landmark_I, color=(125, 255, 125))

            #     # Show rotated raw face image
            #     xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])
            #     xmin -= extend_pixel
            #     xmax += extend_pixel
            #     ymin -= 2 * extend_pixel
            #     ymax += extend_pixel

            #     xmin = 0 if xmin < 0 else xmin
            #     ymin = 0 if ymin < 0 else ymin
            #     xmax = frame_width if xmax >= frame_width else xmax
            #     ymax = frame_height if ymax >= frame_height else ymax

            #     face_I = orig_image[ymin:ymax, xmin:xmax]
            #     face_I = align_face(face_I, landmark_I[34], landmark_I[88])

            #     cv2.imshow('Rotated raw face image', face_I)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # ----------------------------------------------------------------------------------------------------------------

            for bbox_I, landmark_I in zip(temp_boxes, temp_marks):
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                xmin -= extend_pixel
                xmax += extend_pixel
                ymin -= extend_pixel
                ymax += extend_pixel

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                xmax = frame_width if xmax >= frame_width else xmax
                ymax = frame_height if ymax >= frame_height else ymax

                face_I = orig_image[ymin:ymax, xmin:xmax]
                rotated_face_I = align_face(face_I, landmark_I[34], landmark_I[88])

                # --------------------------------- Show rotated resized face image ----------------------------------------------
                # cv2.imshow('Rotated resized face image', rotated_face_I)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # ----------------------------------------------------------------------------------------------------------------

                queue = [t for t in queue if t.is_alive()]
                if len(queue) < 3:
                    queue.append(threading.Thread(target=face_recognize, args=(rotated_face_I,)))
                    queue[-1].start()

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))

            cv2.putText(final_frame, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

            # Convert the frame to a jpeg image
            ret, jpeg = cv2.imencode('.jpg', final_frame)

            # Return the image as bytes
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def get_frame_1():
    # Open the webcam stream
    webcam_1 = cv2.VideoCapture('rtsp://admin:pilot2214@192.168.50.14:554/Streaming/channels/1/')
    # webcam_1 = cv2.VideoCapture(0)

    frame_width = int(webcam_1.get(3))
    frame_height = int(webcam_1.get(4))

    prev_frame_time = 0
    new_frame_time = 0
    queue = []

    count = 0
    frequency = 4

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_1.read()
        # orig_image = cv2.flip(orig_image, 1)
        final_frame = orig_image.copy()
        if (count % frequency) == 0:
            count = 0

            temp_boxes, _ = fd_1.inference(orig_image)

            # Draw boundary boxes around faces
            draw_box(final_frame, temp_boxes, color=(125, 255, 125))

            # Find landmarks of each face
            temp_marks = fa_1.get_landmarks(orig_image, temp_boxes)

            # ---------------------------------------- Draw landmarks of each face ---------------------------------------------
            # for bbox_I, landmark_I in zip(temp_boxes, temp_resized_marks):
            #     landmark_I = landmark_I * (1 / scale_ratio)
            #     draw_landmark(final_frame, landmark_I, color=(125, 255, 125))

            #     # Show rotated raw face image
            #     xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])
            #     xmin -= extend_pixel
            #     xmax += extend_pixel
            #     ymin -= 2 * extend_pixel
            #     ymax += extend_pixel

            #     xmin = 0 if xmin < 0 else xmin
            #     ymin = 0 if ymin < 0 else ymin
            #     xmax = frame_width if xmax >= frame_width else xmax
            #     ymax = frame_height if ymax >= frame_height else ymax

            #     face_I = orig_image[ymin:ymax, xmin:xmax]
            #     face_I = align_face(face_I, landmark_I[34], landmark_I[88])

            #     cv2.imshow('Rotated raw face image', face_I)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # ----------------------------------------------------------------------------------------------------------------

            for bbox_I, landmark_I in zip(temp_boxes, temp_marks):
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                xmin -= extend_pixel
                xmax += extend_pixel
                ymin -= extend_pixel
                ymax += extend_pixel

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                xmax = frame_width if xmax >= frame_width else xmax
                ymax = frame_height if ymax >= frame_height else ymax

                face_I = orig_image[ymin:ymax, xmin:xmax]
                rotated_face_I = align_face(face_I, landmark_I[34], landmark_I[88])

                # ------------------------------- Show rotated resized face image ----------------------------------------------
                # cv2.imshow('Rotated resized face image', rotated_face_I)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # ----------------------------------------------------------------------------------------------------------------

                queue = [t for t in queue if t.is_alive()]
                if len(queue) < 3:
                    queue.append(threading.Thread(target=face_recognize, args=(rotated_face_I,)))
                    queue[-1].start()

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))

            cv2.putText(final_frame, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

            # Convert the frame to a jpeg image
            ret, jpeg = cv2.imencode('.jpg', final_frame)

            # Return the image as bytes
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_0')
def video_feed_0():
    return Response(get_frame_0(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_1')
def video_feed_1():
    return Response(get_frame_1(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    global predict_labels
    if len(predict_labels) > 3:
        predict_labels = predict_labels[-3:]
    newest_data = list(reversed(predict_labels))
    return jsonify({'info': newest_data})

if __name__ == '__main__':
    app.run(debug=True)
