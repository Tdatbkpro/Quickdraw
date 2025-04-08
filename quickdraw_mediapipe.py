import mediapipe as mp
import cv2
from random import randint
import numpy as np
import time
from src.utils import *
from src.config import *
import torch
from argparse import ArgumentParser
from src.model import QuickDraw
from collections import deque

def get_args():
    parse = ArgumentParser()
    parse.add_argument("--img-size", "-is", type=tuple, default= (28,28))
    parse.add_argument("--line-colour", "-lc", type=tuple, default=(255,255,255))
    parse.add_argument("--line-size", "-ls", type=int, default=5)
    parse.add_argument("--text-size", "-ts", type=int , default=20)
    parse.add_argument("--font-style", "-fs", type=str, default="Arial")
    parse.add_argument("--checkpoint", "-c", type=str, default="trained_models/last.pt")
    args = parse.parse_args()

    return args
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    model = QuickDraw(num_classes=len(CLASSES)).to(device=device)
    softmax = torch.nn.Softmax()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    else:
        print("No existed checkpoint")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    points = deque(maxlen=1024)
    canvas = np.zeros((480,640,3), dtype=np.uint8)

    drawing_time = 20   
    game_start = False
    class_images = get_images("images", CLASSES)
    is_drawing = False
    is_shown = False
    pause_drawing = False
    acc_perc = None
    idx = None
    ran = None
    max_distance = 30 
    score = 0
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            key = cv2.waitKey(3)
            
            flags, frame = cap.read()
            frame = cv2.flip(frame, 1)
            canvas_h, canvas_w = canvas.shape[:2]
            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / canvas_w
            scale_y = frame_h / canvas_h


            if flags:
                if not game_start:
                    start_prompt_time = time.time()
                    ran = randint(0, len(CLASSES)-1)
                    while time.time() - start_prompt_time < 100:
                        flags, frame = cap.read()
                        frame = cv2.flip(frame, 1)
                        
                        frame_h, frame_w = frame.shape[:2]

                        cv2.putText(frame, "Draw:", (frame_w//2 - 100, frame_h//2 - 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 140, 255), 4, cv2.LINE_AA)
                        cv2.putText(frame, "{}".format(CLASSES[ran]), (frame_w//2 - 80, frame_h//2),
                                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 140, 255), 4, cv2.LINE_AA)
                        cv2.putText(frame, "in under 20 seconds", (frame_w//2 - 200, frame_h//2 + 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 4, cv2.LINE_AA)

                        cv2.imshow("Quick Draw by Mediapipe Hand", frame)
                        if cv2.waitKey(1) & 0xFF == 32:
                            game_start = True
                            drawing_time = 20  # Giới hạn vẽ 20 giây
                            start_time = time.time()
                            break
                elapsed = int(time.time() - start_time)
                remaining = drawing_time - elapsed

                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame)

                frame.flags.writeable = True
                frame =  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    cv2.putText(frame, "{}".format(remaining), (frame_w - 100, frame_h - 10),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.3, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_4)
                    cv2.putText(frame, "Score : {}".format(score), (10, frame_h - 10),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.3, color=(0, 128, 255), thickness=2, lineType=cv2.LINE_4)
                    for hand_landmarks in results.multi_hand_landmarks:
                        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[12].y < \
                                hand_landmarks.landmark[11].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y and hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y or remaining <= 0:
                                    if len(points):
                                         
                                         is_drawing = False
                                         pause_drawing = False
                                         is_shown = True
                                         canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                                         canvas_gs = cv2.medianBlur(canvas_gs, 9) # Tranh nhieu anh
                                         canvas_gs = cv2.GaussianBlur(canvas_gs,(3,3), 0)
                                         ys, xs = np.nonzero(canvas_gs) # lay nhung cai khong phai mau den
                                         if len(ys) and len(xs):
                                            is_drawing = False
                                            min_y = np.min(ys)
                                            min_x = np.min(xs)
                                            max_y = np.max(ys)
                                            max_x = np.max(xs)
                                            cropped_canvas = canvas_gs[min_y:max_y, min_x:max_x]
                                            cropped_canvas = cv2.resize(cropped_canvas, args.img_size)
                                            cropped_canvas = np.array(cropped_canvas, dtype=np.float32)[None,None,:,:]
                                            cropped_canvas = torch.from_numpy(cropped_canvas).to(device)
                                            output = model(cropped_canvas)
                                            probs = softmax(output)
                                            idx = torch.argmax(probs,dim=1).item()
                                            print(idx)
                                            acc_perc = probs[0][idx]*100
                                            print(acc_perc)
                                            points.clear()
                                            canvas.fill(0)
                                            start_show_time = time.time()
                                            
                        
                        elif hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[7].y < hand_landmarks.landmark[6].y\
                             and hand_landmarks.landmark[12].y > hand_landmarks.landmark[11].y :
                        
                            index_x = int(hand_landmarks.landmark[8].x * canvas.shape[1])
                            index_y = int(hand_landmarks.landmark[8].y * canvas.shape[0])
                           
                            points.append((index_x, index_y))
                            mp_drawing.draw_landmarks(
                                        frame,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                            is_drawing = True
                            is_shown = False
                            for i in range(len(points)-1):
                                x1, y1 = points[i]
                                x2, y2 = points[i+1]
                                distance = ((x2-x1)**2 + (y2-y1)**2)**0.5 
                                if distance < max_distance:
                                    pt1 = (int(x1 * scale_x), int(y1 * scale_y))
                                    pt2 = (int(x2 * scale_x), int(y2 * scale_y))
                                    cv2.line(frame, pt1, pt2,(0,255,0),4) 
                                    cv2.line(canvas, points[i], points[i+1], args.line_colour, args.line_size)
                                else :
                                     continue
                        else:
                            mp_drawing.draw_landmarks(
                                        frame,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                        if not is_drawing and is_shown:
                            
                            cv2.putText(frame, "You are drawing {:.2f}% :".format(acc_perc), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2,cv2.LINE_AA)
                            overlay_img = cv2.resize(class_images[idx], (50,50))
                            frame[:50, frame.shape[1]-210:frame.shape[1]-160] = get_overlay(
                                frame[0:50, frame.shape[1]-210:frame.shape[1]-160], overlay_img, (50,50)
                            )
                            if time.time() - start_show_time >= 4:
                                game_start = False  # reset để chơi vòng tiếp theo nếu muốn
                                is_shown = False 
                            else:
                                if idx == ran :

                                    print(CLASSES[ran], CLASSES[idx])
                                    score += 1
                                    cv2.putText(frame, "Correct", (frame_w//2 - 100, frame_h//2),
                                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2.5, (0, 255, 0), 4, cv2.LINE_AA)
                                else:
                                    print(CLASSES[ran], CLASSES[idx])
                                    cv2.putText(frame, "Incorrect", (frame_w//2 - 100, frame_h//2),
                                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2.5, (0, 0, 255), 4, cv2.LINE_AA)
                        print(is_drawing, is_shown)
                cv2.imshow("Quick Draw by Mediapipe Hand", frame)
                if cv2.waitKey(5) & 0xFF == 27:
                     break
        cap.release()    





                                              
                            

