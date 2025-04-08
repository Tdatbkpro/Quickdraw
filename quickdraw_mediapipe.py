import mediapipe as mp
import cv2
import numpy as np
from src.utils import *
from src.config import *
import torch
from argparse import ArgumentParser
from src.model import QuickDraw
from collections import deque

def get_args():
    parse = ArgumentParser()
    parse.add_argument("--img-size", "-is", type=int, default= 28)
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

    points = deque(maxlen=512)
    canvas = np.zeros((480,640,3), dtype=np.uint8)

    class_images = get_images("images", CLASSES)
    is_drawing = False
    is_shown = False

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            flags, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if flags:
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame)

                frame.flags.writeable = True
                frame =  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[12].y < \
                                hand_landmarks.landmark[11].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
                                    mp_drawing.draw_landmarks(
                                        frame,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                                    if len(points):
                                         is_drawing = False
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
                                            cropped_canvas = canvas_gs[min_y:max_y, min_x:max_y]
                                            cropped_canvas = cv2.resize(cropped_canvas, (args.img_size, args.img_size))
                                            cropped_canvas = np.array(cropped_canvas, dtype=np.float32)[None,None,:,:]
                                            cropped_canvas = torch.from_numpy(cropped_canvas)
                                            output = model(cropped_canvas)
                                            probs = softmax(output)
                                            idx = torch.argmax(probs)
                                            acc_perc = probs[0][idx]*100
                                            points = deque(maxlen=512)
                                            canvas = np.zeros((480,640,3), dtype=np.uint8)
                                    else:
                                         is_drawing = True
                                         is_shown = False
                                         points.append((int(hand_landmarks.landmark[8].x * 640), int(hand_landmarks.landmark[8].y * 480)))
                                         for i in range(len(points)-1):
                                            cv2.line(frame, points[i], points[i+1],(0,255,0),2 ) 
                                            cv2.line(canvas, points[i - 1], points[i], args.line_colour, args.line_size)
                                   
                                    if not is_drawing and is_shown:
                                        cv2.putText(frame, "You are drawing {}% :".format(acc_perc), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), cv2.LINE_AA)
                                        frame[5:65, 490:550] = get_overlay(frame[5:65, 490:550], class_images[idx], (60,60))

                cv2.imshow("Quick Draw by Mediapipe Hand", frame)
                if cv2.waitKey(5) & 0xFF == 27:
                     break
        cap.release()    





                                              
                            

