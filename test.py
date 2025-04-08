import cv2, os, shutil, torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import numpy as np
from argparse import ArgumentParser
from src.model import QuickDraw
from src.config import CLASSES
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary

drawing = False
ix, iy = -1, -1

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", "-cp", type=str, default="trained_models/last.pt")
    parser.add_argument("--img-size", "-is", type=tuple, default=(28, 28))
    return parser.parse_args()

def paint_draw(event, x, y, flags, param):
    global ix, iy, drawing, image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(image, (ix, iy), (x, y), (255,255,255), 12)
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(image, (ix, iy), (x, y), (255,255,255), 12)

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image_path = "painted_image.jpg"

    cv2.namedWindow("Canvas")
    cv2.setMouseCallback("Canvas", paint_draw)

    model = QuickDraw(num_classes=len(CLASSES)).to(device)
    summary(model, input_data=(1, 28, 28))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print("✅ Loaded checkpoint")
    else:
        print("❌ No checkpoint provided")

    while True:
        cv2.imshow("Canvas", image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            cv2.imwrite(image_path, image)
            print("✅ Saved image to disk")

            # ----- Xử lý ảnh test -----
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1]
            x, y, w, h = cv2.boundingRect(thresh)
            img = img[y:y+h, x:x+w]
            # img = cv2.GaussianBlur(img, (3,3), 0)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)            
            img = img.astype(np.float32) / 255.0

            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 28, 28)
            plt.imshow(img_tensor.squeeze().cpu().numpy(), cmap="gray")
            plt.title("Image Sent to Model")
            plt.axis("off")
            plt.show()
            print(f"Image shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
            print(f"Min: {img_tensor.min().item():.4f}, Max: {img_tensor.max().item():.4f}")

            # ----- Dự đoán -----
            model.eval()
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                pred_class = CLASSES[pred_idx]
                print(f"\n🎯 Predicted: {pred_class} (Prob: {probs[0, pred_idx]*100:.2f})%")

            break
        elif key == ord("r"):
            image = np.zeros((480, 640, 3), dtype=np.uint8)


    cv2.destroyAllWindows()
