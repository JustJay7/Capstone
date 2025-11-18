import onnxruntime as ort
import numpy as np
import cv2
import torch
import os

# =====================
# CONFIG
# =====================
onnx_path = r"D:\Capstone_research\models\model weights\generalist model weights\yolov8_crackdetection_selftrained\yolov8_crack_epoch_60_custom.onnx"
img_path = r"D:\Capstone_research\samples\road_crack_test.jpg"
anchors = np.array([
    [490, 192],
    [64, 50],
    [230, 63],
    [151, 196],
    [159, 499]
], dtype=np.float32)
img_size = 640
conf_thresh = 0.25
iou_thresh = 0.45


# =====================
# UTILITIES
# =====================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_threshold):
    """Basic NMS implementation"""
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def compute_iou(box1, boxes):
    inter_x1 = np.maximum(box1[0], boxes[:, 0])
    inter_y1 = np.maximum(box1[1], boxes[:, 1])
    inter_x2 = np.minimum(box1[2], boxes[:, 2])
    inter_y2 = np.minimum(box1[3], boxes[:, 3])
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    boxes_area = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    union = box1_area + boxes_area - inter_area
    return inter_area / np.maximum(union, 1e-6)

# =====================
# PREPROCESS
# =====================
img = cv2.imread(img_path)
h0, w0 = img.shape[:2]
img_resized = cv2.resize(img, (img_size, img_size))
inp = img_resized[..., ::-1].transpose(2, 0, 1)
inp = np.expand_dims(inp, 0).astype(np.float32) / 255.0

# =====================
# RUN ONNX INFERENCE
# =====================
sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
output = sess.run(None, {"images": inp})[0]
print(f"✅ ONNX output shape: {output.shape}")

# =====================
# DECODE OUTPUT
# =====================
# output shape: [1, 30, 20, 20] = [B, anchors*(5+cls), H, W]
B, C, H, W = output.shape
A = len(anchors)
out = output.reshape(B, A, -1, H, W)
out = out.transpose(0, 1, 3, 4, 2)[0]  # [A,H,W,6]
boxes = []
scores = []

for ai, anchor in enumerate(anchors):
    pred = out[ai]
    bx = sigmoid(pred[..., 0]) * img_size
    by = sigmoid(pred[..., 1]) * img_size
    bw = np.exp(pred[..., 2]) * anchor[0]
    bh = np.exp(pred[..., 3]) * anchor[1]
    conf = sigmoid(pred[..., 4])
    cls = sigmoid(pred[..., 5])
    score = conf * cls

    mask = score > conf_thresh
    bx, by, bw, bh, score = bx[mask], by[mask], bw[mask], bh[mask], score[mask]

    for x, y, w, h, s in zip(bx, by, bw, bh, score):
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append([x1, y1, x2, y2])
        scores.append(s)

if not boxes:
    print("⚠️ No detections above threshold.")
    exit()

boxes = np.array(boxes)
scores = np.array(scores)

# NMS
keep = nms(boxes, scores, iou_thresh)
boxes = boxes[keep]
scores = scores[keep]

# =====================
# VISUALIZE
# =====================
for (x1, y1, x2, y2), s in zip(boxes, scores):
    cv2.rectangle(img_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img_resized, f"{s:.2f}", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite("pred_result.jpg", img_resized)
print("✅ Saved result → pred_result.jpg")
