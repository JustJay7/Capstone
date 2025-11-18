import torch
import torch.nn.functional as F

def xywh_to_xyxy(boxes, img_size=640, normalize=False):
    """
    Convert [cx, cy, w, h] → [x1, y1, x2, y2]
    Set normalize=True if boxes are in [0,1] (YOLO format)
    """
    cx, cy, w, h = boxes.unbind(-1)
    if normalize:
        cx *= img_size
        cy *= img_size
        w *= img_size
        h *= img_size
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def bbox_iou(box1, box2, eps=1e-7):
    """
    box1: (N, 4), box2: (M, 4)
    Returns IoU: (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    tl = torch.max(box1[:, None, :2], box2[:, :2])  # top-left
    br = torch.min(box1[:, None, 2:], box2[:, 2:])  # bottom-right
    wh = (br - tl).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter

    return inter / (union + eps)

def mpd_loss(pred_xyxy, gt_xyxy):
    """Min Point Distance between box corners"""
    p_tl, p_br = pred_xyxy[:, :2], pred_xyxy[:, 2:]
    g_tl, g_br = gt_xyxy[:, :2], gt_xyxy[:, 2:]
    loss = F.smooth_l1_loss(p_tl, g_tl) + F.smooth_l1_loss(p_br, g_br)
    return loss
