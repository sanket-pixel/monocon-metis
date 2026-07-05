import sys
import os
import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from python.model.backbone import DLA34
from python.model.neck import DLAUp
from python.model.head import MonoConHead
from python.model.preprocess import load_image_rgb, preprocess_image
from python.model.decode import decode_predictions
from python.utils.geometry_ops import extract_corners_from_bboxes_3d, points_cam2img



CKPT_PATH = "../../weights/monocon.pth"
IMAGE_PATH = "../../data/KITTI_sample/image_02/data/0000000342.png"
CALIB_PATH = "../../data/KITTI_sample/calib/calib_cam_to_cam.txt"

OUT_DIR = "../../output"

CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car']
CLASS_COLORS = {  # BGR for cv2
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
}

# 12 edges connecting the 8 box corners (from visualizer.py's convention)
BOX_EDGES = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2),
            (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))


def parse_calib_p2(calib_file: str) -> np.ndarray:
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        key, value = line.split(': ', 1)
        if key == 'P_rect_02':
            return np.array(value.split(), dtype=np.float32).reshape(3, 4)
    raise ValueError(f"'P_rect_02' not found in {calib_file}")


def load_all_weights(backbone, neck, head, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_state = ckpt["state_dict"]["model"]
    for name, module in [("backbone", backbone), ("neck", neck), ("head", head)]:
        prefix = f"{name}."
        sub_state = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
        missing, unexpected = module.load_state_dict(sub_state, strict=False)
        assert not missing and not unexpected, f"{name}: missing={missing}, unexpected={unexpected}"


def draw_2d_boxes(image: np.ndarray, bboxes_2d: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    out = image.copy().astype(np.uint8)
    for box, label in zip(bboxes_2d, labels):
        x1, y1, x2, y2, score = box.tolist()
        color = CLASS_COLORS[int(label.item())]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(out, f"{CLASS_NAMES[int(label.item())]} {score:.1f}",
                   (int(x1), max(int(y1) - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def draw_3d_boxes(image: np.ndarray, bboxes_3d: torch.Tensor, labels: torch.Tensor, P2: np.ndarray) -> np.ndarray:
    out = image.copy().astype(np.uint8)
    if len(bboxes_3d) == 0:
        return out

    corners_all = extract_corners_from_bboxes_3d(bboxes_3d)  # (N, 8, 3)

    for corners, label in zip(corners_all, labels):
        proj = points_cam2img(corners, P2)  # (8, 2)
        proj = proj.round().astype(int)

        color = CLASS_COLORS[int(label.item())]
        for start, end in BOX_EDGES:
            pt1 = tuple(proj[start])
            pt2 = tuple(proj[end])
            cv2.line(out, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
    return out


def draw_bev(bboxes_3d: torch.Tensor, labels: torch.Tensor, max_dist: int = 60, scale: int = 10) -> np.ndarray:
    R = max_dist * scale
    space = np.zeros((R, R * 2, 3), dtype=np.uint8)

    # Range rings + angle guide lines
    for theta in np.linspace(0, np.pi, 7):
        pt = (int(R - R * np.cos(theta)), int(R - R * np.sin(theta)))
        cv2.line(space, pt, (R, R), (255, 255, 255), 1, cv2.LINE_AA)
    for radius in np.linspace(0, R, 5)[1:]:
        cv2.circle(space, (R, R), int(radius), (255, 255, 255), 1, cv2.LINE_AA)

    if len(bboxes_3d) == 0:
        return space

    # (x, z, length, width, rotation_y) -> BEV plot coords
    bev = bboxes_3d[:, [0, 2, 3, 5, 6]].clone()
    bev[:, :-1] *= scale
    bev[:, 1] *= -1
    bev[:, :2] += R

    for row, label in zip(bev, labels):
        x, z, l, w, ry = row.tolist()
        box_pts = cv2.boxPoints(((x, z), (l, w), ry * 180 / np.pi))
        box_pts = np.intp(box_pts)
        color = CLASS_COLORS[int(label.item())]
        cv2.drawContours(space, [box_pts], -1, color, thickness=-1, lineType=cv2.LINE_AA)

    return space[:R]  # only show what's in front of the camera


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    backbone = DLA34()
    neck = DLAUp(backbone.get_out_channels(start_level=2), start_level=2)
    head = MonoConHead()
    load_all_weights(backbone, neck, head, CKPT_PATH)
    backbone.eval(); neck.eval(); head.eval()

    raw_img = load_image_rgb(IMAGE_PATH)
    input_tensor, ori_img, pad_shape = preprocess_image(raw_img)
    P2 = parse_calib_p2(CALIB_PATH)

    with torch.no_grad():
        feats = backbone(input_tensor)
        neck_out = neck(list(feats))[0]
        pred_dict = head(neck_out)

    pad_h, pad_w = pad_shape
    bboxes_2d, bboxes_3d, labels = decode_predictions(
        pred_dict, calib_P2_list=[P2], img_h=pad_h, img_w=pad_w, score_thres=0.3)

    b2d, b3d, lbl = bboxes_2d[0], bboxes_3d[0], labels[0]
    print(f"Drawing {len(b2d)} detections...")

    # Draw on the ORIGINAL (unpadded) image, since boxes were decoded at
    # padded-image scale but padding only added empty space bottom/right —
    # cropping back to ori_img's shape is safe and correct.
    ori_h, ori_w = ori_img.shape[:2]

    img_2d = draw_2d_boxes(ori_img[:ori_h, :ori_w], b2d, lbl)
    cv2.imwrite(os.path.join(OUT_DIR, "result_2d.png"), cv2.cvtColor(img_2d, cv2.COLOR_RGB2BGR))

    img_3d = draw_3d_boxes(ori_img[:ori_h, :ori_w], b3d, lbl, P2)
    cv2.imwrite(os.path.join(OUT_DIR, "result_3d.png"), cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR))

    img_bev = draw_bev(b3d, lbl)
    cv2.imwrite(os.path.join(OUT_DIR, "result_bev.png"), cv2.cvtColor(img_bev, cv2.COLOR_RGB2BGR))

    print(f"Saved to {OUT_DIR}/result_2d.png, result_3d.png, result_bev.png")