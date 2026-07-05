import sys
import os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from python.model.backbone import DLA34
from python.model.neck import DLAUp
from python.model.head import MonoConHead
from python.model.preprocess import load_image_rgb, preprocess_image
from python.model.decode import decode_predictions


CKPT_PATH = "../../weights/monocon.pth"
IMAGE_PATH = "../../data/KITTI_sample/image_02/data/0000000342.png"
CALIB_PATH = "../../data/KITTI_sample/calib/calib_cam_to_cam.txt"


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


if __name__ == "__main__":
    # Build + load model
    backbone = DLA34()
    neck = DLAUp(backbone.get_out_channels(start_level=2), start_level=2)
    head = MonoConHead()
    load_all_weights(backbone, neck, head, CKPT_PATH)
    backbone.eval(); neck.eval(); head.eval()

    # Load + preprocess real image
    raw_img = load_image_rgb(IMAGE_PATH)
    print(f"Raw image shape: {raw_img.shape}")

    input_tensor, ori_img, pad_shape = preprocess_image(raw_img)
    print(f"Preprocessed input tensor shape: {tuple(input_tensor.shape)}")
    print(f"Padded shape used for decode: {pad_shape}")

    # Parse real calibration
    P2 = parse_calib_p2(CALIB_PATH)
    print(f"P2:\n{P2}")

    # Forward pass
    with torch.no_grad():
        feats = backbone(input_tensor)
        neck_out = neck(list(feats))[0]
        pred_dict = head(neck_out)

    print("\nRaw heatmap stats (before decode):")
    for c_idx, c_name in enumerate(['Pedestrian', 'Cyclist', 'Car']):
        heat = pred_dict['center_heatmap'][0, c_idx]
        print(f"  {c_name}: max={heat.max().item():.4f}, mean={heat.mean().item():.6f}")

    # Decode
    pad_h, pad_w = pad_shape
    bboxes_2d, bboxes_3d, labels = decode_predictions(
        pred_dict,
        calib_P2_list=[P2],
        img_h=pad_h,
        img_w=pad_w,
        score_thres=0.3)

    print(f"\nDetections found: {len(bboxes_2d[0])}")
    class_names = ['Pedestrian', 'Cyclist', 'Car']
    for i in range(len(bboxes_2d[0])):
        cls = class_names[labels[0][i].item()]
        score = bboxes_2d[0][i, -1].item()
        box2d = bboxes_2d[0][i, :4].tolist()
        box3d = bboxes_3d[0][i].tolist()
        print(f"  [{cls}] score={score:.3f}  2D box={[round(x,1) for x in box2d]}  3D box={[round(x,2) for x in box3d]}")
