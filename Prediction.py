import os
import torch
import numpy as np
from PIL import Image
import argparse
import cv2

from model.cracknex import CrackNex  # 使用你的模型结构定义文件路径
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Compose
import torchvision.transforms.functional as TF


# === 自定义图像归一化（仅图像，不需要 mask） ===
def normalize_only_img(img):
    # 确保图像是三通道
    if img.mode != 'RGB':
        img = img.convert('RGB')  # 如果图像是灰度图或其他格式，转换为 RGB

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return TF.normalize(ToTensor()(img), mean=mean, std=std)


# === 加载图像并归一化 ===
def load_image(path):
    img = Image.open(path)  # 不进行 convert('RGB')，保留原始图像格式
    transform = Compose([
        normalize_only_img
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor, img  # 返回原始图像（不转换为 RGB）


# === 滑窗预测函数，限制 ROI 范围 ===
# 在原代码的基础上进行修改
def sliding_window_predict_roi(model, img_tensor, roi_coords, window_size=448, overlap=0.2, cls=1,
                               save_blocks_dir=None):
    _, _, H, W = img_tensor.shape
    x1, y1, x2, y2 = roi_coords
    roi_w, roi_h = x2 - x1, y2 - y1
    stride = int(window_size * (1 - overlap))

    final_mask = np.zeros((H, W), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        block_idx = 0  # 计数滑窗块
        # 从ROI的左上角开始，按照固定步长滑动窗口
        for top in range(y1, y2, stride):
            for left in range(x1, x2, stride):
                # 确保窗口大小为window_size，即使在边界处也要保持大小一致
                bottom = min(top + window_size, y2)
                right = min(left + window_size, x2)

                crop = img_tensor[:, :, top:bottom, left:right]
                pad_bottom = window_size - crop.shape[2]
                pad_right = window_size - crop.shape[3]
                if pad_bottom > 0 or pad_right > 0:
                    crop = F.pad(crop, (0, pad_right, 0, pad_bottom), mode="constant", value=0)

                # 仅使用查询图像进行推理
                pred = model(crop)  # 模型的前向推理
                pred = torch.argmax(pred, dim=1)
                pred_crop = pred[:, :bottom - top, :right - left].squeeze(0).cpu().numpy()
                pred_crop_bin = (pred_crop == cls).astype(np.uint8) * 255

                # 保存单独的mask
                if save_blocks_dir:
                    block_mask = Image.fromarray(pred_crop_bin)
                    block_mask.save(os.path.join(save_blocks_dir, f"block_{block_idx}_{top}_{left}.png"))

                    # 生成对比图：裁剪后的原图和彩色掩码对比
                    orig_crop = img_tensor[0, :, top:bottom, left:right]  # [3,H,W]
                    orig_crop_np = (orig_crop.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    orig_crop_np = orig_crop_np[:bottom - top, :right - left, :]

                    # 创建黑白掩码
                    pred_color = np.stack([pred_crop_bin[:bottom - top, :right - left]] * 3, axis=-1)

                    # 拼接原图裁剪部分（彩色）和预测结果（黑白掩膜）
                    compare = np.concatenate([orig_crop_np, pred_color], axis=1)
                    compare_pil = Image.fromarray(compare)
                    compare_pil.save(os.path.join(save_blocks_dir, f"compare_{block_idx}_{top}_{left}.png"))

                block_idx += 1

                final_mask[top:bottom, left:right] = np.maximum(
                    final_mask[top:bottom, left:right],
                    pred_crop_bin[:bottom - top, :right - left]
                )

    return final_mask


# === 主程序入口 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--weight', type=str, required=True, help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--crop-size', type=int, default=448)
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--save-dir', type=str, default='results/single_prediction')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('x1', 'y1', 'x2', 'y2'),
                        default=None, help='仅在指定 ROI 区域滑窗预测')
    parser.add_argument('--save-blocks-dir', type=str, default='roi_blocks', help='保存每个滑窗掩码和对比图的目录')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_blocks_dir, exist_ok=True)

    # === 加载模型 ===
    model = CrackNex(backbone=args.backbone)
    model.load_state_dict(torch.load(args.weight, map_location='cuda'), strict=False)
    model = model.cuda().eval()

    # === 加载图像 ===
    img_tensor, orig_pil = load_image(args.image)
    img_tensor = img_tensor.cuda()
    orig_np = np.array(orig_pil.convert("RGB"))

    # === 滑窗预测（是否启用 ROI） ===
    if args.roi:
        x1, y1, x2, y2 = args.roi
        print(f"ROI 限定区域: 左上({x1},{y1})，右下({x2},{y2})")
        pred_mask_np = sliding_window_predict_roi(
            model, img_tensor,
            roi_coords=(x1, y1, x2, y2),
            window_size=args.crop_size, overlap=args.overlap,
            save_blocks_dir=args.save_blocks_dir  # 保存每个滑窗的掩码和对比图
        )
    else:
        pred_mask_np = sliding_window_predict_roi(
            model, img_tensor,
            roi_coords=(0, 0, img_tensor.shape[3], img_tensor.shape[2]),
            window_size=args.crop_size, overlap=args.overlap,
            save_blocks_dir=args.save_blocks_dir  # 保存每个滑窗的掩码和对比图
        )

    # === 保存掩码图像 ===
    basename = os.path.splitext(os.path.basename(args.image))[0]
    pred_path = os.path.join(args.save_dir, f"pred_{basename}.png")
    Image.fromarray(pred_mask_np).save(pred_path)

    # === 绘制蓝色覆盖裂缝区域 ===
    overlay_img = orig_np.copy()
    overlay_img[pred_mask_np > 0] = [0, 0, 255]  # 用蓝色覆盖裂缝区域

    overlay_path = os.path.join(args.save_dir, f"overlay_{basename}.png")
    Image.fromarray(overlay_img).save(overlay_path)

    print(f"✅ 预测完成：\n- 掩码图：{pred_path}\n- 可视化图：{overlay_path}")


if __name__ == '__main__':
    main()




