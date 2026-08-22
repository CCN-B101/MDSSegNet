import os
import torch
import numpy as np
from PIL import Image
import argparse
import cv2

from model.ML_Mutil import MLiteUNet  # 使用 MLiteUNet 模型
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
    img = Image.open(path).convert('RGB')  # 统一转换为 RGB
    transform = Compose([
        normalize_only_img
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor, img  # 返回原始图像


# === 滑窗预测函数，限制 ROI 范围 ===
def sliding_window_predict_roi(model, img_tensor, roi_coords, window_size=448, overlap=0.2, cls=1,
                               save_blocks_dir=None, threshold=0.5):
    """
    滑窗预测函数
    Args:
        model: MLiteUNet 模型
        img_tensor: 输入图像张量 [1, 3, H, W]
        roi_coords: ROI区域坐标 (x1, y1, x2, y2)
        window_size: 滑窗大小
        overlap: 重叠率
        cls: 目标类别索引（裂缝类别）
        save_blocks_dir: 保存滑窗结果的目录
        threshold: 二值化阈值
    """
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

                # 模型推理
                output = model(crop)  # MLiteUNet 输出

                # 处理输出：如果是元组（训练模式），取第一个作为主输出
                if isinstance(output, tuple):
                    output = output[0]  # 主输出是 out_s2

                # 使用 sigmoid 将输出映射到 [0,1] 范围
                output_prob = torch.sigmoid(output)

                # 二值化：大于阈值视为裂缝
                pred_binary = (output_prob > threshold).float()

                # 提取指定类别的预测结果
                if pred_binary.shape[1] > 1:  # 多分类情况，取指定类别
                    pred_crop = pred_binary[:, cls:cls + 1, :, :]  # 提取指定类别
                else:  # 二分类情况
                    pred_crop = pred_binary

                # 裁剪到原始大小（去除padding）
                pred_crop = pred_crop[:, :, :bottom - top, :right - left]
                pred_crop_np = pred_crop.squeeze(0).squeeze(0).cpu().numpy()
                pred_crop_bin = (pred_crop_np * 255).astype(np.uint8)

                # 保存单独的mask
                if save_blocks_dir:
                    block_mask = Image.fromarray(pred_crop_bin)
                    block_mask.save(os.path.join(save_blocks_dir, f"block_{block_idx}_{top}_{left}.png"))

                    # 生成对比图：裁剪后的原图和彩色掩码对比
                    orig_crop = img_tensor[0, :, top:bottom, left:right]  # [3,H,W]
                    orig_crop_np = (orig_crop.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    orig_crop_np = orig_crop_np[:bottom - top, :right - left, :]

                    # 创建彩色掩码
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
    parser = argparse.ArgumentParser(description='MLiteUNet sliding window prediction')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--weight', type=str, required=True, help='模型权重路径')
    parser.add_argument('--crop-size', type=int, default=448, help='滑窗大小')
    parser.add_argument('--overlap', type=float, default=0.2, help='滑窗重叠率')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值 (0-1)')
    parser.add_argument('--save-dir', type=str, default='results/single_prediction', help='保存预测结果的目录')
    parser.add_argument('--n-classes', type=int, default=2, help='分类数量')
    parser.add_argument('--class-idx', type=int, default=1, help='目标类别索引（裂缝类别）')
    parser.add_argument('--pretrained-backbone', action='store_true', default=False, help='是否使用预训练骨干网络')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('x1', 'y1', 'x2', 'y2'),
                        default=None, help='仅在指定 ROI 区域滑窗预测')
    parser.add_argument('--save-blocks-dir', type=str, default='roi_blocks', help='保存每个滑窗掩码和对比图的目录')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_blocks_dir:
        os.makedirs(args.save_blocks_dir, exist_ok=True)

    # === 加载模型 ===
    print(f"Loading MLiteUNet model with n_classes={args.n_classes}...")
    model = MLiteUNet(
        n_classes=args.n_classes,
        aux_mode='eval',  # 评估模式，只返回主输出
        pretrained_backbone=args.pretrained_backbone
    )

    # 加载权重
    if torch.cuda.is_available():
        state_dict = torch.load(args.weight, map_location='cuda')
    else:
        state_dict = torch.load(args.weight, map_location='cpu')

    # 处理权重加载（兼容不同的保存格式）
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # 加载权重，忽略不匹配的键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")

    model = model.cuda().eval()
    print("Model loaded successfully!")

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
            window_size=args.crop_size,
            overlap=args.overlap,
            cls=args.class_idx,
            threshold=args.threshold,
            save_blocks_dir=args.save_blocks_dir
        )
    else:
        print("预测全图...")
        pred_mask_np = sliding_window_predict_roi(
            model, img_tensor,
            roi_coords=(0, 0, img_tensor.shape[3], img_tensor.shape[2]),
            window_size=args.crop_size,
            overlap=args.overlap,
            cls=args.class_idx,
            threshold=args.threshold,
            save_blocks_dir=args.save_blocks_dir
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
