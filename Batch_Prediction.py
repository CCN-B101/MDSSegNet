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


# === 图像预测 ===
def predict_and_save(model, image_path, output_path, threshold=0.5):
    """
    预测并保存结果
    Args:
        model: 训练好的模型
        image_path: 输入图像路径
        output_path: 输出图像路径
        threshold: 二值化阈值，默认0.5
    """
    # 加载图像并进行预处理
    img_tensor, orig_pil = load_image(image_path)
    img_tensor = img_tensor.cuda()  # 移动到 GPU

    # 进行模型推理
    model.eval()  # 进入评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(img_tensor)  # 获取模型输出

        # 如果输出是元组（训练模式返回多个输出），取第一个作为主输出
        if isinstance(output, tuple):
            output = output[0]  # 主输出是 out_s2

    # 对输出进行二值化
    # 使用 sigmoid 将输出映射到 [0,1] 范围
    output_prob = torch.sigmoid(output)

    # 二值化：大于阈值视为裂缝（1），小于阈值视为背景（0）
    output_binary = (output_prob > threshold).float().squeeze(0).cpu().numpy()

    # 如果是多通道输出（n_classes > 1），取第一个通道
    if output_binary.ndim == 3:
        output_binary = output_binary[0]  # 如果是 (1, H, W) 形式，去掉第一个维度
    elif output_binary.ndim == 4:
        output_binary = output_binary[0, 0]  # 如果是 (1, n_classes, H, W) 形式

    # 转换为 0-255 的图像 (裂缝为白色，背景为黑色)
    output_binary = (output_binary * 255).astype(np.uint8)

    # 保存预测掩码图像
    output_image = Image.fromarray(output_binary, mode='L')  # 使用灰度模式
    output_image.save(output_path, format='PNG')
    print(f"Predicted binary image saved to {output_path}")


# === 批量预测文件夹中的所有图像 ===
def predict_folder(model, input_folder, output_folder, threshold=0.5):
    """
    批量预测文件夹中的所有图像
    Args:
        model: 训练好的模型
        input_folder: 输入图像文件夹路径
        output_folder: 输出图像文件夹路径
        threshold: 二值化阈值
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中的所有图像文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"Warning: No image files found in {input_folder}")
        return

    print(f"Found {len(image_files)} images to process")

    # 遍历文件夹中的每个图像文件
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"pred_{os.path.splitext(image_file)[0]}.png")

        print(f"Processing [{idx + 1}/{len(image_files)}]: {image_file}")

        # 预测并保存结果
        predict_and_save(model, image_path, output_path, threshold)


# === 主程序入口 ===
def main():
    parser = argparse.ArgumentParser(description='Batch prediction using MLiteUNet model')
    parser.add_argument('--image-folder', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--weight', type=str, required=True, help='模型权重路径')
    parser.add_argument('--save-dir', type=str, default='results/prediction', help='保存预测结果的目录')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值 (0-1)')
    parser.add_argument('--n-classes', type=int, default=2, help='分类数量')
    parser.add_argument('--pretrained-backbone', action='store_true', default=False, help='是否使用预训练骨干网络')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # === 加载模型 ===
    print(f"Loading model with n_classes={args.n_classes}...")
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
        print(f"Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")

    # 移动到 GPU 并设置为评估模式
    model = model.cuda().eval()
    print("Model loaded successfully!")

    # === 批量预测文件夹中的所有图像 ===
    predict_folder(model, args.image_folder, args.save_dir, args.threshold)


if __name__ == '__main__':
    main()
