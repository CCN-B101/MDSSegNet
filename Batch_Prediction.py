import os
import torch
import numpy as np
from PIL import Image
import argparse
import cv2
from model.cracknex import CrackNex  # 使用你的模型结构定义文件路径
# from model.UNet import CrackNex
# from model.MobileNetV3ASPPEAA import CrackNex
# from model.Deeplabv3 import CrackNex
# from model.DenseNet121 import CrackNex

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


# === 图像预测 ===
# 预测并保存结果
def predict_and_save(model, image_path, output_path):
    # 加载图像并进行预处理
    img_tensor, orig_pil = load_image(image_path)
    img_tensor = img_tensor.cuda()  # 移动到 GPU

    # 进行模型推理
    model.eval()  # 进入评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(img_tensor)  # 获取模型输出

    # 对输出进行二值化（背景区域为白色，裂缝区域为黑色）
    # 修改：小于或等于 0.5 的值视为裂缝区域（1），大于 0.5 的值视为背景区域（0）
    output_binary = (output <= 0.5).float().squeeze(0).cpu().numpy()  # 裂缝区域为 1，背景为 0
    output_binary = (output_binary * 255).astype(np.uint8)  # 转换为 0-255 的图像

    # 确保 output_binary 是一个二维数组 (H, W)
    if output_binary.ndim == 3:
        output_binary = output_binary[0]  # 如果是 (1, H, W) 形式，去掉第一个维度

    # 保存预测掩码图像，确保是 PNG 格式
    output_image = Image.fromarray(output_binary)  # 使用二值化的图像
    output_image.save(output_path, format='PNG')  # 显式指定格式为 PNG
    print(f"Predicted binary image saved to {output_path}")



# === 批量预测文件夹中的所有图像 ===
def predict_folder(model, input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # 遍历文件夹中的每个图像文件
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)  # 输入图像的完整路径
        output_path = os.path.join(output_folder, f"pred_{os.path.splitext(image_file)[0]}.png")  # 输出图像的完整路径

        # 预测并保存结果
        predict_and_save(model, image_path, output_path)


# === 主程序入口 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--weight', type=str, required=True, help='模型权重路径')
    parser.add_argument('--save-dir', type=str, default='results/prediction', help='保存预测结果的目录')
    parser.add_argument('--backbone', type=str, choices=['resnet101', 'densenet121'], default='resnet101', help='模型的骨干网络')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # === 加载模型 ===
    model = CrackNex(backbone=args.backbone)
    model.load_state_dict(torch.load(args.weight, map_location='cuda'), strict=False)
    model = model.cuda().eval()

    # === 批量预测文件夹中的所有图像 ===
    predict_folder(model, args.image_folder, args.save_dir)


if __name__ == '__main__':
    main()


