# 单张图像448*448裂缝宽度、长度检测（带编号叠加+保存文本结果）
# import os
# import torch
# import cv2
# import numpy as np
# from torchvision import transforms
# from PIL import Image
# from model.cracknex import CrackNex  # 假设这是你的模型
# from skimage.morphology import skeletonize  # 用于细化裂缝区域
# from skimage.measure import label  # 用于连通组件标记
#
#
# # 1. 加载训练好的模型权重
# def load_trained_model(model_path, backbone='resnet101'):
#     model = CrackNex(backbone=backbone)
#     model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
#     model = model.cuda()  # 如果有GPU，移动到GPU
#     model.eval()  # 设置为评估模式
#     return model
#
#
# # 2. 图像预处理（返回“已缩放为448×448”的PIL图，便于可视化坐标一致）
# def preprocess_image(image_path, size=(448, 448)):
#     img = Image.open(image_path).convert('RGB')
#     img_resized = img.resize(size, Image.BILINEAR)
#     transform = transforms.Compose([
#         transforms.Resize(size),  # 与模型输入保持一致
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img_tensor = transform(img).unsqueeze(0).cuda()  # [1,3,H,W]
#     return img_tensor, img_resized  # 返回缩放后的图像用于叠加编号
#
#
# # 3. 获取裂缝掩码（模型推理）
# def get_crack_mask(model, img_tensor, save_path="crack_mask.png"):
#     with torch.no_grad():
#         model_output = model(img_tensor)  # 形状 [1, 2, 448, 448]（背景和裂缝）
#     crack_mask = torch.argmax(model_output, dim=1)  # [1, H, W]，值为0或1
#
#     # 保存裂缝掩码图像
#     crack_mask_img = (crack_mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255)
#     crack_mask_bgr = cv2.cvtColor(crack_mask_img, cv2.COLOR_GRAY2BGR)
#     cv2.imwrite(save_path, crack_mask_bgr)
#
#     return crack_mask  # Tensor, 0/1
#
#
# # 4. 计算裂缝的长度和宽度（对单个连通域掩码）
# def calculate_crack_length_and_width(pred_mask):
#     """
#     pred_mask: 单个裂缝的二值掩码(0/1), ndarray(H,W)
#     返回: (骨架像素数, 距离变换最大值)
#     """
#     skeleton = skeletonize(pred_mask.astype(np.uint8)).astype(np.uint8)
#     crack_length = int(np.sum(skeleton))  # 骨架像素数
#
#     # 距离变换：L2度量，取最大值（注意其物理含义更接近半径；若要直径，可再×2）
#     distance_transform = cv2.distanceTransform(pred_mask.astype(np.uint8), cv2.DIST_L2, 5)
#     crack_width = float(np.max(distance_transform))
#     return crack_length, crack_width
#
#
# # 5. 处理预测结果：返回每条裂缝的长度、宽度、编号图
# def process_predictions(crack_mask_tensor):
#     # 转为CPU numpy，形状(H, W)，二值(0/1)
#     crack_mask = crack_mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
#     crack_mask = (crack_mask > 0).astype(np.uint8)
#
#     # 连通组件编号图：背景0，裂缝1..N
#     labeled_mask = label(crack_mask, connectivity=2)
#     num_labels = int(labeled_mask.max())
#
#     crack_length_list, crack_width_list = [], []
#     for label_id in range(1, num_labels + 1):
#         pred_mask = (labeled_mask == label_id).astype(np.uint8)
#         L, W = calculate_crack_length_and_width(pred_mask)
#         crack_length_list.append(L)
#         crack_width_list.append(W)
#
#     return crack_length_list, crack_width_list, labeled_mask  # 方便后续标注
#
#
# # 6. 将编号叠加到图像上并保存
# def save_numbered_overlay(labeled_mask, base_image_pil, lengths, widths, save_path="crack_overlay_numbered.png"):
#     """
#     labeled_mask: ndarray(H,W) 的连通域编号图（0为背景, 1..N为裂缝ID）
#     base_image_pil: 已缩放到与mask同尺寸的PIL图（448×448）
#     lengths, widths: 与ID顺序一致的列表
#     """
#     overlay_bgr = cv2.cvtColor(np.array(base_image_pil), cv2.COLOR_RGB2BGR)
#
#     # 把裂缝像素涂成蓝色，增强可见性（可选）
#     mask_bin = (labeled_mask > 0).astype(np.uint8)
#     overlay_bgr[mask_bin == 1] = (255, 0, 0)  # BGR的蓝色
#
#     count = int(labeled_mask.max())
#     for label_id in range(1, count + 1):
#         ys, xs = np.where(labeled_mask == label_id)
#         if ys.size == 0:
#             continue
#         cy, cx = int(ys.mean()), int(xs.mean())
#         # 画一个小圆心与编号
#         cv2.circle(overlay_bgr, (cx, cy), 4, (255, 255, 255), -1)
#         cv2.putText(overlay_bgr, str(label_id), (cx + 6, cy - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
#
#     cv2.imwrite(save_path, overlay_bgr)
#     return save_path
#
#
# # 主程序：加载模型，推理，计算并编号可视化
# def main(image_path, model_path):
#     # 1. 加载模型
#     model = load_trained_model(model_path)
#
#     # 2. 预处理（返回与mask同尺寸的img_resized用于可视化）
#     img_tensor, img_resized = preprocess_image(image_path)
#
#     # 3. 推理并保存掩码（crack_mask.png）
#     mask_path = "crack_mask.png"
#     crack_mask = get_crack_mask(model, img_tensor, save_path=mask_path)
#
#     # 4. 计算每条裂缝的长度与宽度，并取连通域编号图
#     crack_lengths, crack_widths, labeled_mask = process_predictions(crack_mask)
#
#     # 5. 可视化编号叠加并保存（crack_overlay_numbered.png）
#     overlay_path = save_numbered_overlay(labeled_mask, img_resized, crack_lengths, crack_widths,
#                                          save_path="crack_overlay_numbered.png")
#
#     # 6. 输出 & 保存文本结果（crack_results.txt）
#     total_count = int(labeled_mask.max())
#     print(f"总裂缝数: {total_count}")
#     for i, (L, W) in enumerate(zip(crack_lengths, crack_widths), start=1):
#         print(f"#{i}: 长度={L} 像素, 宽度(maxDist)={W:.4f} 像素")
#
#     report_path = "crack_results.txt"
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(f"总裂缝数: {total_count}\n")
#         for i, (L, W) in enumerate(zip(crack_lengths, crack_widths), start=1):
#             f.write(f"#{i}: 长度={L} 像素, 宽度(maxDist)={W:.4f} 像素\n")
#
#     print(f"掩码已保存: {os.path.abspath(mask_path)}")
#     print(f"编号覆盖图已保存: {os.path.abspath(overlay_path)}")
#     print(f"结果文本已保存: {os.path.abspath(report_path)}")
#
#
# # 示例：运行主程序
# if __name__ == "__main__":
#     image_path = '/home/test/002.jpg'  # 输入图像路径
#     model_path = '/home/test/CCN/SCDN14/checkpoints/best_mIoU_resnet101_0.8185.pth'  # 训练好的模型权重路径
#     main(image_path, model_path)




#  二
# # 整幅图像裂缝宽度、长度数量检测并保存 .裂缝长度计算用skeletonize() 做骨架，再用骨架像素数作为长度
# import os
# import torch
# import numpy as np
# from PIL import Image
# import argparse
# import cv2
# from model.cracknex import CrackNex  # 使用你的模型结构定义文件路径
# import torch.nn.functional as F
# from torchvision.transforms import ToTensor, Compose
# import torchvision.transforms.functional as TF
# from skimage.morphology import skeletonize  # 用于细化裂缝区域
# from skimage.measure import label  # 用于连通组件标记
#
#
# # === 自定义图像归一化（仅图像，不需要 mask） ===
# def normalize_only_img(img):
#     # 确保图像是三通道
#     if img.mode != 'RGB':
#         img = img.convert('RGB')  # 如果图像是灰度图或其他格式，转换为 RGB
#
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     return TF.normalize(ToTensor()(img), mean=mean, std=std)
#
#
# # === 加载图像并归一化 ===
# def load_image(path):
#     img = Image.open(path)  # 不进行 convert('RGB')，保留原始图像格式
#     transform = Compose([
#         normalize_only_img
#     ])
#     tensor = transform(img).unsqueeze(0)
#     return tensor, img  # 返回原始图像（不转换为 RGB）
#
#
# # === 计算裂缝的长度和宽度 ===
# def calculate_crack_length_and_width(pred_mask):
#     """
#     计算裂缝的长度和宽度
#     :param pred_mask: 二值化的裂缝掩码图（0:背景，1:裂缝）
#     :return: 裂缝的长度和宽度
#     """
#     # 骨架提取：细化裂缝区域
#     skeleton = skeletonize(pred_mask.astype(np.uint8))  # 使用细化算法
#     skeleton = skeleton.astype(np.uint8)
#
#     # 计算裂缝的长度（骨架的像素数量）
#     crack_length = np.sum(skeleton)  # 计算骨架中1的个数（即裂缝的长度）
#
#     # 计算裂缝的宽度：使用距离变换来计算
#     distance_transform = cv2.distanceTransform(pred_mask.astype(np.uint8), cv2.DIST_L2, 5)
#     crack_width = np.max(distance_transform)  # 获取最大宽度
#
#     return crack_length, crack_width
#
#
# # === 连通组件标记，返回裂缝的数量和每个裂缝的掩码 ===
# def process_predictions(crack_mask):
#     crack_length_list = []
#     crack_width_list = []
#     crack_count = 0
#
#     # 打印 crack_mask 的形状，检查它的维度
#     print(f"crack_mask shape: {crack_mask.shape}")
#
#     # 检查 crack_mask 的维度，并做出相应的处理
#     if len(crack_mask.shape) == 4:  # 如果是批次（batch size，通常是[batch_size, channels, H, W]）
#         # 只处理批次中的第一张图像
#         crack_mask_cpu = crack_mask[0].cpu().numpy()  # 取出批次中的第一张图像
#     elif len(crack_mask.shape) == 3:  # 如果只有 (channels, H, W)
#         # 如果是单通道的图像
#         crack_mask_cpu = crack_mask.cpu().numpy()  # 转换为 numpy 数组
#     elif len(crack_mask.shape) == 2:  # 如果是单张图像 (H, W)
#         # 如果只有 (H, W)，直接处理
#         crack_mask_cpu = crack_mask
#     else:
#         raise ValueError("Unexpected crack_mask shape, please check the input dimensions")
#
#     # 连通组件标记
#     labeled_mask = label(crack_mask_cpu)  # 标记每个裂缝
#     num_labels = np.max(labeled_mask)  # 获取标记的数量（即裂缝的数量）
#     crack_count = num_labels  # 记录裂缝数量
#
#     for label_id in range(1, num_labels + 1):
#         # 获取当前裂缝的二值掩码
#         pred_mask = (labeled_mask == label_id).astype(np.uint8)
#
#         # 计算该裂缝的长度和宽度
#         length, width = calculate_crack_length_and_width(pred_mask)
#         crack_length_list.append(length)
#         crack_width_list.append(width)
#
#     return crack_length_list, crack_width_list, crack_count
#
#
#
# #  === 滑窗预测函数，限制 ROI 范围 ===
# # 在原代码的基础上进行修改
# def sliding_window_predict_roi(model, img_tensor, roi_coords, window_size=448, overlap=0.2, cls=1,
#                                save_blocks_dir=None):
#     _, _, H, W = img_tensor.shape
#     x1, y1, x2, y2 = roi_coords
#     roi_w, roi_h = x2 - x1, y2 - y1
#     stride = int(window_size * (1 - overlap))
#
#     final_mask = np.zeros((H, W), dtype=np.uint8)
#
#     model.eval()
#     with torch.no_grad():
#         block_idx = 0  # 计数滑窗块
#         # 从ROI的左上角开始，按照固定步长滑动窗口
#         for top in range(y1, y2, stride):
#             for left in range(x1, x2, stride):
#                 # 确保窗口大小为window_size，即使在边界处也要保持大小一致
#                 bottom = min(top + window_size, y2)
#                 right = min(left + window_size, x2)
#
#                 crop = img_tensor[:, :, top:bottom, left:right]
#                 pad_bottom = window_size - crop.shape[2]
#                 pad_right = window_size - crop.shape[3]
#                 if pad_bottom > 0 or pad_right > 0:
#                     crop = F.pad(crop, (0, pad_right, 0, pad_bottom), mode="constant", value=0)
#
#                 # 仅使用查询图像进行推理
#                 pred = model(crop)  # 模型的前向推理
#                 pred = torch.argmax(pred, dim=1)
#                 pred_crop = pred[:, :bottom - top, :right - left].squeeze(0).cpu().numpy()
#                 pred_crop_bin = (pred_crop == cls).astype(np.uint8) * 255
#
#                 # 保存单独的mask
#                 if save_blocks_dir:
#                     block_mask = Image.fromarray(pred_crop_bin)
#                     block_mask.save(os.path.join(save_blocks_dir, f"block_{block_idx}_{top}_{left}.png"))
#
#                     # 生成对比图：裁剪后的原图和彩色掩码对比
#                     orig_crop = img_tensor[0, :, top:bottom, left:right]  # [3,H,W]
#                     orig_crop_np = (orig_crop.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#                     orig_crop_np = orig_crop_np[:bottom - top, :right - left, :]
#
#                     # 创建黑白掩码
#                     pred_color = np.stack([pred_crop_bin[:bottom - top, :right - left]] * 3, axis=-1)
#
#                     # 拼接原图裁剪部分（彩色）和预测结果（黑白掩膜）对比
#                     compare = np.concatenate([orig_crop_np, pred_color], axis=1)
#                     compare_pil = Image.fromarray(compare)
#                     compare_pil.save(os.path.join(save_blocks_dir, f"compare_{block_idx}_{top}_{left}.png"))
#
#                 block_idx += 1
#
#                 final_mask[top:bottom, left:right] = np.maximum(
#                     final_mask[top:bottom, left:right],
#                     pred_crop_bin[:bottom - top, :right - left]
#                 )
#
#     return final_mask
#
#
# # === 主程序入口 ===
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, required=True, help='输入图像路径')
#     parser.add_argument('--weight', type=str, required=True, help='模型权重路径')
#     parser.add_argument('--backbone', type=str, default='resnet101')
#     parser.add_argument('--crop-size', type=int, default=448)
#     parser.add_argument('--overlap', type=float, default=0.2)
#     parser.add_argument('--save-dir', type=str, default='results/single_prediction')
#     parser.add_argument('--roi', type=int, nargs=4, metavar=('x1', 'y1', 'x2', 'y2'),
#                         default=None, help='仅在指定 ROI 区域滑窗预测')
#     parser.add_argument('--save-blocks-dir', type=str, default='roi_blocks', help='保存每个滑窗掩码和对比图的目录')
#     parser.add_argument('--output-file', type=str, default='crack_results.txt', help='保存裂缝数量、长度和宽度的文本文件')
#     args = parser.parse_args()
#
#     os.makedirs(args.save_dir, exist_ok=True)
#     os.makedirs(args.save_blocks_dir, exist_ok=True)
#
#     # === 加载模型 ===
#     model = CrackNex(backbone=args.backbone)
#     model.load_state_dict(torch.load(args.weight, map_location='cuda'), strict=False)
#     model = model.cuda().eval()
#
#     # === 加载图像 ===
#     img_tensor, orig_pil = load_image(args.image)
#     img_tensor = img_tensor.cuda()
#     orig_np = np.array(orig_pil.convert("RGB"))
#
#     # === 滑窗预测（是否启用 ROI） ===
#     if args.roi:
#         x1, y1, x2, y2 = args.roi
#         print(f"ROI 限定区域: 左上({x1},{y1})，右下({x2},{y2})")
#         pred_mask_np = sliding_window_predict_roi(
#             model, img_tensor,
#             roi_coords=(x1, y1, x2, y2),
#             window_size=args.crop_size, overlap=args.overlap,
#             save_blocks_dir=args.save_blocks_dir  # 保存每个滑窗的掩码和对比图
#         )
#     else:
#         pred_mask_np = sliding_window_predict_roi(
#             model, img_tensor,
#             roi_coords=(0, 0, img_tensor.shape[3], img_tensor.shape[2]),
#             window_size=args.crop_size, overlap=args.overlap,
#             save_blocks_dir=args.save_blocks_dir  # 保存每个滑窗的掩码和对比图
#         )
#
#     # === 保存掩码图像 ===
#     basename = os.path.splitext(os.path.basename(args.image))[0]
#     pred_path = os.path.join(args.save_dir, f"pred_{basename}.png")
#     Image.fromarray(pred_mask_np).save(pred_path)
#
#     # === 绘制蓝色覆盖裂缝区域 ===
#     overlay_img = orig_np.copy()
#     overlay_img[pred_mask_np > 0] = [0, 0, 255]  # 用蓝色覆盖裂缝区域
#
#     overlay_path = os.path.join(args.save_dir, f"overlay_{basename}.png")
#     Image.fromarray(overlay_img).save(overlay_path)
#
#     # === 计算裂缝长度、宽度和数量 ===
#     crack_length, crack_width, crack_count = process_predictions(pred_mask_np)
#
#     # === 输出计算结果 ===
#     print(f"裂缝数量: {crack_count}")
#     print(f"裂缝长度: {crack_length}")
#     print(f"裂缝宽度: {crack_width}")
#
#     # === 保存裂缝数据到文本文件 ===
#     output_file = os.path.join(args.save_dir, args.output_file)
#     with open(output_file, 'w') as f:
#         f.write(f"裂缝数量: {crack_count}\n")
#         f.write(f"裂缝长度: {crack_length}\n")
#         f.write(f"裂缝宽度: {crack_width}\n")
#
#     # === 保存带有标注的图像 ===
#     final_image_path = os.path.join(args.save_dir, f"labeled_{basename}.png")
#     Image.fromarray(overlay_img).save(final_image_path)
#
#     print(f"✅ 预测完成：\n- 掩码图：{pred_path}\n- 可视化图：{overlay_path}\n- 裂缝数据：{output_file}")
#
#
# if __name__ == '__main__':
#     main()




#  三
# 整幅图像裂缝宽度、长度数量检测并保存（带编号叠加）用 Zhang–Suen 细化（thin()）做骨架，再用骨架像素数作为长度。
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
from skimage.morphology import skeletonize  # 用于细化裂缝区域
from skimage.measure import label  # 用于连通组件标记


# === 自定义图像归一化（仅图像，不需要 mask） ===
def normalize_only_img(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return TF.normalize(ToTensor()(img), mean=mean, std=std)


# === 加载图像并归一化 ===
def load_image(path):
    img = Image.open(path)  # 保留原始格式
    transform = Compose([normalize_only_img])
    tensor = transform(img).unsqueeze(0)
    return tensor, img


# === 计算裂缝的长度和宽度 ===
def calculate_crack_length_and_width(pred_mask):
    """
    pred_mask: 二值掩码(0/1)，单个连通域
    返回: (骨架像素数, 距离变换最大值)
    """
    skeleton = skeletonize(pred_mask.astype(np.uint8)).astype(np.uint8)
    crack_length = np.sum(skeleton)  # 骨架像素数
    # 距离变换用0/1更稳
    distance_transform = cv2.distanceTransform(pred_mask.astype(np.uint8), cv2.DIST_L2, 5)
    crack_width = float(np.max(distance_transform))
    return crack_length, crack_width


# === 连通组件标记，返回裂缝的数量和每条裂缝的度量 + 编号图 ===
def process_predictions(crack_mask):
    crack_length_list = []
    crack_width_list = []

    print(f"crack_mask shape: {crack_mask.shape}")

    if len(crack_mask.shape) == 4:
        crack_mask_cpu = crack_mask[0].cpu().numpy()
    elif len(crack_mask.shape) == 3:
        crack_mask_cpu = crack_mask.cpu().numpy()
    elif len(crack_mask.shape) == 2:
        crack_mask_cpu = crack_mask
    else:
        raise ValueError("Unexpected crack_mask shape")

    # 若输入是0/255，将其规整为0/1（label对0/255也能工作，但后续距离变换更稳）
    if crack_mask_cpu.max() > 1:
        crack_mask_cpu = (crack_mask_cpu > 0).astype(np.uint8)

    # 连通域编号图: 0为背景，1..N为各裂缝
    labeled_mask = label(crack_mask_cpu, connectivity=2)
    num_labels = int(labeled_mask.max())

    for label_id in range(1, num_labels + 1):
        pred_mask = (labeled_mask == label_id).astype(np.uint8)
        length, width = calculate_crack_length_and_width(pred_mask)
        crack_length_list.append(int(length))
        crack_width_list.append(float(width))

    crack_count = num_labels
    # 现在额外返回 labeled_mask 用于可视化编号
    return crack_length_list, crack_width_list, crack_count, labeled_mask


#  === 滑窗预测函数，限制 ROI 范围 ===
def sliding_window_predict_roi(model, img_tensor, roi_coords, window_size=448, overlap=0.2, cls=1,
                               save_blocks_dir=None):
    _, _, H, W = img_tensor.shape
    x1, y1, x2, y2 = roi_coords
    stride = int(window_size * (1 - overlap))

    final_mask = np.zeros((H, W), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        block_idx = 0
        for top in range(y1, y2, stride):
            for left in range(x1, x2, stride):
                bottom = min(top + window_size, y2)
                right = min(left + window_size, x2)

                crop = img_tensor[:, :, top:bottom, left:right]
                pad_bottom = window_size - crop.shape[2]
                pad_right = window_size - crop.shape[3]
                if pad_bottom > 0 or pad_right > 0:
                    crop = F.pad(crop, (0, pad_right, 0, pad_bottom), mode="constant", value=0)

                pred = model(crop)
                pred = torch.argmax(pred, dim=1)
                pred_crop = pred[:, :bottom - top, :right - left].squeeze(0).cpu().numpy()
                pred_crop_bin = (pred_crop == cls).astype(np.uint8) * 255

                if save_blocks_dir:
                    os.makedirs(save_blocks_dir, exist_ok=True)
                    block_mask = Image.fromarray(pred_crop_bin)
                    block_mask.save(os.path.join(save_blocks_dir, f"block_{block_idx}_{top}_{left}.png"))

                    orig_crop = img_tensor[0, :, top:bottom, left:right]
                    orig_crop_np = (orig_crop.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    orig_crop_np = orig_crop_np[:bottom - top, :right - left, :]

                    pred_color = np.stack([pred_crop_bin[:bottom - top, :right - left]] * 3, axis=-1)
                    compare = np.concatenate([orig_crop_np, pred_color], axis=1)
                    compare_pil = Image.fromarray(compare)
                    compare_pil.save(os.path.join(save_blocks_dir, f"compare_{block_idx}_{top}_{left}.png"))

                block_idx += 1

                final_mask[top:bottom, left:right] = np.maximum(
                    final_mask[top:bottom, left:right],
                    pred_crop_bin[:bottom - top, :right - left]
                )

    return final_mask


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
    parser.add_argument('--output-file', type=str, default='crack_results.txt',
                        help='保存裂缝数量、长度和宽度的文本文件')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_blocks_dir:
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
            save_blocks_dir=args.save_blocks_dir
        )
    else:
        pred_mask_np = sliding_window_predict_roi(
            model, img_tensor,
            roi_coords=(0, 0, img_tensor.shape[3], img_tensor.shape[2]),
            window_size=args.crop_size, overlap=args.overlap,
            save_blocks_dir=args.save_blocks_dir
        )

    # === 保存掩码图像 ===
    basename = os.path.splitext(os.path.basename(args.image))[0]
    pred_path = os.path.join(args.save_dir, f"pred_{basename}.png")
    Image.fromarray(pred_mask_np).save(pred_path)

    # === 绘制蓝色覆盖裂缝区域 ===
    overlay_img = orig_np.copy()
    overlay_img[pred_mask_np > 0] = [0, 0, 255]  # 蓝色覆盖裂缝
    overlay_path = os.path.join(args.save_dir, f"overlay_{basename}.png")
    Image.fromarray(overlay_img).save(overlay_path)

    # === 计算裂缝长度、宽度、数量 + 获得编号图 ===
    crack_length, crack_width, crack_count, labeled_mask = process_predictions(pred_mask_np)

    # === 在覆盖图上叠加编号（1..N） ===
    # 计算每个连通域的质心并绘制编号
    overlay_numbered = overlay_img.copy()
    h, w = labeled_mask.shape
    for label_id in range(1, crack_count + 1):
        ys, xs = np.where(labeled_mask == label_id)
        if ys.size == 0:
            continue
        cy = int(ys.mean())
        cx = int(xs.mean())
        # 画一点小圆心增强可见性
        cv2.circle(overlay_numbered, (cx, cy), 4, (255, 255, 255), -1)
        # 写编号
        cv2.putText(overlay_numbered, str(label_id), (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    overlay_numbered_path = os.path.join(args.save_dir, f"overlay_numbered_{basename}.png")
    Image.fromarray(overlay_numbered).save(overlay_numbered_path)

    # === 输出计算结果 ===
    print(f"裂缝数量: {crack_count}")
    print(f"裂缝长度: {crack_length}")
    print(f"裂缝宽度: {crack_width}")
    print(f"编号可视化已保存: {overlay_numbered_path}")

    # === 保存裂缝数据到文本文件（逐条对应编号-长度-宽度） ===
    output_file = os.path.join(args.save_dir, args.output_file)
    with open(output_file, 'w') as f:
        f.write(f"裂缝数量: {crack_count}\n")
        f.write("编号, 长度(像素), 宽度(像素, 距离变换最大值)\n")
        for i, (L, W) in enumerate(zip(crack_length, crack_width), start=1):
            f.write(f"{i}, {L}, {W:.4f}\n")

    # 同时保留原有提示
    print(f"✅ 预测完成：\n- 掩码图：{pred_path}\n- 覆盖图：{overlay_path}\n- 编号覆盖图：{overlay_numbered_path}\n- 裂缝数据：{output_file}")


if __name__ == '__main__':
    main()


# 第一段代码：专注于单张图像的裂缝检测，计算和保存裂缝的长度与宽度。
# 第二段代码：支持批量处理和滑窗预测，适用于大图像，计算裂缝的数量和特征。
# 第三段代码：与第二段相似，但使用不同的细化算法，强调裂缝的骨架化处理。

