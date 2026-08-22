# 单张图像448*448裂缝宽度、长度检测（带编号叠加+保存文本结果）
# import os
# import torch
# import cv2
# import numpy as np
# from torchvision import transforms
# from PIL import Image
# from model.ML_Mutil import MLiteUNet  # 导入您的模型
# from skimage.morphology import skeletonize  # 用于细化裂缝区域
# from skimage.measure import label  # 用于连通组件标记
#
#
# # 1. 加载训练好的模型权重（修改为CPU兼容）
# def load_trained_model(model_path, n_classes=2, backbone='resnet101'):
#     # 创建模型实例（与训练时参数一致）
#     model = MLiteUNet(n_classes=n_classes, aux_mode='eval', pretrained_backbone=False)
#
#     # 检测是否有可用的GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#
#     # 根据设备加载模型权重
#     try:
#         if device.type == 'cuda':
#             state_dict = torch.load(model_path)
#             model.load_state_dict(state_dict)
#             model = model.cuda()
#         else:
#             # CPU模式：使用map_location参数
#             state_dict = torch.load(model_path, map_location=torch.device('cpu'))
#             model.load_state_dict(state_dict)
#             model = model.cpu()
#     except RuntimeError as e:
#         print(f"加载模型时出错: {e}")
#         print("尝试使用严格=False加载...")
#         if device.type == 'cuda':
#             state_dict = torch.load(model_path)
#             model.load_state_dict(state_dict, strict=False)
#             model = model.cuda()
#         else:
#             state_dict = torch.load(model_path, map_location=torch.device('cpu'))
#             model.load_state_dict(state_dict, strict=False)
#             model = model.cpu()
#
#     model.eval()  # 设置为评估模式
#     return model, device
#
#
# # 2. 图像预处理（返回"已缩放为448×448"的PIL图，便于可视化坐标一致）
# def preprocess_image(image_path, device, size=(448, 448)):
#     img = Image.open(image_path).convert('RGB')
#     img_resized = img.resize(size, Image.BILINEAR)
#     transform = transforms.Compose([
#         transforms.Resize(size),  # 与模型输入保持一致
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     img_tensor = transform(img).unsqueeze(0)  # [1,3,H,W]
#
#     # 根据设备移动张量
#     if device.type == 'cuda':
#         img_tensor = img_tensor.cuda()
#     else:
#         img_tensor = img_tensor.cpu()
#
#     return img_tensor, img_resized  # 返回缩放后的图像用于叠加编号
#
#
# # 3. 获取裂缝掩码（模型推理）
# def get_crack_mask(model, img_tensor, device, save_path="crack_mask.png"):
#     with torch.no_grad():
#         model_output = model(img_tensor)  # 形状 [1, 2, 448, 448]（背景和裂缝）
#
#     # 处理不同的输出格式
#     if isinstance(model_output, tuple):
#         # 如果是训练模式，取第一个输出（主输出）
#         model_output = model_output[0]
#
#     crack_mask = torch.argmax(model_output, dim=1)  # [1, H, W]，值为0或1
#
#     # 保存裂缝掩码图像（移动到CPU处理）
#     crack_mask_img = (crack_mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255)
#     crack_mask_bgr = cv2.cvtColor(crack_mask_img, cv2.COLOR_GRAY2BGR)
#     cv2.imwrite(save_path, crack_mask_bgr)
#
#     return crack_mask  # Tensor, 0/1
#
#
# # 4. 计算裂缝的长度和宽度，并返回最大宽度位置
# def calculate_crack_length_and_width_with_position(pred_mask):
#     """
#     pred_mask: 单个裂缝的二值掩码(0/1), ndarray(H,W)
#     返回: (骨架像素数, 距离变换最大值, 最大宽度坐标)
#     """
#     # 检查掩码是否为空
#     if np.sum(pred_mask) == 0:
#         return 0, 0.0, (0, 0)
#
#     # 计算骨架（用于长度）
#     skeleton = skeletonize(pred_mask.astype(np.uint8)).astype(np.uint8)
#     crack_length = int(np.sum(skeleton))  # 骨架像素数
#
#     # 距离变换：L2度量，取最大值
#     distance_transform = cv2.distanceTransform(pred_mask.astype(np.uint8), cv2.DIST_L2, 5)
#
#     # 找到距离变换的最大值及其位置
#     max_width = float(np.max(distance_transform))
#
#     # 找到最大宽度对应的坐标（如果有多个最大值，取第一个）
#     max_positions = np.where(distance_transform == np.max(distance_transform))
#     if len(max_positions[0]) > 0:
#         max_y = int(max_positions[0][0])
#         max_x = int(max_positions[1][0])
#     else:
#         # 如果没有找到最大值（理论上不会发生），取中心点
#         y_coords, x_coords = np.where(pred_mask > 0)
#         if len(y_coords) > 0:
#             max_y = int(np.mean(y_coords))
#             max_x = int(np.mean(x_coords))
#         else:
#             max_y, max_x = 0, 0
#
#     return crack_length, max_width, (max_y, max_x)
#
#
# # 5. 处理预测结果：返回每条裂缝的长度、宽度、最大宽度位置
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
#     max_width_positions = []  # 存储最大宽度位置
#
#     for label_id in range(1, num_labels + 1):
#         pred_mask = (labeled_mask == label_id).astype(np.uint8)
#         L, W, position = calculate_crack_length_and_width_with_position(pred_mask)
#         crack_length_list.append(L)
#         crack_width_list.append(W)
#         max_width_positions.append(position)
#
#     return crack_length_list, crack_width_list, labeled_mask, max_width_positions
#
#
# # 6. 将编号和最大宽度位置叠加到图像上并保存
# def save_numbered_overlay_with_positions(labeled_mask, base_image_pil, lengths, widths,
#                                          max_width_positions, save_path="crack_overlay_numbered.png"):
#     """
#     labeled_mask: ndarray(H,W) 的连通域编号图（0为背景, 1..N为裂缝ID）
#     base_image_pil: 已缩放到与mask同尺寸的PIL图（448×448）
#     lengths, widths: 与ID顺序一致的列表
#     max_width_positions: 每个裂缝最大宽度位置的列表 [(y1,x1), (y2,x2), ...]
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
#
#         # 计算裂缝中心点（用于显示编号）
#         cy, cx = int(ys.mean()), int(xs.mean())
#
#         # 获取最大宽度位置
#         if label_id - 1 < len(max_width_positions):
#             max_y, max_x = max_width_positions[label_id - 1]
#
#             # 在最大宽度位置画一个红色圆点（更醒目）
#             cv2.circle(overlay_bgr, (max_x, max_y), 8, (0, 0, 255), -1)  # 红色大圆点
#             cv2.circle(overlay_bgr, (max_x, max_y), 10, (0, 0, 255), 2)  # 红色外圈
#
#             # 在最大宽度位置添加宽度值标注
#             width_text = f"W={widths[label_id - 1]:.2f}"
#             cv2.putText(overlay_bgr, width_text, (max_x + 12, max_y - 8),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
#
#         # 画一个小圆心与编号（在裂缝中心位置）
#         cv2.circle(overlay_bgr, (cx, cy), 4, (255, 255, 255), -1)
#         cv2.putText(overlay_bgr, str(label_id), (cx + 6, cy - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
#
#     cv2.imwrite(save_path, overlay_bgr)
#     return save_path
#
#
# # 7. 保存详细的裂缝分析报告（包含最大宽度位置）
# def save_detailed_report(report_path, total_count, crack_lengths, crack_widths,
#                          max_width_positions, image_path):
#     """
#     保存详细的裂缝分析报告
#     """
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write("=" * 60 + "\n")
#         f.write("裂缝检测详细报告\n")
#         f.write("=" * 60 + "\n")
#         f.write(f"图像路径: {image_path}\n")
#         f.write(f"总裂缝数: {total_count}\n")
#         f.write("\n" + "-" * 60 + "\n")
#
#         for i in range(total_count):
#             f.write(f"\n裂缝 #{i + 1}:\n")
#             f.write(f"  - 长度: {crack_lengths[i]} 像素\n")
#             f.write(f"  - 最大宽度: {crack_widths[i]:.4f} 像素\n")
#
#             if i < len(max_width_positions):
#                 y, x = max_width_positions[i]
#                 f.write(f"  - 最大宽度位置: 坐标 ({x}, {y})\n")
#                 f.write(f"  - 位置说明: 第{y}行, 第{x}列\n")
#
#             # 添加一些额外的建议
#             if crack_widths[i] > 0:
#                 f.write(f"  - 宽度评估: ")
#                 if crack_widths[i] < 1.0:
#                     f.write("微小裂缝\n")
#                 elif crack_widths[i] < 3.0:
#                     f.write("中等裂缝\n")
#                 else:
#                     f.write("较宽裂缝\n")
#
#         f.write("\n" + "=" * 60 + "\n")
#         f.write("注: 坐标位置为图像坐标系 (左上角为原点)\n")
#         f.write("最大宽度位置已用红色圆点标记在叠加图中\n")
#         f.write("=" * 60 + "\n")
#
#
# # 8. 主程序：加载模型，推理，计算并编号可视化
# def main(image_path, model_path, n_classes=2):
#     print("=" * 50)
#     print("程序开始运行...")
#     print(f"使用模型: MLiteUNet")
#     print(f"类别数: {n_classes}")
#
#     # 1. 加载模型（返回模型和设备）
#     model, device = load_trained_model(model_path, n_classes=n_classes)
#     print(f"模型加载成功！使用设备: {device}")
#
#     # 2. 预处理（返回与mask同尺寸的img_resized用于可视化）
#     img_tensor, img_resized = preprocess_image(image_path, device)
#
#     # 3. 推理并保存掩码（crack_mask.png）
#     mask_path = "crack_mask.png"
#     crack_mask = get_crack_mask(model, img_tensor, device, save_path=mask_path)
#     print("掩码生成完成！")
#
#     # 4. 计算每条裂缝的长度与宽度，并获取最大宽度位置
#     crack_lengths, crack_widths, labeled_mask, max_width_positions = process_predictions(crack_mask)
#
#     total_count = len(crack_lengths)
#     print(f"裂缝数量检测完成！共发现 {total_count} 条裂缝")
#
#     # 打印每条裂缝的详细信息
#     if total_count > 0:
#         for i, (L, W, pos) in enumerate(zip(crack_lengths, crack_widths, max_width_positions), start=1):
#             y, x = pos
#             print(f"  裂缝 #{i}: 长度={L}像素, 最大宽度={W:.4f}像素, 最大宽度位置=({x}, {y})")
#     else:
#         print("警告：未检测到任何裂缝！")
#         print("请检查：")
#         print("  1. 图片是否包含裂缝")
#         print("  2. 模型是否适用于当前图片类型")
#         print("  3. 图片预处理是否正确")
#
#     # 5. 可视化编号叠加并保存（包含最大宽度位置标记）
#     overlay_path = "crack_overlay_numbered.png"
#     if total_count > 0:
#         overlay_path = save_numbered_overlay_with_positions(
#             labeled_mask, img_resized, crack_lengths, crack_widths,
#             max_width_positions, save_path=overlay_path
#         )
#         print("编号叠加图生成完成！（红色圆点标记最大宽度位置）")
#     else:
#         # 即使没有裂缝，也保存一张原始图片
#         overlay_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
#         cv2.imwrite(overlay_path, overlay_bgr)
#         print("未检测到裂缝，保存原始图像")
#
#     # 6. 输出 & 保存文本结果（crack_results.txt）
#     print(f"\n总裂缝数: {total_count}")
#     for i, (L, W) in enumerate(zip(crack_lengths, crack_widths), start=1):
#         print(f"#{i}: 长度={L} 像素, 宽度(maxDist)={W:.4f} 像素")
#
#     report_path = "crack_results.txt"
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(f"总裂缝数: {total_count}\n")
#         for i, (L, W) in enumerate(zip(crack_lengths, crack_widths), start=1):
#             f.write(f"#{i}: 长度={L} 像素, 宽度(maxDist)={W:.4f} 像素\n")
#
#     # 7. 保存详细的裂缝分析报告
#     detailed_report_path = "crack_detailed_report.txt"
#     if total_count > 0:
#         save_detailed_report(detailed_report_path, total_count, crack_lengths,
#                              crack_widths, max_width_positions, image_path)
#         print(f"详细报告已保存: {os.path.abspath(detailed_report_path)}")
#     else:
#         with open(detailed_report_path, "w", encoding="utf-8") as f:
#             f.write("未检测到任何裂缝\n")
#             f.write(f"图像路径: {image_path}\n")
#             f.write("请检查输入图片或模型配置\n")
#         print(f"无裂缝检测报告已保存: {os.path.abspath(detailed_report_path)}")
#
#     print(f"\n掩码已保存: {os.path.abspath(mask_path)}")
#     print(f"编号覆盖图已保存: {os.path.abspath(overlay_path)}")
#     print(f"结果文本已保存: {os.path.abspath(report_path)}")
#     print("=" * 50)
#     print("程序运行完成！")
#
#     if total_count > 0:
#         print("\n提示：")
#         print("  - 红色圆点标记了每条裂缝的最大宽度位置")
#         print("  - 黄色文字显示了该位置的具体宽度值")
#         print("  - 详细报告包含每个裂缝的完整信息")
#     else:
#         print("\n提示：未检测到裂缝，请检查输入图片是否正确")
#
#
# # 示例：运行主程序
# if __name__ == "__main__":
#     # 使用Windows路径（根据您的系统）
#     image_path = r"./examples/test_image.jpg"
#     model_path = r"./checkpoints/model_weights.pth"
#
#     # 类别数（背景+裂缝=2）
#     NUM_CLASSES = 2
#
#     # 检查文件是否存在
#     if not os.path.exists(image_path):
#         print(f"错误：图片文件不存在 - {image_path}")
#     elif not os.path.exists(model_path):
#         print(f"错误：模型文件不存在 - {model_path}")
#     else:
#         main(image_path, model_path, n_classes=NUM_CLASSES)




#  二
# 整幅图像裂缝宽度、长度数量检测并保存（支持全图预测或ROI指定预测）
# 裂缝长度计算用skeletonize()做骨架，再用骨架像素数作为长度
# import os
# import torch
# import numpy as np
# from PIL import Image
# import argparse
# import cv2
# from model.ML_Mutil import MLiteUNet  # 修改为您的模型
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
#     # 如果输入是0/255，将其规整为0/1
#     if crack_mask_cpu.max() > 1:
#         crack_mask_cpu = (crack_mask_cpu > 0).astype(np.uint8)
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
# # === 加载模型（适配MLiteUNet） ===
# def load_model(model_path, n_classes=2):
#     """
#     加载MLiteUNet模型
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#
#     # 创建模型实例
#     model = MLiteUNet(n_classes=n_classes, aux_mode='eval', pretrained_backbone=False)
#
#     # 加载权重
#     if device.type == 'cuda':
#         state_dict = torch.load(model_path)
#         model.load_state_dict(state_dict, strict=False)
#         model = model.cuda()
#     else:
#         state_dict = torch.load(model_path, map_location=torch.device('cpu'))
#         model.load_state_dict(state_dict, strict=False)
#         model = model.cpu()
#
#     model.eval()
#     return model, device
#
#
# # === 滑窗预测函数，限制 ROI 范围 ===
# def sliding_window_predict_roi(model, img_tensor, roi_coords, window_size=448, overlap=0.2, cls=1,
#                                save_blocks_dir=None):
#     _, _, H, W = img_tensor.shape
#     x1, y1, x2, y2 = roi_coords
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
#                 # 模型推理
#                 pred = model(crop)
#                 # 处理模型输出（如果是元组，取第一个）
#                 if isinstance(pred, tuple):
#                     pred = pred[0]
#                 pred = torch.argmax(pred, dim=1)
#                 pred_crop = pred[:, :bottom - top, :right - left].squeeze(0).cpu().numpy()
#                 pred_crop_bin = (pred_crop == cls).astype(np.uint8) * 255
#
#                 # 保存单独的mask
#                 if save_blocks_dir:
#                     os.makedirs(save_blocks_dir, exist_ok=True)
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
#     parser = argparse.ArgumentParser(description='裂缝检测与量化（支持全图预测或ROI指定预测）')
#     parser.add_argument('--image', type=str, required=True, help='输入图像路径')
#     parser.add_argument('--weight', type=str, required=True, help='模型权重路径')
#     parser.add_argument('--n-classes', type=int, default=2, help='类别数（背景+裂缝）')
#     parser.add_argument('--crop-size', type=int, default=448, help='滑窗大小')
#     parser.add_argument('--overlap', type=float, default=0.2, help='滑窗重叠率')
#     parser.add_argument('--save-dir', type=str, default='results/single_prediction', help='结果保存目录')
#     parser.add_argument('--roi', type=int, nargs=4, metavar=('x1', 'y1', 'x2', 'y2'),
#                         default=None, help='【可选】ROI区域: 左上角(x1,y1) 右下角(x2,y2)，不指定则全图预测')
#     parser.add_argument('--save-blocks-dir', type=str, default='roi_blocks', help='保存每个滑窗掩码和对比图的目录')
#     parser.add_argument('--output-file', type=str, default='crack_results.txt',
#                         help='保存裂缝数量、长度和宽度的文本文件')
#     args = parser.parse_args()
#
#     os.makedirs(args.save_dir, exist_ok=True)
#     if args.save_blocks_dir:
#         os.makedirs(args.save_blocks_dir, exist_ok=True)
#
#     # === 加载模型（适配MLiteUNet） ===
#     model, device = load_model(args.weight, n_classes=args.n_classes)
#
#     # === 加载图像 ===
#     img_tensor, orig_pil = load_image(args.image)
#     img_tensor = img_tensor.to(device)
#     orig_np = np.array(orig_pil.convert("RGB"))
#
#     # 获取图像尺寸
#     _, _, H, W = img_tensor.shape
#     print(f"图像尺寸: {W} x {H}")
#
#     # === 判断预测模式 ===
#     if args.roi:
#         # ROI模式
#         x1, y1, x2, y2 = args.roi
#         print(f"ROI 限定区域: 左上({x1},{y1})，右下({x2},{y2})")
#
#         # 检查ROI是否在图像范围内
#         if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
#             print(f"警告: ROI超出图像范围! 图像尺寸为 {W}x{H}")
#             print(f"自动裁剪ROI到图像范围内...")
#             x1 = max(0, x1)
#             y1 = max(0, y1)
#             x2 = min(W, x2)
#             y2 = min(H, y2)
#             print(f"调整后ROI: 左上({x1},{y1})，右下({x2},{y2})")
#
#         pred_mask_np = sliding_window_predict_roi(
#             model, img_tensor,
#             roi_coords=(x1, y1, x2, y2),
#             window_size=args.crop_size,
#             overlap=args.overlap,
#             save_blocks_dir=args.save_blocks_dir
#         )
#     else:
#         # 全图模式
#         print("全图预测模式")
#         pred_mask_np = sliding_window_predict_roi(
#             model, img_tensor,
#             roi_coords=(0, 0, W, H),
#             window_size=args.crop_size,
#             overlap=args.overlap,
#             save_blocks_dir=args.save_blocks_dir
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
#     print(f"\n{'=' * 50}")
#     print(f"检测结果:")
#     print(f"裂缝数量: {crack_count}")
#     if crack_count > 0:
#         print(f"\n各裂缝详细信息:")
#         print(f"{'编号':<6} {'长度(像素)':<12} {'宽度(像素)':<12}")
#         print("-" * 30)
#         for i, (L, W) in enumerate(zip(crack_length, crack_width), start=1):
#             print(f"{i:<6} {L:<12} {W:<12.4f}")
#     print(f"{'=' * 50}\n")
#
#     # === 保存裂缝数据到文本文件 ===
#     output_file = os.path.join(args.save_dir, args.output_file)
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write(f"图像路径: {args.image}\n")
#         if args.roi:
#             f.write(f"ROI区域: ({x1},{y1}) - ({x2},{y2})\n")
#         else:
#             f.write(f"预测模式: 全图预测\n")
#         f.write(f"裂缝数量: {crack_count}\n")
#         f.write("编号, 长度(像素), 宽度(像素, 距离变换最大值)\n")
#         for i, (L, W) in enumerate(zip(crack_length, crack_width), start=1):
#             f.write(f"{i}, {L}, {W:.4f}\n")
#
#     print(f"✅ 预测完成：")
#     print(f"  - 掩码图: {pred_path}")
#     print(f"  - 可视化图: {overlay_path}")
#     print(f"  - 裂缝数据: {output_file}")
#
#
# if __name__ == '__main__':
#     main()




#  三
# 整幅图像裂缝宽度、长度数量检测并保存（带编号叠加）（支持全图预测或ROI指定预测）
# 用 Zhang–Suen 细化（thin()）做骨架，再用骨架像素数作为长度。
import os
import torch
import numpy as np
from PIL import Image
import argparse
import cv2
from model.ML_Mutil import MLiteUNet  # 修改为您的模型
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


# === 加载模型（适配MLiteUNet） ===
def load_model(model_path, n_classes=2):
    """
    加载MLiteUNet模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型实例
    model = MLiteUNet(n_classes=n_classes, aux_mode='eval', pretrained_backbone=False)

    # 加载权重
    if device.type == 'cuda':
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model = model.cpu()

    model.eval()
    return model, device


# === 滑窗预测函数，限制 ROI 范围 ===
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
                # 处理模型输出（如果是元组，取第一个）
                if isinstance(pred, tuple):
                    pred = pred[0]
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
    parser = argparse.ArgumentParser(description='裂缝检测与量化（带编号叠加，支持全图预测或ROI指定预测）')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--weight', type=str, required=True, help='模型权重路径')
    parser.add_argument('--n-classes', type=int, default=2, help='类别数（背景+裂缝）')
    parser.add_argument('--crop-size', type=int, default=448, help='滑窗大小')
    parser.add_argument('--overlap', type=float, default=0.2, help='滑窗重叠率')
    parser.add_argument('--save-dir', type=str, default='results/single_prediction', help='结果保存目录')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('x1', 'y1', 'x2', 'y2'),
                        default=None, help='【可选】ROI区域: 左上角(x1,y1) 右下角(x2,y2)，不指定则全图预测')
    parser.add_argument('--save-blocks-dir', type=str, default='roi_blocks', help='保存每个滑窗掩码和对比图的目录')
    parser.add_argument('--output-file', type=str, default='crack_results.txt',
                        help='保存裂缝数量、长度和宽度的文本文件')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_blocks_dir:
        os.makedirs(args.save_blocks_dir, exist_ok=True)

    # === 加载模型（适配MLiteUNet） ===
    model, device = load_model(args.weight, n_classes=args.n_classes)

    # === 加载图像 ===
    img_tensor, orig_pil = load_image(args.image)
    img_tensor = img_tensor.to(device)
    orig_np = np.array(orig_pil.convert("RGB"))

    # 获取图像尺寸
    _, _, H, W = img_tensor.shape
    print(f"图像尺寸: {W} x {H}")

    # === 判断预测模式 ===
    if args.roi:
        # ROI模式
        x1, y1, x2, y2 = args.roi
        print(f"ROI 限定区域: 左上({x1},{y1})，右下({x2},{y2})")

        # 检查ROI是否在图像范围内
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            print(f"警告: ROI超出图像范围! 图像尺寸为 {W}x{H}")
            print(f"自动裁剪ROI到图像范围内...")
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            print(f"调整后ROI: 左上({x1},{y1})，右下({x2},{y2})")

        pred_mask_np = sliding_window_predict_roi(
            model, img_tensor,
            roi_coords=(x1, y1, x2, y2),
            window_size=args.crop_size,
            overlap=args.overlap,
            save_blocks_dir=args.save_blocks_dir
        )
    else:
        # 全图模式
        print("全图预测模式")
        pred_mask_np = sliding_window_predict_roi(
            model, img_tensor,
            roi_coords=(0, 0, W, H),
            window_size=args.crop_size,
            overlap=args.overlap,
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
    overlay_numbered = overlay_img.copy()
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
    print(f"\n{'=' * 50}")
    print(f"检测结果:")
    print(f"裂缝数量: {crack_count}")
    if crack_count > 0:
        print(f"\n各裂缝详细信息:")
        print(f"{'编号':<6} {'长度(像素)':<12} {'宽度(像素)':<12}")
        print("-" * 30)
        for i, (L, W) in enumerate(zip(crack_length, crack_width), start=1):
            print(f"{i:<6} {L:<12} {W:<12.4f}")
    print(f"{'=' * 50}\n")

    # === 保存裂缝数据到文本文件（逐条对应编号-长度-宽度） ===
    output_file = os.path.join(args.save_dir, args.output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"图像路径: {args.image}\n")
        if args.roi:
            f.write(f"ROI区域: ({x1},{y1}) - ({x2},{y2})\n")
        else:
            f.write(f"预测模式: 全图预测\n")
        f.write(f"裂缝数量: {crack_count}\n")
        f.write("编号, 长度(像素), 宽度(像素, 距离变换最大值)\n")
        for i, (L, W) in enumerate(zip(crack_length, crack_width), start=1):
            f.write(f"{i}, {L}, {W:.4f}\n")

    print(f"✅ 预测完成：")
    print(f"  - 掩码图: {pred_path}")
    print(f"  - 覆盖图: {overlay_path}")
    print(f"  - 编号覆盖图: {overlay_numbered_path}")
    print(f"  - 裂缝数据: {output_file}")


if __name__ == '__main__':
    print("=== 当前执行文件 ===")
    print(os.path.abspath(__file__))
    print("=== 开始进入 main() ===")
    main()


# 第一段代码：专注于单张图像的裂缝检测，计算和保存裂缝的长度与宽度。
# 第二段代码：支持批量处理和滑窗预测，适用于大图像，计算裂缝的数量和特征。
# 第三段代码：与第二段相似，但使用不同的细化算法，强调裂缝的骨架化处理。


