import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_binary_mask(mask_path, threshold=128):
    """
    读取mask并转为二值矩阵
    裂缝=1，背景=0
    默认认为白色区域是裂缝，黑色区域是背景
    """
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)
    binary = (arr >= threshold).astype(np.uint8)
    return binary, img.size

def compute_error_map(gt_mask, pred_mask):
    """
    输入：
        gt_mask: Ground Truth二值矩阵
        pred_mask: 预测结果二值矩阵
    输出：
        error_rgb: 误差图RGB数组
        iou: 单张图像IoU
        f1: 单张图像F1
    """

    # 四类像素
    tp = (gt_mask == 1) & (pred_mask == 1)   # 裂缝正确识别
    fp = (gt_mask == 0) & (pred_mask == 1)   # 背景误检为裂缝
    fn = (gt_mask == 1) & (pred_mask == 0)   # 裂缝漏检
    tn = (gt_mask == 0) & (pred_mask == 0)   # 背景正确识别

    # 统计像素数
    TP = np.sum(tp)
    FP = np.sum(fp)
    FN = np.sum(fn)
    TN = np.sum(tn)

    # 计算IoU和F1
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

    # 生成误差图
    # 白色=TP, 红色=FP, 青色=FN, 黑色=TN
    error_rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    error_rgb[tp] = [255, 255, 255]   # TP -> 白色
    error_rgb[fp] = [255, 0, 0]       # FP -> 红色
    error_rgb[fn] = [0, 255, 255]     # FN -> 青色
    error_rgb[tn] = [0, 0, 0]         # TN -> 黑色

    return error_rgb, iou, f1, TP, FP, FN, TN

def save_error_map(error_rgb, save_path):
    """
    保存纯误差图
    """
    img = Image.fromarray(error_rgb)
    img.save(save_path)

def save_error_map_with_text(error_rgb, save_path, iou, f1, title="Error Map"):
    """
    保存带标题和IoU/F1文字的误差图
    """
    error_img = Image.fromarray(error_rgb)

    w, h = error_img.size
    margin_top = 60
    margin_bottom = 80

    canvas = Image.new("RGB", (w, h + margin_top + margin_bottom), "white")
    canvas.paste(error_img, (0, margin_top))

    draw = ImageDraw.Draw(canvas)

    # 如果系统没有Times New Roman，会自动退回默认字体
    try:
        font_title = ImageFont.truetype("times.ttf", 28)
        font_text = ImageFont.truetype("times.ttf", 24)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    draw.text((20, 15), title, fill="black", font=font_title)
    draw.text((20, h + margin_top + 20), f"IoU = {iou:.4f}    F1 = {f1:.4f}", fill="black", font=font_text)

    canvas.save(save_path)

def generate_error_map(gt_path, pred_path, save_dir, prefix="result"):
    """
    主函数：从GT和预测mask生成误差图
    """
    os.makedirs(save_dir, exist_ok=True)

    gt_mask, gt_size = load_binary_mask(gt_path)
    pred_mask, pred_size = load_binary_mask(pred_path)

    # 尺寸检查
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"GT和预测mask尺寸不一致！GT尺寸={gt_mask.shape}, Pred尺寸={pred_mask.shape}"
        )

    error_rgb, iou, f1, TP, FP, FN, TN = compute_error_map(gt_mask, pred_mask)

    # 保存纯误差图
    pure_save_path = os.path.join(save_dir, f"{prefix}_error_map.png")
    save_error_map(error_rgb, pure_save_path)

    # 保存带文字的误差图
    text_save_path = os.path.join(save_dir, f"{prefix}_error_map_with_text.png")
    save_error_map_with_text(
        error_rgb,
        text_save_path,
        iou,
        f1,
        title=f"{prefix} Pixel-wise Error Map"
    )

    print(f"误差图已保存：{pure_save_path}")
    print(f"带文字误差图已保存：{text_save_path}")
    print(f"TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}")
    print(f"IoU = {iou:.6f}")
    print(f"F1  = {f1:.6f}")

    return iou, f1

if __name__ == "__main__":
    # ======= 你只需要修改这里的路径 =======

    gt_path = r"C:\Users\15176\Desktop\error\ctc\truth.png"                # Ground Truth mask
    pred_path = r"C:\Users\15176\Desktop\error\ctc\mds.png"     # 预测 mask（比如MDSSegNet）
    save_dir = r"C:\Users\15176\Desktop\error\ctc"
    prefix = "MDSSegNet"

    generate_error_map(gt_path, pred_path, save_dir, prefix)