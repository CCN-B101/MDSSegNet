import json
import numpy as np
import cv2
import os
import glob

def create_mask_from_json(json_file, img_path):
    """
    从 LabelMe 导出的 JSON 文件生成掩膜图像，标定区域为白线，背景为黑色。
    参数：
    - json_file: LabelMe 导出的 JSON 文件路径
    - img_path: 对应的原始图像路径，用来获取图像尺寸
    
    返回：
    - mask: 掩膜图像，白线为标定区域，黑色为背景
    """
    # 读取原始图像并获取其尺寸（高、宽）
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # 打开 JSON 文件并加载数据
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建一个与原图大小相同的黑色背景图（初始化为全0）
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 遍历 JSON 中的每一个标定区域
    for shape in data['shapes']:
        shape_type = shape['shape_type']
        points = shape['points']  # 点的坐标，通常是一个 list 列表
        
        if shape_type == 'linestrip':
            # 线条标注，将点序列作为线条连接
            points = np.array(points, dtype=np.int32)
            cv2.polylines(mask, [points], isClosed=False, color=255, thickness=6)
        
        elif shape_type == 'polygon':
            # 多边形标注，封闭的多边形
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], color=255)
    
    return mask

def save_mask_as_image(mask, output_path):
    """
    保存掩膜图像到文件
    """
    cv2.imwrite(output_path, mask)
    print(f"掩膜图像已保存: {output_path}")

def batch_convert_masks(input_dir, image_extension='.jpg'):
    """
    批量转换目录中的LabelMe JSON文件为掩膜图像
    参数：
    - input_dir: 包含图像和JSON文件的目录路径
    - image_extension: 图像文件扩展名，默认为'.jpg'
    """
    # 查找目录中所有的图像文件
    image_pattern = os.path.join(input_dir, f'*{image_extension}')
    image_files = glob.glob(image_pattern)
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    success_count = 0
    for img_path in image_files:
        # 获取对应的JSON文件路径
        base_name = os.path.splitext(img_path)[0]
        json_path = base_name + '.json'
        
        # 检查JSON文件是否存在
        if not os.path.exists(json_path):
            print(f"警告: 找不到对应的JSON文件 {json_path}")
            continue
            
        try:
            # 生成掩膜图像
            mask = create_mask_from_json(json_path, img_path)
            
            # 保存掩膜图像到原始图像所在目录
            mask_output_path = os.path.join(input_dir, f"{os.path.basename(base_name)}.png")
            save_mask_as_image(mask, mask_output_path)
            success_count += 1
            
        except Exception as e:
            print(f"处理文件 {img_path} 时出错: {e}")
    
    print(f"批量转换完成，成功处理 {success_count} 个文件")

# ==== 示例用法 ====
if __name__ == '__main__':
    # 单个文件转换示例
    json_file = './examples/.json'  # 你的 LabelMe JSON 文件路径
    img_path = './examples/test_image.jpg'  # 原始图像路径

    # 生成掩膜图像
    mask = create_mask_from_json(json_file, img_path)

    # 保存掩膜图像到原始图像所在目录
    img_dir = os.path.dirname(img_path)
    mask_output_path = os.path.join(img_dir, 'mask.png')
    save_mask_as_image(mask, mask_output_path)
    

