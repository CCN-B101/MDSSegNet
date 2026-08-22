# 1. 基本预测（全图滑窗预测）
python Prediction.py --image "./examples/test_image.jpg" --weight "./checkpoints/model_weights.pth" --save-dir "./results"

# 2. 指定ROI区域预测（仅预测图像的特定区域）
python Prediction.py --image "./examples/test_image.jpg" --weight "./checkpoints/model_weights.pth" --save-dir "./results" --roi 100 100 800 800

# 3. 调整阈值和分类参数（阈值越低，裂缝检测越敏感）
python Prediction.py --image "./examples/test_image.jpg" --weight "./checkpoints/model_weights.pth" --save-dir "./results" --threshold 0.3 --class-idx 1 --n-classes 2

# 4. 自定义滑窗参数（调整滑窗大小和重叠率）
python Prediction.py --image "./examples/test_image.jpg" --weight "./checkpoints/model_weights.pth" --crop-size 512 --overlap 0.3 --save-dir "./results"
