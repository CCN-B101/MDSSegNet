# 版本2
# ===== 全图预测模式（不指定 --roi） =====
python Crack_length_and_width_extraction.py \
    --image "./examples/test_image.jpg" \
    --weight "./checkpoints/model_weights.pth" \
    --save-dir ./results/full_image

# ===== ROI指定预测模式（指定 --roi） =====
python Crack_length_and_width_extraction.py \
    --image "./examples/test_image.jpg" \
    --weight "./checkpoints/model_weights.pth" \
    --roi 100 100 800 600 \
    --save-dir ./results/roi_region

# ===== 全图预测 + 自定义滑窗参数 =====
python Crack_length_and_width_extraction.py \
    --image "./examples/test_image.jpg" \
    --weight "./checkpoints/model_weights.pth" \
    --crop-size 512 \
    --overlap 0.3 \
    --save-dir ./results/full_image_custom

# ===== ROI预测 + 保存滑窗中间结果 =====
python Crack_length_and_width_extraction.py \
    --image "./examples/test_image.jpg" \
    --weight "./checkpoints/model_weights.pth" \
    --roi 200 150 1200 900 \
    --save-blocks-dir ./outputs/roi_blocks \
    --save-dir ./results/roi_with_blocks

# 版本3文章中的使用到的算法
# ===== 全图预测模式（不指定 --roi） =====
python Crack_length_and_width_extraction.py \
    --image "./examples/test_image.jpg" \
    --weight "./checkpoints/model_weights.pth" \
    --save-dir ./results/full_image_numbered

# ===== ROI指定预测模式（指定 --roi） =====
python Crack_length_and_width_extraction.py \
    --image "./examples/test_image.jpg" \
    --weight "./checkpoints/model_weights.pth" \
    --roi 100 100 800 600 \
    --save-dir ./results/roi_numbered

# ===== 全图预测 + 自定义参数 =====
python Crack_length_and_width_extraction.py \
    --image "./examples/test_image.jpg" \
    --weight "./checkpoints/model_weights.pth" \
    --crop-size 512 \
    --overlap 0.25 \
    --save-dir ./results/full_numbered_custom

# ===== ROI预测 + 保存滑窗中间结果 =====
python Crack_length_and_width_extraction.py \
    --image "./examples/test_image.jpg" \
    --weight "./checkpoints/model_weights.pth" \
    --roi 0 0 1500 1000 \
    --save-blocks-dir ./outputs/roi_blocks_numbered \
    --save-dir ./results/roi_numbered_blocks
