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
#     labem���-�G����ƭy�eval()
    metric = mIOU(num_classes=2)
    preds_all, gts_all = [], []

    with torch.no_grad():
        for img, mask in tqdm(dataloader, desc="Validating", leave=False):
            img = img.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            out = model(img)[0]
            pred = torch.argmax(out, dim=1)

            pred_np = pred.cpu().numpy()
            gt_np = mask.cpu().numpy()
            preds_all.append(pred_np)
            gts_all.append(gt_np)
            metric.add_batch(pred_np, gt_np)

    miou, class_ious = metric.evaluate(return_class_iou=True)

    preds_np = np.concatenate(preds_all, axis=0).reshape(-1)
    gts_np = np.concatenate(gts_all, axis=0).reshape(-1)
    precision, recall, f1, accuracy = precision_recall_f1_acc(preds_np, gts_np)

    return {
        "mIoU": float(miou),
        "ClassIoU": class_ious,
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def main():
    args = parse_args()
    set_reproducibility(args.seed)

    size = tuple(args.crop_size)

    train_img_dir = os.path.join(args.data_root, "train", "images")
    train_mask_dir = os.path.join(args.data_root, "train", "masks")
    val_img_dir = os.path.join(args.data_root, "val", "images")
    val_mask_dir = os.path.join(args.data_root, "val", "masks")

    for p in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Required directory not found: {p}")

    # IMPORTANT:
    # - Training randomness changes with args.seed.
    # - Validation input itself must remain fixed across runs.
    #   The accompanying transform.py makes val_transform deterministic.
    trainset = CrackDataset(
        train_img_dir, train_mask_dir,
        size=size, mode="train", seed=args.seed
    )
    valset = CrackDataset(
        val_img_dir, val_mask_dir,
        size=size, mode="val", seed=None
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator,
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )

    model = MLiteUNet(
        n_classes=2,
        aux_mode="train",
        pretrained_backbone=False
    ).cuda()

    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Total Parameters: {count_params(model):.2f} M")

    model = DataParallel(model).cuda()

    criterion = CombinedLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path(args.checkpoint_root) / args.dataset / f"seed_{args.seed}"
    result_dir = Path(args.result_root) / args.dataset / f"seed_{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_weight_path = checkpoint_dir / "best_mIoU.pth"
    csv_path = result_dir / "train_metrics.csv"
    log_path = result_dir / "train.log"

    headers = [
        "Epoch", "Loss", "mIoU", "Accuracy",
        "Precision", "Recall", "F1", "LearningRate"
    ]

    best_miou = -1.0
    best_epoch = -1

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile, \
         open(log_path, "w", encoding="utf-8") as logfile:

        writer = csv.writer(csvfile)
        writer.writerow(headers)

        logfile.write(f"Dataset: {args.dataset}\n")
        logfile.write(f"Seed: {args.seed}\n")
        logfile.write(f"Epochs: {args.epochs}\n")
        logfile.write(f"Batch size: {args.batch_size}\n")
        logfile.write(f"Learning rate: {args.lr}\n")
        logfile.write(f"Weight decay: {args.weight_decay}\n\n")

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0

            tbar = tqdm(
                enumerate(trainloader),
                total=len(trainloader),
                desc=f"{args.dataset} seed={args.seed} epoch={epoch}/{args.epochs}"
            )

            for i, (img, mask) in tbar:
                img = img.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                output, out_s3, out_s4, out_s5 = model(img)

                # Correct multi-scale deep supervision:
                # main output + three genuine auxiliary outputs.
                loss1 = criterion(output, mask)
                loss2 = criterion(out_s3, mask)
                loss3 = criterion(out_s4, mask)
                loss4 = criterion(out_s5, mask)

                loss = (
                    0.5 * loss1
                    + 0.2 * loss2
                    + 0.2 * loss3
                    + 0.1 * loss4
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                tbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}")

            metrics = evaluate(model, valloader)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch,
                total_loss / len(trainloader),
                metrics["mIoU"],
                metrics["Accuracy"],
                metrics["Precision"],
                metrics["Recall"],
                metrics["F1"],
                current_lr,
            ])
            csvfile.flush()

            logfile.write(
                f"Epoch {epoch:03d} | "
                f"Loss={total_loss / len(trainloader):.6f} | "
                f"mIoU={metrics['mIoU']:.6f} | "
                f"Precision={metrics['Precision']:.6f} | "
                f"Recall={metrics['Recall']:.6f} | "
                f"F1={metrics['F1']:.6f} | "
                f"Accuracy={metrics['Accuracy']:.6f} | "
                f"LR={current_lr:.8f}\n"
            )
            logfile.flush()

            # Use one fixed model-selection rule for every run:
            # best validation mIoU.
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                best_epoch = epoch
                torch.save(model.module.state_dict(), best_weight_path)

                print(
                    f"Best checkpoint updated: epoch={best_epoch}, "
                    f"val mIoU={best_miou:.6f} -> {best_weight_path}"
                )

            scheduler.step()

        logfile.write(
            f"\nTraining complete. Best epoch={best_epoch}, "
            f"Best validation mIoU={best_miou:.6f}\n"
        )

    print("\nTraining complete")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation mIoU: {best_miou:.6f}")
    print(f"Checkpoint: {best_weight_path}")
    print(f"Training log: {log_path}")
    print(f"Training CSV: {csv_path}")


if __name__ == "__main__":
    main()
