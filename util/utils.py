import numpy as np
import random
import torch
import os
import cv2
from sklearn.utils.class_weight import compute_class_weight
from thop import profile
import pickle
from pathlib import Path
import time


def calculate_flops_and_params(model, input_size=(1, 3, 448, 448), device='cuda'):

    model.to(device)
    if len(input_size) == 5:
        input_size = input_size[1:]

    input = torch.randn(*input_size).to(device)

    flops, params = profile(model, (input,))

    return flops, params


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def calc_crack_pixel_weight(data_dir):
    print('Computing class weights...')
    cweight_path = data_dir + '/cweight.pkl'

    if os.path.exists(cweight_path):
        print('Loading saved class weights.')
        with open(cweight_path, 'rb') as f:
            weight = pickle.load(f)
    else:
        files = []
        for path in Path(data_dir + '/SegmentationClass').glob('*.*'):
            label = cv2.imread(str(path)).astype(np.uint8)
            if 2 not in np.unique(label):
                files.append(label)
        all_arr = np.stack(files, axis=0)[:, :, :, 0]
        weight = compute_class_weight(class_weight='balanced', classes=np.unique(label), y=all_arr.flatten())
        with open(cweight_path, 'wb') as f:
            pickle.dump(weight, f)
        print('Saved class weights under dataset path.')

    return weight

class mIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self, return_class_iou=False):
        intersection = np.diag(self.hist)
        union = self.hist.sum(axis=1) + self.hist.sum(axis=0) - intersection

        valid_indices = union > 0
        if not valid_indices.any():
            print("⚠️  Warning: no ground truth or prediction in this batch")
            if return_class_iou:
                return float('nan'), [0.0 for _ in range(self.num_classes)]
            else:
                return float('nan')

        iu = np.zeros(self.num_classes)
        iu[valid_indices] = intersection[valid_indices] / union[valid_indices]
        mean_iou = np.nanmean(iu)

        if return_class_iou:
            return mean_iou, iu.tolist()
        else:
            return mean_iou

def precision_recall_f1_acc(preds, gts, num_classes=2, ignore_index=255):
    """
    支持二分类/多分类分割的 Pixel-Level Precision, Recall, F1, Accuracy 计算。
    自动支持 torch.Tensor 或 numpy 输入。
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(gts, torch.Tensor):
        gts = gts.detach().cpu().numpy()

    preds = preds.flatten()
    gts = gts.flatten()

    mask = (gts != ignore_index)
    preds = preds[mask]
    gts = gts[mask]

    precision_list, recall_list, f1_list = [], [], []
    acc = (preds == gts).sum() / len(gts)

    for cls in range(num_classes):
        tp = np.sum((preds == cls) & (gts == cls))
        fp = np.sum((preds == cls) & (gts != cls))
        fn = np.sum((preds != cls) & (gts == cls))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)

    return precision, recall, f1, acc

def compute_fps(model, sample_input, device='cuda', warmup=10, trials=30):
    """
    评估模型FPS（每秒帧数），只用于inference阶段
    """
    model.eval()
    model = model.to(device)
    sample_input = sample_input.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

    # 正式计时
    start = time.time()
    with torch.no_grad():
        for _ in range(trials):
            _ = model(sample_input)
    end = time.time()

    avg_time = (end - start) / trials
    fps = 1.0 / avg_time
    return fps

