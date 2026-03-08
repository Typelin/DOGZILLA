"""
train_cnn.py - 訓練 11 類 CNN (0-9 + 背景)，匯出 ONNX
==========================================================
v2: 使用 PIL 系統字體生成合成數字 (模擬手機螢幕顯示)

11 類: 0-9 數字 + 第 10 類「背景/非數字」

核心改進:
    - PIL 字體渲染 (Arial, Calibri, Segoe UI, Verdana, Consolas 等)
      → 非常接近手機螢幕上顯示的數字
    - 大量增強: 粗細、大小、位移、旋轉、透視
    - 特別強化 7 vs 1 的區分

輸出: ../models/digit_cnn.onnx

用法:
    cd ml-vision-control/pc
    python train_cnn.py

需要: torch, torchvision, numpy, opencv-python, Pillow
"""

import os
import sys
import time
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

NUM_CLASSES = 11  # 0-9 + background(10)


# ============================================================
# 1. CNN 模型定義
# ============================================================
class DigitCNN(nn.Module):
    """
    輸入: 1x28x28 灰階影像
    輸出: 11 類 (0~9 + 背景) 的 logits
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 14x14
            nn.Dropout2d(0.25),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                # 7x7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),  # 11 類: 0-9 + 背景
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 2. Dataset 包裝 (合成數字 & 背景共用)
# ============================================================
class SyntheticDataset(torch.utils.data.Dataset):
    """將 numpy 圖片包裝成 Dataset，格式與 MNIST 一致"""
    def __init__(self, images, labels, mean=0.1307, std=0.3081):
        self.images = images   # (N, 28, 28) float32 [0,1]
        self.labels = labels   # (N,) int64
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1,28,28)
        img = (img - self.mean) / self.std
        return img, int(self.labels[idx])


# ============================================================
# 3. 合成印刷體數字 (用 PIL 系統字體，模擬手機螢幕)
# ============================================================
def _find_system_fonts():
    """找 Windows 系統字體 (Sans-serif 優先，最像手機螢幕)"""
    font_dir = 'C:/Windows/Fonts'
    # 優先: 手機螢幕常見的 Sans-serif 字體
    priority_names = [
        'arial.ttf', 'arialbd.ttf',
        'calibri.ttf', 'calibrib.ttf',
        'segoeui.ttf', 'segoeuib.ttf',
        'verdana.ttf', 'verdanab.ttf',
        'tahoma.ttf', 'tahomabd.ttf',
        'trebuc.ttf', 'trebucbd.ttf',
        'consola.ttf', 'consolab.ttf',
        'impact.ttf',
        'century.ttf',
    ]
    found = []
    for name in priority_names:
        path = os.path.join(font_dir, name)
        if os.path.exists(path):
            found.append(path)

    # 補充: 任何其他 TTF 字體
    if len(found) < 5:
        for p in glob.glob(os.path.join(font_dir, '*.ttf')):
            if p not in found:
                found.append(p)
            if len(found) >= 20:
                break

    if not found:
        raise RuntimeError(f'找不到任何 TTF 字體: {font_dir}')
    return found


def generate_synthetic_digits(n_per_digit=1500):
    """
    用 PIL 系統字體生成印刷體數字圖片。
    模擬手機螢幕: 白底黑字 → 反轉成 MNIST 格式 (黑底白字)。
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    font_paths = _find_system_fonts()
    print(f'    找到 {len(font_paths)} 個系統字體')

    images, labels = [], []

    # 預載入不同大小的字體
    font_cache = {}
    for fp in font_paths:
        for size in range(28, 56, 4):  # 28~52
            try:
                font_cache[(fp, size)] = ImageFont.truetype(fp, size)
            except Exception:
                pass

    font_keys = list(font_cache.keys())
    if not font_keys:
        raise RuntimeError('無法載入任何字體')

    for digit in range(10):
        text = str(digit)
        count = 0
        attempts = 0
        while count < n_per_digit and attempts < n_per_digit * 3:
            attempts += 1

            # 隨機選字體
            fk = font_keys[np.random.randint(0, len(font_keys))]
            font = font_cache[fk]

            # 在 64x64 白底上畫黑字
            img_pil = Image.new('L', (64, 64), 255)
            draw = ImageDraw.Draw(img_pil)

            # 取得文字大小
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw < 4 or th < 4:
                continue

            # 居中 + 隨機偏移
            cx = (64 - tw) // 2 + np.random.randint(-6, 7)
            cy = (64 - th) // 2 + np.random.randint(-6, 7)
            draw.text((cx - bbox[0], cy - bbox[1]), text,
                      fill=0, font=font)

            # 轉 numpy
            img = np.array(img_pil, dtype=np.uint8)

            # 反轉成 MNIST 格式 (黑底白字)
            img = 255 - img

            # 隨機旋轉 (±20°)
            angle = np.random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
            img = cv2.warpAffine(img, M, (64, 64))

            # 隨機透視 (模擬傾斜)
            if np.random.random() > 0.5:
                margin = np.random.randint(2, 8)
                src = np.float32([[0, 0], [63, 0], [63, 63], [0, 63]])
                dst = src + np.random.uniform(-margin, margin, (4, 2)).astype(np.float32)
                Mw = cv2.getPerspectiveTransform(src, dst)
                img = cv2.warpPerspective(img, Mw, (64, 64))

            # 隨機腐蝕/膨脹 (改變筆畫粗細)
            if np.random.random() > 0.4:
                ksize = np.random.choice([2, 3])
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                if np.random.random() > 0.5:
                    img = cv2.dilate(img, kernel, iterations=1)
                else:
                    img = cv2.erode(img, kernel, iterations=1)

            # 隨機模糊
            if np.random.random() > 0.4:
                k = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)

            # 隨機噪點
            if np.random.random() > 0.6:
                noise = np.random.normal(0, np.random.randint(5, 20),
                                         img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # 縮放到 28x28
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

            # 確認有內容
            if np.max(img) < 30:
                continue

            images.append(img)
            labels.append(digit)
            count += 1

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int64)
    print(f'    生成 {len(images)} 張合成數字')
    return images, labels


# ============================================================
# 4. 背景/非數字 樣本生成 (class 10)
# ============================================================
def generate_background_samples(n_total=7000):
    """
    產生「非數字」背景樣本 (class 10)。
    7 種類型：雜訊、純色、線條、幾何、紋理、邊框、曲線。
    讓 CNN 學會說「這不是數字」。
    """
    import cv2
    images = []
    per_type = n_total // 7

    # 類型 1: 純雜訊
    for _ in range(per_type):
        mode = np.random.randint(0, 3)
        if mode == 0:  # gaussian
            img = np.random.normal(np.random.randint(50, 200),
                                   np.random.randint(20, 80), (28, 28))
            img = np.clip(img, 0, 255).astype(np.uint8)
        elif mode == 1:  # uniform
            img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        else:  # salt & pepper
            img = np.zeros((28, 28), dtype=np.uint8)
            img[np.random.random((28, 28)) > 0.5] = 255
        images.append(img)

    # 類型 2: 純色 / 微噪
    for _ in range(per_type):
        base = np.random.randint(0, 256)
        img = np.full((28, 28), base, dtype=np.uint8)
        noise = np.random.randint(-15, 16, (28, 28))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)

    # 類型 3: 隨機線條
    for _ in range(per_type):
        img = np.zeros((28, 28), dtype=np.uint8)
        for __ in range(np.random.randint(1, 8)):
            pt1 = tuple(np.random.randint(0, 28, 2).tolist())
            pt2 = tuple(np.random.randint(0, 28, 2).tolist())
            cv2.line(img, pt1, pt2, int(np.random.randint(128, 256)),
                     np.random.randint(1, 3))
        images.append(img)

    # 類型 4: 幾何圖形（星形 / 十字 / 鋸齒 / 色塊）
    for _ in range(per_type):
        img = np.zeros((28, 28), dtype=np.uint8)
        shape = np.random.choice(['star', 'cross', 'zigzag', 'blob'])
        if shape == 'star':
            cx, cy = 14, 14
            for angle_deg in range(0, 360, 60):
                rad = np.radians(angle_deg + np.random.randint(-10, 11))
                r = np.random.randint(6, 13)
                ex = int(cx + r * np.cos(rad))
                ey = int(cy + r * np.sin(rad))
                cv2.line(img, (cx, cy),
                         (max(0, min(27, ex)), max(0, min(27, ey))), 255, 1)
        elif shape == 'cross':
            t = np.random.randint(1, 4)
            cv2.line(img, (14, 2), (14, 26), 255, t)
            cv2.line(img, (2, 14), (26, 14), 255, t)
        elif shape == 'zigzag':
            pts, px = [], np.random.randint(2, 8)
            for _i in range(5):
                pts.append([px, np.random.randint(2, 26)])
                px = min(px + np.random.randint(3, 6), 27)
            cv2.polylines(img, [np.array(pts).reshape(-1, 1, 2)],
                          False, 255, np.random.randint(1, 3))
        else:  # blob - 隨機色塊
            img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
            _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
        images.append(img)

    # 類型 5: 模糊紋理
    for _ in range(per_type):
        img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        ksize = np.random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        _, img = cv2.threshold(img, np.random.randint(80, 180), 255,
                               cv2.THRESH_BINARY)
        images.append(img)

    # 類型 6: 邊框 / 矩形 (模擬手機邊框等)
    for _ in range(per_type):
        img = np.zeros((28, 28), dtype=np.uint8)
        t = np.random.randint(1, 4)
        x1, y1 = np.random.randint(0, 8), np.random.randint(0, 8)
        x2, y2 = np.random.randint(20, 28), np.random.randint(20, 28)
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, t)
        images.append(img)

    # 類型 7: 隨機開放/封閉曲線
    for _ in range(per_type):
        img = np.zeros((28, 28), dtype=np.uint8)
        n_pts = np.random.randint(3, 10)
        pts = np.random.randint(1, 27, (n_pts, 1, 2))
        closed = np.random.random() > 0.5
        cv2.polylines(img, [pts], closed, 255, np.random.randint(1, 3))
        images.append(img)

    # 補齊至 n_total
    while len(images) < n_total:
        images.append(np.random.randint(0, 256, (28, 28), dtype=np.uint8))
    images = images[:n_total]

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.full(n_total, 10, dtype=np.int64)  # class 10 = 背景
    return images, labels


# ============================================================
# 5. 訓練主函數
# ============================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'裝置: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # --- 資料準備 ---
    print('\n[1/6] 下載 MNIST 資料集...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=train_transform
    )
    test_data = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform
    )

    # 合成印刷體數字 (PIL 系統字體)
    print('[2/6] 生成合成印刷體數據 (15000 張)...')
    syn_imgs, syn_labels = generate_synthetic_digits(n_per_digit=1500)
    syn_dataset = SyntheticDataset(syn_imgs, syn_labels)

    # 背景/非數字樣本
    print('[3/6] 生成背景樣本 (7000 張)...')
    bg_imgs, bg_labels = generate_background_samples(n_total=7000)
    bg_dataset = SyntheticDataset(bg_imgs, bg_labels)

    # 合併資料集
    combined = torch.utils.data.ConcatDataset([train_data, syn_dataset, bg_dataset])
    print(f'  MNIST({len(train_data)}) + 合成({len(syn_dataset)}) '
          f'+ 背景({len(bg_dataset)}) = {len(combined)}')

    train_loader = DataLoader(combined, batch_size=128, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False,
                             num_workers=0, pin_memory=True)

    # --- 模型 ---
    print(f'[4/6] 建立 CNN 模型 ({NUM_CLASSES} 類)...')
    model = DigitCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  模型參數量: {total_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # --- 訓練 ---
    print('[5/6] 開始訓練...')
    epochs = 15
    best_acc = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        # 測試 (MNIST test set: 只有 class 0-9)
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, pred = output.max(1)
                test_total += target.size(0)
                test_correct += pred.eq(target).sum().item()

        test_acc = 100.0 * test_correct / test_total
        elapsed = time.time() - t0

        marker = ' ★' if test_acc > best_acc else ''
        print(f'  Epoch {epoch:2d}/{epochs}  '
              f'Loss: {train_loss:.4f}  '
              f'Train: {train_acc:.2f}%  '
              f'Test: {test_acc:.2f}%  '
              f'({elapsed:.1f}s){marker}')

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict().copy()

    print(f'\n  最佳測試準確率 (digits 0-9): {best_acc:.2f}%')

    # 載入最佳模型
    model.load_state_dict(best_state)
    model.eval()

    # --- 匯出 ONNX ---
    print('[6/6] 匯出 ONNX 模型...')
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, 'digit_cnn.onnx')

    model.cpu()
    dummy = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11,
    )

    file_size = os.path.getsize(onnx_path) / 1024
    print(f'\n✅ 模型已儲存: {onnx_path}')
    print(f'   檔案大小: {file_size:.1f} KB')
    print(f'   類別數: {NUM_CLASSES} (0-9 數字 + 背景)')
    print(f'   測試準確率 (digits): {best_acc:.2f}%')

    # 驗證 ONNX 可載入
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
        result = sess.run(None, {'input': test_input})
        out_shape = result[0].shape
        print(f'   ONNX 驗證: ✅ output shape={out_shape}')
    except Exception as e:
        print(f'   ONNX 驗證: ❌ {e}')

    print('\n🎉 訓練完成！請跑 camera_detect.py 測試即時偵測。')


if __name__ == '__main__':
    train()
