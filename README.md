# DOGZILLA 數字視覺控制系統

透過攝影機辨識手機螢幕上的數字 (0-9)，控制 Yahboom DOGZILLA 機器狗執行對應動作。

## 運作原理

```
手機顯示數字 → 攝影機拍攝 → 偵測白色螢幕 → 透視校正 → CNN 辨識 → 機器狗動作
```

1. **螢幕偵測**：在畫面中找到白色四邊形（手機螢幕），透過矩形相似度驗證排除誤判
2. **透視校正**：將傾斜的螢幕校正為正面視角
3. **數字辨識**：11 類 CNN (0-9 + 背景)，ONNX 格式，測試準確率 99.58%
4. **防彈跳**：連續 15 幀偵測到同一數字才觸發，5 秒冷卻期防止重複

## 數字動作對應

| 數字 | 動作 | 說明 |
|:---:|------|------|
| 0 | 重置歸位 | 恢復初始站姿 |
| 1 | 前進 | 速度 80，持續 2 秒 |
| 2 | 後退 | 速度 80，持續 2 秒 |
| 3 | 向左轉 | 轉速 80，持續 2 秒 |
| 4 | 向右轉 | 轉速 80，持續 2 秒 |
| 5 | 坐下 | 預設動作 ID 12 |
| 6 | 伸懶腰 | 預設動作 ID 14 |
| 7 | 招手 | 預設動作 ID 13 |
| 8 | 握手 | 預設動作 ID 19 |
| 9 | 轉圈 | 預設動作 ID 4 |

## 專案結構

```
DOGZILLA/
├── ml-vision-control/
│   ├── pc/                         # PC 端開發與訓練
│   │   ├── digit_engine.py         # 數字偵測引擎 (v7)
│   │   ├── camera_detect.py        # 即時攝影機偵測 GUI
│   │   └── train_cnn.py            # CNN 訓練腳本
│   ├── deploy_pi/                  # Pi 部署包 (2 個檔案)
│   │   ├── digit_control.py        # All-in-One 控制腳本
│   │   └── digit_cnn.onnx          # CNN 模型 (~1.8MB)
│   ├── models/                     # 訓練產出
│   │   └── digit_cnn.onnx
│   ├── data/                       # MNIST 訓練資料 (未上傳)
│   └── 更新日誌.MD
├── core_files/                     # 官方原始碼參考
├── test_action/                    # 動作測試腳本 (test_action_1~5)
├── 動作控制.MD                     # 預設動作 ID 清單
├── 2_13_步伐控制.md                # 步伐與高度參數
└── 總流程.MD                       # 系統架構說明
```

## 快速開始

### Pi 部署

```bash
# 1. 複製檔案到 Pi
scp ml-vision-control/deploy_pi/digit_control.py \
    ml-vision-control/deploy_pi/digit_cnn.onnx \
    pi@<IP>:~/Desktop/ml-vision-control/

# 2. Pi 上安裝依賴 (首次)
pip install onnxruntime

# 3. 停止官方背景服務，啟動辨識控制
sudo pkill -f app_dogzilla.py
python3 digit_control.py
```

### 啟動選項

```bash
python3 digit_control.py                # 預設啟動
python3 digit_control.py --no-dog       # 純辨識測試 (不接機器狗)
python3 digit_control.py --headless     # 無畫面模式
python3 digit_control.py --votes 20     # 需連續 20 幀才觸發
python3 digit_control.py --cooldown 8   # 觸發後冷卻 8 秒
python3 digit_control.py --confidence 0.7  # 信心度門檻 70%
```

### PC 開發

```bash
cd ml-vision-control/pc

# 即時偵測測試
python camera_detect.py --debug

# 重新訓練 CNN
python train_cnn.py
```

## 技術細節

### 偵測引擎
- 多閾值搜尋 (220/200/185) 找白色區域
- `approxPolyDP` 四邊形擬合 + 凸性檢查
- 矩形驗證：對邊比 >0.4、面積比 >0.80、角度 65°-115°
- 亮度過濾：平均 >195、標準差 <55
- `getPerspectiveTransform` 透視校正
- Otsu 二值化 → 最大前景輪廓 → 正規化至 28×28

### CNN 模型
- 架構：Conv(1→32) → Conv(32→64) → FC(64×7×7→128→11)
- 訓練資料：MNIST 60K + PIL 合成字體 15K + 背景 7K = 82K 張
- 字體：Arial, Calibri, Segoe UI, Verdana, Tahoma, Consolas 等 16 種
- 訓練：15 輪，測試準確率 99.58%

### 硬體環境
- **機器狗**：Yahboom DOGZILLA (DOGZILLALib 3.1.9)
- **Pi**：Raspberry Pi，Python 3.11，onnxruntime (CPU)
- **PC 訓練**：Python 3.12，PyTorch 2.6.0，NVIDIA RTX 4080 Laptop

## 開發進度

- [x] 白色四邊形螢幕偵測
- [x] 透視校正 (傾斜手機可辨識)
- [x] 矩形相似度驗證 (減少誤判)
- [x] 11 類 CNN 訓練 (含背景類別)
- [x] PIL 系統字體合成訓練資料
- [x] PC 即時偵測 GUI
- [x] Pi All-in-One 部署腳本
- [x] 10 種數字動作對應 (移動/轉向/預設動作)
- [x] 連續幀防彈跳 + 冷卻機制
- [x] 更新日誌
- [ ] 數字 7 與 1 的混淆優化
- [ ] 低光源環境適應
- [ ] 多數字序列輸入

## 已知限制

- 數字 7 偶爾被誤判為 1（筆畫相似）
- 需白底黑字清晰顯示，反光或過暗會影響偵測
- 手機螢幕需佔畫面 2%~30% 面積

## 授權

本專案為學習用途，基於 Yahboom DOGZILLA 官方 SDK 開發。
