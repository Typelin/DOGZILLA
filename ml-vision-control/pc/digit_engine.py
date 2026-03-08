"""
digit_engine.py v7 - 白色四邊形偵測 + 透視校正 + CNN 辨識
==========================================================
核心邏輯 (使用者定義的特徵):
    手機螢幕 = 畫面中一塊白色四邊形區域 (可傾斜 → 梯形/平行四邊形)
    數字 = 四邊形內的黑色筆畫

流程：
    1. 高門檻二值化 → 找輪廓
    2. approxPolyDP 取四邊形 → 角度/亮度/均勻度驗證 → 確定是螢幕
    3. 透視校正拉回正方形 → Otsu → 找數字輪廓
    4. 數字送 CNN 辨識
    5. 輪廓逆透視回原圖 → 畫點+線

★ 畫面中沒有「白色四邊形 + 內有黑色特徵」→ 什麼都不輸出

接口 (與 camera_detect.py 完全相容):
    engine = DigitEngine()
    results, thresh, all_info = engine.detect_digits_verbose(frame, min_confidence)
    display = engine.draw_results(frame, results)

依賴: opencv-python, numpy, onnxruntime
"""

import cv2
import numpy as np
import os


class DigitEngine:
    IMG_SIZE = 28
    MEAN = 0.1307
    STD = 0.3081
    BG_CLASS = 10   # 背景類別 (11 類模型)

    def __init__(self, model_path=None):
        self.session = None
        self.trained = False
        self.input_name = None
        self.n_classes = 0
        self._contour_map = {}          # bbox → contour (給 draw_results 用)
        self._screen_quads = []         # 偵測到的螢幕四邊形 (給 draw_results 畫邊框)

        # 預渲染數字模板輪廓
        self.digit_templates = self._generate_templates()
        n_tmpls = sum(len(v) for v in self.digit_templates.values())
        print(f'[DigitEngine] 已產生 {n_tmpls} 個數字形狀模板 (0-9)')

        # 載入 ONNX 模型
        if model_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base, '..', 'models', 'digit_cnn.onnx')
            if not os.path.exists(model_path):
                model_path = os.path.join(base, 'models', 'digit_cnn.onnx')

        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f'[DigitEngine] 找不到 ONNX 模型: {model_path}')
            print(f'[DigitEngine] 請先執行: python train_cnn.py')

    # ==========================================================
    #  ONNX 模型載入
    # ==========================================================
    def load_model(self, path):
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            out_shape = self.session.get_outputs()[0].shape
            self.n_classes = out_shape[-1] if out_shape[-1] is not None else 11
            self.trained = True
            print(f'[DigitEngine] ONNX: {self.n_classes} 類, '
                  f'後端: {self.session.get_providers()[0]}')
        except Exception as e:
            print(f'[DigitEngine] 載入失敗: {e}')

    # ==========================================================
    #  預渲染 0-9 模板輪廓
    # ==========================================================
    def _generate_templates(self):
        """
        用 OpenCV 多種字型渲染 0-9，提取輪廓。
        回傳 {digit: [contour, contour, ...]} 供 matchShapes 比對。
        """
        templates = {}
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_PLAIN,
        ]
        for digit in range(10):
            cnt_list = []
            for font in fonts:
                for thickness in [2, 3, 4]:
                    img = np.zeros((128, 128), dtype=np.uint8)
                    text = str(digit)
                    scale = 3.0 if font != cv2.FONT_HERSHEY_PLAIN else 5.0
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                    x = (128 - tw) // 2
                    y = (128 + th) // 2
                    cv2.putText(img, text, (x, y), font, scale, 255, thickness)
                    cnts, _ = cv2.findContours(
                        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    if cnts:
                        best = max(cnts, key=cv2.contourArea)
                        if cv2.contourArea(best) > 50:
                            cnt_list.append(best)
            templates[digit] = cnt_list
        return templates

    # ==========================================================
    #  形狀匹配：候選輪廓 vs 所有模板
    # ==========================================================
    def _match_shape(self, contour):
        """
        用 Hu 矩比對候選輪廓與所有數字模板。
        回傳 (最像的數字, 相似度分數)，分數越低越像。
        """
        best_digit = -1
        best_score = float('inf')
        for digit, tmpl_list in self.digit_templates.items():
            for tmpl in tmpl_list:
                try:
                    score = cv2.matchShapes(
                        contour, tmpl, cv2.CONTOURS_MATCH_I1, 0
                    )
                except Exception:
                    continue
                if score < best_score:
                    best_score = score
                    best_digit = digit
        return best_digit, best_score

    # ==========================================================
    #  四點排序 (透視校正用)
    # ==========================================================
    @staticmethod
    def _order_points(pts):
        """排序四個點: [左上, 右上, 右下, 左下]"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # 左上: x+y 最小
        rect[2] = pts[np.argmax(s)]   # 右下: x+y 最大
        d = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(d)]   # 右上: y-x 最小
        rect[3] = pts[np.argmax(d)]   # 左下: y-x 最大
        return rect

    # ==========================================================
    #  候選區域提取: 找白色四邊形 → 透視校正 → 裡面找數字
    # ==========================================================
    def _find_candidates(self, gray):
        """
        白色四邊形偵測邏輯：
          1. 高門檻找亮區域 → approxPolyDP 提取四邊形
          2. 驗證: 凸、角度合理、內部亮且均勻
          3. 透視校正 → Otsu → 找數字 → CNN
          ★ 沒有白色四邊形 → 回傳空
        """
        h, w = gray.shape
        frame_area = h * w
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        all_cands = []
        thresh_debug = np.zeros_like(gray)

        # ===== 第一步: 找白色四邊形 =====
        screens = []
        for thresh_val in [220, 200, 185]:
            _, binary = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

            # 閉運算填數字筆畫的洞 (小 kernel，避免牆+螢幕黏合)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

            contours, _ = cv2.findContours(
                filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # 螢幕: 畫面的 2%~30% (手持手機不會超過 30%)
                if area < frame_area * 0.02 or area > frame_area * 0.30:
                    continue

                # ★ 核心: approxPolyDP 嘗試找四邊形 ★
                peri = cv2.arcLength(cnt, True)
                quad = None
                for eps in [0.03, 0.05, 0.07, 0.10]:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4:
                        quad = approx
                        break

                if quad is None:
                    continue

                # 凸性
                if not cv2.isContourConvex(quad):
                    continue

                # ★ 寬高比: 螢幕顯示的白色區域接近正方形 (0.4~2.5) ★
                sx, sy, sw, sh = cv2.boundingRect(quad)
                bbox_aspect = sw / sh if sh > 0 else 99
                if bbox_aspect < 0.3 or bbox_aspect > 3.0:
                    continue

                # ==========================================================
                # ★★ 矩形相似度檢查 ★★
                # 背景幾乎不可能同時是白色+四邊形+像矩形
                # ==========================================================
                pts = quad.reshape(4, 2).astype(np.float64)
                ordered = self._order_points(pts.astype(np.float32))

                # (1) 對邊長度比: 上/下、左/右 各自的短邊/長邊 > 0.5
                side_top = np.linalg.norm(ordered[1] - ordered[0])
                side_bot = np.linalg.norm(ordered[2] - ordered[3])
                side_left = np.linalg.norm(ordered[3] - ordered[0])
                side_right = np.linalg.norm(ordered[2] - ordered[1])

                if max(side_top, side_bot) < 1 or max(side_left, side_right) < 1:
                    continue
                ratio_tb = min(side_top, side_bot) / max(side_top, side_bot)
                ratio_lr = min(side_left, side_right) / max(side_left, side_right)
                if ratio_tb < 0.4 or ratio_lr < 0.4:
                    continue  # 對邊差太多 → 不是矩形

                # (2) 四邊形面積 vs 最小外接旋轉矩形面積 > 0.80
                min_rect = cv2.minAreaRect(quad)
                min_rect_area = min_rect[1][0] * min_rect[1][1]
                if min_rect_area > 0:
                    rect_similarity = area / min_rect_area
                    if rect_similarity < 0.80:
                        continue  # 形狀跟矩形差太多

                # (3) 角度檢查: 每個角 65°~115°
                angles_ok = True
                for i in range(4):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % 4]
                    p3 = pts[(i + 2) % 4]
                    v1 = p1 - p2
                    v2 = p3 - p2
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 < 1 or norm2 < 1:
                        angles_ok = False
                        break
                    cos_a = np.dot(v1, v2) / (norm1 * norm2)
                    angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                    if angle < 65 or angle > 115:
                        angles_ok = False
                        break
                if not angles_ok:
                    continue

                # 內部亮度: 用凸多邊形遮罩取像素
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask, quad, 255)
                roi_pixels = gray[mask > 0]
                if len(roi_pixels) < 100:
                    continue
                mean_val = float(np.mean(roi_pixels))
                std_val = float(np.std(roi_pixels))

                # ★ 嚴格: 手機白底均值>200, 標準差<55 ★
                if mean_val < 195:
                    continue
                if std_val > 55:
                    continue

                screens.append({
                    'quad': pts.astype(np.float32),
                    'area': area,
                    'mean': mean_val,
                    'rect_sim': rect_similarity if min_rect_area > 0 else 0,
                })

            if screens:
                break

        if not screens:
            self._screen_quads = []
            return [], thresh_debug

        # 儲存螢幕四邊形 (供 draw_results 畫邊框)
        self._screen_quads = [scr['quad'].reshape(4, 2).astype(np.int32)
                              for scr in screens]

        # ===== 第二步: 透視校正 + 螢幕內找數字 =====
        for scr in screens:
            pts = scr['quad']
            ordered = self._order_points(pts)

            # 計算校正後的目標尺寸
            w_top = np.linalg.norm(ordered[1] - ordered[0])
            w_bot = np.linalg.norm(ordered[2] - ordered[3])
            h_left = np.linalg.norm(ordered[3] - ordered[0])
            h_right = np.linalg.norm(ordered[2] - ordered[1])
            dst_w = int(max(w_top, w_bot))
            dst_h = int(max(h_left, h_right))
            if dst_w < 30 or dst_h < 30:
                continue

            dst_pts = np.array([
                [0, 0], [dst_w - 1, 0],
                [dst_w - 1, dst_h - 1], [0, dst_h - 1]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(ordered, dst_pts)
            warped = cv2.warpPerspective(gray, M, (dst_w, dst_h))

            # 內縮避免邊框干擾
            margin = max(4, min(dst_w, dst_h) // 12)
            crop = warped[margin:dst_h - margin, margin:dst_w - margin]
            if crop.size == 0:
                continue

            # Otsu 二值化 (白底黑字 → INV 讓數字變白)
            _, crop_bin = cv2.threshold(
                crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # debug 圖: 在原圖四邊形區域顯示 threshold 結果
            sx, sy, sw, sh = cv2.boundingRect(ordered.astype(np.int32))
            sy1, sy2 = max(0, sy), min(h, sy + sh)
            sx1, sx2 = max(0, sx), min(w, sx + sw)
            if sy2 > sy1 and sx2 > sx1:
                dbg_roi = gray[sy1:sy2, sx1:sx2]
                _, dbg_bin = cv2.threshold(
                    dbg_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                thresh_debug[sy1:sy2, sx1:sx2] = dbg_bin

            # 開運算去小雜點
            k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            crop_bin = cv2.morphologyEx(crop_bin, cv2.MORPH_OPEN, k_small)

            # 找數字輪廓
            digit_cnts, _ = cv2.findContours(
                crop_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            crop_area = crop.shape[0] * crop.shape[1]

            # 逆透視矩陣 (warped → 原圖)
            M_inv = cv2.getPerspectiveTransform(dst_pts, ordered)

            # ★ 每個螢幕只保留最大的一個前景輪廓 = 數字 ★
            # (螢幕上一次只顯示一個數字，邊框/雜訊一定比數字小)
            best_cand = None
            best_area = 0

            for cnt in digit_cnts:
                area = cv2.contourArea(cnt)
                if area < crop_area * 0.01 or area > crop_area * 0.70:
                    continue

                lx, ly, lw, lh = cv2.boundingRect(cnt)
                aspect = lw / lh if lh > 0 else 99
                if aspect < 0.04 or aspect > 1.6:
                    continue

                # ROI → 28x28
                roi = crop_bin[ly:ly + lh, lx:lx + lw]
                roi_28 = self._normalize_to_28x28(roi)
                if roi_28 is None:
                    continue

                # 輪廓逆透視回原圖座標
                cnt_in_warped = cnt.reshape(-1, 2).astype(np.float32)
                cnt_in_warped[:, 0] += margin  # crop offset
                cnt_in_warped[:, 1] += margin
                cnt_orig = cv2.perspectiveTransform(
                    cnt_in_warped.reshape(1, -1, 2), M_inv
                ).reshape(-1, 1, 2).astype(np.int32)

                gx, gy, gw, gh = cv2.boundingRect(cnt_orig)

                # 形狀匹配僅作參考 (不過濾)
                match_digit, match_score = self._match_shape(cnt)

                cand = {
                    'bbox': (gx, gy, gw, gh),
                    'contour': cnt_orig,
                    'roi': roi_28,
                    'area': area,
                    'aspect': aspect,
                    'extent': area / (lw * lh) if lw * lh > 0 else 0,
                    'match_digit': match_digit,
                    'match_score': match_score,
                }

                if area > best_area:
                    best_area = area
                    best_cand = cand

            if best_cand is not None:
                all_cands.append(best_cand)

        result = self._nms(all_cands, iou_thresh=0.4)
        return result, thresh_debug

    # ==========================================================
    #  ROI → 28x28 MNIST 正規化
    # ==========================================================
    def _normalize_to_28x28(self, roi_bin):
        coords = cv2.findNonZero(roi_bin)
        if coords is None:
            return None

        nx, ny, nw, nh = cv2.boundingRect(coords)
        crop = roi_bin[ny:ny + nh, nx:nx + nw]
        if crop.size == 0 or nw < 3 or nh < 3:
            return None

        # 等比縮放到 20x20，居中放入 28x28 (MNIST 慣例)
        scale = min(20.0 / max(nw, 1), 20.0 / max(nh, 1))
        new_w = max(1, int(nw * scale))
        new_h = max(1, int(nh * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        tx = (28 - new_w) // 2
        ty = (28 - new_h) // 2
        canvas[ty:ty + new_h, tx:tx + new_w] = resized

        img = canvas.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        return img.reshape(1, 1, 28, 28)

    # ==========================================================
    #  NMS 去重疊
    # ==========================================================
    def _nms(self, candidates, iou_thresh=0.4):
        if len(candidates) <= 1:
            return candidates

        boxes = np.array([c['bbox'] for c in candidates], dtype=np.float32)
        areas = np.array([c['area'] for c in candidates], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        # 優先保留面積大的 (螢幕內最大的前景最可能是數字)
        scores = np.array([c['area'] for c in candidates], dtype=np.float32)
        order = scores.argsort()[::-1]  # descending: largest first

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-6)

            mask = iou <= iou_thresh
            order = order[1:][mask]

        return [candidates[i] for i in keep]

    # ==========================================================
    #  CNN 推論
    # ==========================================================
    def predict(self, tensor):
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        return pred, conf, probs

    # ==========================================================
    #  主偵測流程
    # ==========================================================
    def detect_digits_verbose(self, frame, min_confidence=0.5):
        """
        回傳格式 (與 camera_detect.py 完全相容):
          results:  list of ((x,y,w,h), digit, conf)
          thresh:   debug 二值化圖
          all_info: list of dict
        """
        if not self.trained:
            return [], None, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates, thresh_debug = self._find_candidates(gray)

        # 不清空 contour_map，讓舊幀的輪廓也能被 draw_results 使用
        # 限制大小避免無限增長
        if len(self._contour_map) > 50:
            self._contour_map = {}
        results = []
        all_info = []

        for cand in candidates:
            x, y, w, h = cand['bbox']
            pred, conf, probs = self.predict(cand['roi'])

            is_bg = (self.n_classes >= 11 and pred >= self.BG_CLASS)
            digit = pred if not is_bg else -1

            info = {
                'bbox': (x, y, w, h),
                'area': cand['area'],
                'aspect': cand['aspect'],
                'solidity': cand['extent'],
                'hull_ratio': cand['extent'],
                'filter_reason': 'background' if is_bg else None,
                'digit': digit,
                'confidence': conf,
                'below_threshold': (conf < min_confidence) or is_bg,
                'shape_fail': False,
                'neighbours': [digit] if not is_bg else [],
                'match_digit': cand['match_digit'],
                'match_score': cand['match_score'],
                'top3': sorted(enumerate(probs), key=lambda p: -p[1])[:3],
            }
            all_info.append(info)

            if not is_bg and conf >= min_confidence:
                results.append(((x, y, w, h), digit, conf))
                self._contour_map[(x, y, w, h)] = cand['contour']

        return results, thresh_debug, all_info

    # ==========================================================
    #  預處理 (debug 視窗用)
    # ==========================================================
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 跟偵測邏輯一致: 高門檻切出螢幕
        _, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(binary)
        return gray, thresh

    # ==========================================================
    #  繪製結果: 用點 + 線描出輪廓 (不再畫矩形框)
    # ==========================================================
    def draw_results(self, frame, results):
        disp = frame.copy()

        # ★ 先畫螢幕邊框 (青色) ★
        for quad in self._screen_quads:
            cv2.polylines(disp, [quad], isClosed=True,
                          color=(255, 255, 0), thickness=3)  # 青色
            for pt in quad:
                cv2.circle(disp, tuple(pt.tolist()), 6, (255, 200, 0), -1)
                cv2.circle(disp, tuple(pt.tolist()), 6, (255, 255, 255), 1)

        for i, ((x, y, w, h), d, conf) in enumerate(results):
            if i >= 5:
                break

            # 顏色: 信心越高越綠
            if conf >= 0.8:
                line_color = (0, 255, 0)
                dot_color = (0, 200, 0)
            elif conf >= 0.5:
                line_color = (0, 255, 255)
                dot_color = (0, 200, 200)
            else:
                line_color = (0, 165, 255)
                dot_color = (0, 130, 200)

            contour = self._contour_map.get((x, y, w, h))
            if contour is not None:
                # ★ 用線描出完整輪廓 ★
                cv2.drawContours(disp, [contour], -1, line_color, 2)

                # ★ 用點標出關鍵頂點 (approxPolyDP 簡化後的頂點) ★
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
                for pt in approx:
                    px, py = pt[0].tolist()
                    cv2.circle(disp, (px, py), 5, dot_color, -1)       # 填色圓點
                    cv2.circle(disp, (px, py), 5, (255, 255, 255), 1)  # 白邊
            else:
                # 退路: 穩定緩衝跨幀時沒有輪廓 → 畫虛線框
                for sx in range(x, x + w, 8):
                    cv2.line(disp, (sx, y), (min(sx + 4, x + w), y), line_color, 2)
                    cv2.line(disp, (sx, y + h), (min(sx + 4, x + w), y + h),
                             line_color, 2)
                for sy in range(y, y + h, 8):
                    cv2.line(disp, (x, sy), (x, min(sy + 4, y + h)), line_color, 2)
                    cv2.line(disp, (x + w, sy), (x + w, min(sy + 4, y + h)),
                             line_color, 2)

            # 標籤 (黑底 + 黃字)
            label = f"{d} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(disp, (x, y - th - 14), (x + tw + 4, y - 2),
                          (0, 0, 0), -1)
            cv2.putText(disp, label, (x + 2, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        return disp
