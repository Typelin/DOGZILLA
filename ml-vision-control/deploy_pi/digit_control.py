"""
digit_control.py - Pi 端數字偵測 + 機器狗控制 (All-in-One)
============================================================
部署到 Pi 只需要這個檔案 + digit_cnn.onnx 模型。

功能:
    1. 攝影機偵測白色四邊形 (手機螢幕)
    2. 透視校正 → Otsu → CNN 辨識數字
    3. 投票穩定 → 控制 DOGZILLA 機器狗

數字 → 動作 (參考 test_action 系列測試):
    0=重置歸位  1=前進      2=後退      3=向左轉    4=向右轉
    5=坐下      6=伸懶腰    7=招手      8=握手      9=轉圈
    
    移動參數 (參考 test_action_4/5):
        步頻: normal, 高度: 108, 速度: 80, 移動持續: 2秒, 旋轉持續: 2秒

用法:
    # 複製到 Pi:
    scp digit_control.py digit_cnn.onnx pi@<IP>:~/
    
    # Pi 上執行:
    sudo pkill -f app_dogzilla.py
    python3 digit_control.py
    
    # 選項:
    python3 digit_control.py --no-dog          # 純偵測 (不接狗)
    python3 digit_control.py --headless        # 無畫面
    python3 digit_control.py --votes 6         # 6 幀確認
    python3 digit_control.py --model path.onnx # 指定模型

需要: opencv-python, numpy, onnxruntime
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse


# ==============================================================
#  數字辨識引擎
# ==============================================================
class DigitEngine:
    IMG_SIZE = 28
    MEAN = 0.1307
    STD = 0.3081
    BG_CLASS = 10

    def __init__(self, model_path='digit_cnn.onnx'):
        self.session = None
        self.trained = False
        self.input_name = None
        self.n_classes = 0
        self._screen_quads = []

        # 找模型
        if not os.path.exists(model_path):
            # 嘗試同目錄
            base = os.path.dirname(os.path.abspath(__file__))
            alt = os.path.join(base, 'digit_cnn.onnx')
            if os.path.exists(alt):
                model_path = alt

        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f'[Engine] 找不到模型: {model_path}')

    def _load_model(self, path):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                path, providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            out_shape = self.session.get_outputs()[0].shape
            self.n_classes = out_shape[-1] if out_shape[-1] is not None else 11
            self.trained = True
            print(f'[Engine] 模型載入: {self.n_classes} 類')
        except Exception as e:
            print(f'[Engine] 載入失敗: {e}')

    # ---------- 四點排序 ----------
    @staticmethod
    def _order_points(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    # ---------- ROI → 28x28 ----------
    def _normalize_to_28x28(self, roi_bin):
        coords = cv2.findNonZero(roi_bin)
        if coords is None:
            return None
        nx, ny, nw, nh = cv2.boundingRect(coords)
        crop = roi_bin[ny:ny + nh, nx:nx + nw]
        if crop.size == 0 or nw < 3 or nh < 3:
            return None
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

    # ---------- CNN 推論 ----------
    def _predict(self, tensor):
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        return pred, conf

    # ---------- 找螢幕 + 辨識數字 ----------
    def detect(self, frame, min_confidence=0.5):
        """
        主偵測流程。
        回傳: list of (digit, confidence) — 通常 0 或 1 個結果
        """
        if not self.trained:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        frame_area = h * w
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ===== 找白色四邊形 =====
        screens = []
        for thresh_val in [220, 200, 185]:
            _, binary = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
            contours, _ = cv2.findContours(
                filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < frame_area * 0.02 or area > frame_area * 0.30:
                    continue

                peri = cv2.arcLength(cnt, True)
                quad = None
                for eps in [0.03, 0.05, 0.07, 0.10]:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)
                    if len(approx) == 4:
                        quad = approx
                        break
                if quad is None:
                    continue
                if not cv2.isContourConvex(quad):
                    continue

                # 寬高比
                sx, sy, sw, sh = cv2.boundingRect(quad)
                if sw / max(sh, 1) < 0.3 or sw / max(sh, 1) > 3.0:
                    continue

                pts = quad.reshape(4, 2).astype(np.float64)
                ordered = self._order_points(pts.astype(np.float32))

                # 對邊比
                side_top = np.linalg.norm(ordered[1] - ordered[0])
                side_bot = np.linalg.norm(ordered[2] - ordered[3])
                side_left = np.linalg.norm(ordered[3] - ordered[0])
                side_right = np.linalg.norm(ordered[2] - ordered[1])
                if max(side_top, side_bot) < 1 or max(side_left, side_right) < 1:
                    continue
                if min(side_top, side_bot) / max(side_top, side_bot) < 0.4:
                    continue
                if min(side_left, side_right) / max(side_left, side_right) < 0.4:
                    continue

                # 矩形相似度
                min_rect = cv2.minAreaRect(quad)
                min_rect_area = min_rect[1][0] * min_rect[1][1]
                if min_rect_area > 0 and area / min_rect_area < 0.80:
                    continue

                # 角度 65°~115°
                angles_ok = True
                for i in range(4):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % 4]
                    p3 = pts[(i + 2) % 4]
                    v1, v2 = p1 - p2, p3 - p2
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 < 1 or n2 < 1:
                        angles_ok = False; break
                    ang = np.degrees(np.arccos(
                        np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
                    if ang < 65 or ang > 115:
                        angles_ok = False; break
                if not angles_ok:
                    continue

                # 內部亮度
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillConvexPoly(mask, quad, 255)
                roi_pixels = gray[mask > 0]
                if len(roi_pixels) < 100:
                    continue
                if float(np.mean(roi_pixels)) < 195:
                    continue
                if float(np.std(roi_pixels)) > 55:
                    continue

                screens.append(pts.astype(np.float32))
            if screens:
                break

        if not screens:
            self._screen_quads = []
            return []

        self._screen_quads = [s.reshape(4, 2).astype(np.int32) for s in screens]

        # ===== 透視校正 + 找數字 =====
        results = []
        for pts in screens:
            ordered = self._order_points(pts)
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

            margin = max(4, min(dst_w, dst_h) // 12)
            crop = warped[margin:dst_h - margin, margin:dst_w - margin]
            if crop.size == 0:
                continue

            _, crop_bin = cv2.threshold(
                crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            crop_bin = cv2.morphologyEx(crop_bin, cv2.MORPH_OPEN, k_small)

            digit_cnts, _ = cv2.findContours(
                crop_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            crop_area = crop.shape[0] * crop.shape[1]

            # 只取最大前景
            best_roi = None
            best_area = 0
            for cnt in digit_cnts:
                a = cv2.contourArea(cnt)
                if a < crop_area * 0.01 or a > crop_area * 0.70:
                    continue
                lx, ly, lw, lh = cv2.boundingRect(cnt)
                asp = lw / lh if lh > 0 else 99
                if asp < 0.04 or asp > 1.6:
                    continue
                if a > best_area:
                    best_area = a
                    best_roi = crop_bin[ly:ly + lh, lx:lx + lw]

            if best_roi is None:
                continue

            roi_28 = self._normalize_to_28x28(best_roi)
            if roi_28 is None:
                continue

            pred, conf = self._predict(roi_28)
            is_bg = (self.n_classes >= 11 and pred >= self.BG_CLASS)
            if not is_bg and conf >= min_confidence:
                results.append((pred, conf))

        return results

    # ---------- 簡易繪製 ----------
    def draw(self, frame, results):
        disp = frame.copy()
        # 螢幕邊框 (青色)
        for quad in self._screen_quads:
            cv2.polylines(disp, [quad], True, (255, 255, 0), 2)
            for pt in quad:
                cv2.circle(disp, tuple(pt.tolist()), 4, (255, 200, 0), -1)

        # 數字標籤
        if results:
            digit, conf = results[0]
            label = f"{digit} ({conf:.0%})"
            cv2.putText(disp, label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        return disp


# ==============================================================
#  機器狗控制
# ==============================================================
class DogController:
    """
    數字 → 動作對應 (參考 test_action 系列):
        0: 重置歸位   action(255)          — 所有測試結尾的安全歸位
        1: 前進       forward(80) 2秒      — test_action_4 速度 80
        2: 後退       move(-80,0,0) 2秒    — 反向移動
        3: 向左轉     turnleft(80) 2秒     — test_action_5 轉速 80
        4: 向右轉     turnright(80) 2秒    — test_action_5 轉速 80
        5: 坐下       action(12) 3秒       — test_action_3 坐下
        6: 伸懶腰     action(14) 4秒       — test_action_1 伸懶腰
        7: 招手       action(13) 4秒       — test_action_2 招手
        8: 握手       action(19) 4秒       — test_action_2 握手
        9: 轉圈       action(4)  5秒       — 動作控制.MD 轉圈
    """
    # 移動參數 (來自 test_action_4.py)
    MOVE_SPEED = 80        # 移動速度
    MOVE_DURATION = 2.0    # 前進/後退持續秒數
    TURN_SPEED = 80        # 旋轉速度 (來自 test_action_5.py)
    TURN_DURATION = 2.0    # 旋轉持續秒數
    STD_HEIGHT = 108       # 標準站姿高度 (test_action_4: 108)

    ACTION_NAMES = {
        0: "重置歸位", 1: "前進", 2: "後退", 3: "向左轉", 4: "向右轉",
        5: "坐下", 6: "伸懶腰", 7: "招手", 8: "握手", 9: "轉圈",
    }

    def __init__(self):
        self.dog = None
        self.connected = False
        try:
            sys.path.insert(0, "/home/pi/DOGZILLA/app_dogzilla")
            from DOGZILLALib import DOGZILLA
            self.dog = DOGZILLA()
            self.connected = True
            # 初始化: 標準步頻 + 標準高度 (參考 test_action_4/5)
            self.dog.pace("normal")
            self.dog.translation('z', self.STD_HEIGHT)
            time.sleep(1)
            print("[Dog] 已連接 (步頻=normal, 高度=108)")
        except Exception as e:
            print(f"[Dog] 連接失敗: {e} (模擬模式)")

    def execute(self, digit):
        if digit not in self.ACTION_NAMES:
            return
        name = self.ACTION_NAMES[digit]
        print(f"[動作] {digit} → {name}")
        if not self.connected:
            return
        try:
            if digit == 0:
                # 重置歸位 — 恢復初始姿態
                self.dog.action(255)
                time.sleep(1.0)

            elif digit == 1:
                # 前進 — speed 80, 2 秒 (參考 test_action_4)
                self.dog.forward(self.MOVE_SPEED)
                time.sleep(self.MOVE_DURATION)
                self.dog.stop()

            elif digit == 2:
                # 後退 — speed 80, 2 秒 (參考 app_dogzilla.py: g_dog.back(step))
                self.dog.back(self.MOVE_SPEED)
                time.sleep(self.MOVE_DURATION)
                self.dog.stop()

            elif digit == 3:
                # 向左轉 — 轉速 80, 2 秒 (參考 test_action_5)
                self.dog.turnleft(self.TURN_SPEED)
                time.sleep(self.TURN_DURATION)
                self.dog.stop()

            elif digit == 4:
                # 向右轉 — 轉速 80, 2 秒 (參考 test_action_5)
                self.dog.turnright(self.TURN_SPEED)
                time.sleep(self.TURN_DURATION)
                self.dog.stop()

            elif digit == 5:
                # 坐下 — action(12), 等 3 秒 (參考 test_action_3)
                self.dog.action(12)
                time.sleep(3.0)

            elif digit == 6:
                # 伸懶腰 — action(14), 等 4 秒 (參考 test_action_1)
                self.dog.action(14)
                time.sleep(4.0)

            elif digit == 7:
                # 招手 — action(13), 等 4 秒 (參考 test_action_2)
                self.dog.action(13)
                time.sleep(4.0)

            elif digit == 8:
                # 握手 — action(19), 等 4 秒 (參考 test_action_2)
                self.dog.action(19)
                time.sleep(4.0)

            elif digit == 9:
                # 轉圈 — action(4), 等 5 秒
                self.dog.action(4)
                time.sleep(5.0)

        except Exception as e:
            print(f"  動作失敗: {e}")

    def reset(self):
        self.execute(0)


# ==============================================================
#  投票穩定 (連續幀防彈跳)
# ==============================================================
class VoteBuffer:
    """
    防彈跳邏輯:
    - 必須連續 N 幀偵測到「同一個數字」才觸發動作
    - 中間如果斷掉 (沒偵測到 / 偵測到不同數字) → 計數歸零
    - 觸發後進入冷卻期，冷卻期間不接受任何輸入
    - 同一數字不能連續觸發 (必須先看到別的或空)
    """
    def __init__(self, required=15, cooldown=5.0):
        self.required = required      # 需要連續幾幀
        self.cooldown = cooldown      # 觸發後冷卻秒數
        self.streak_digit = -1        # 目前連續的數字
        self.streak_count = 0         # 連續計數
        self.last_trigger_time = 0    # 上次觸發時間
        self.last_trigger_digit = -1  # 上次觸發的數字
        self.cleared = False          # 是否已看到空幀 (防重複觸發)

    def add(self, digit):
        """偵測到數字時呼叫。回傳觸發的數字 or None"""
        now = time.time()

        # 冷卻期中 → 忽略
        if now - self.last_trigger_time < self.cooldown:
            return None

        # 跟上一幀同數字 → 累加
        if digit == self.streak_digit:
            self.streak_count += 1
        else:
            # 不同數字 → 重新開始計數
            self.streak_digit = digit
            self.streak_count = 1

        # 達到門檻
        if self.streak_count >= self.required:
            # 防止同數字連續觸發: 必須中間有「空」或「不同數字」
            if digit == self.last_trigger_digit and not self.cleared:
                return None
            self.last_trigger_time = now
            self.last_trigger_digit = digit
            self.streak_count = 0
            self.streak_digit = -1
            self.cleared = False
            return digit

        return None

    def no_detection(self):
        """該幀沒偵測到任何數字時呼叫"""
        self.streak_digit = -1
        self.streak_count = 0
        self.cleared = True  # 看到空幀，允許同數字再次觸發

    def clear(self):
        self.streak_digit = -1
        self.streak_count = 0
        self.last_trigger_digit = -1
        self.cleared = True


# ==============================================================
#  主程式
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="DOGZILLA 數字控制")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--model", type=str, default="digit_cnn.onnx")
    parser.add_argument("--votes", type=int, default=15,
                        help="連續幾幀偵測到同數字才觸發 (預設15)")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="觸發後冷卻秒數 (預設5秒)")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--no-dog", action="store_true", help="不接機器狗")
    parser.add_argument("--headless", action="store_true", help="無畫面")
    args = parser.parse_args()

    print("=" * 40)
    print("  DOGZILLA 數字控制系統")
    print("=" * 40)

    engine = DigitEngine(args.model)
    if not engine.trained:
        print("\n模型未載入！請確認 digit_cnn.onnx 在同目錄")
        return

    controller = None
    if not args.no_dog:
        controller = DogController()

    vote = VoteBuffer(required=args.votes, cooldown=args.cooldown)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("無法開啟攝影機")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"\n攝影機 {args.camera} ({args.width}x{args.height})")
    print("數字: 0=重置 1=前進 2=後退 3=左轉 4=右轉")
    print("      5=坐下 6=伸懶腰 7=招手 8=握手 9=轉圈")
    if not args.headless:
        print("按鍵: q=退出 d=debug r=重置\n")

    show_debug = False
    fps_count = 0
    fps_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            results = engine.detect(frame, min_confidence=args.confidence)

            if results:
                digit, conf = results[0]
                triggered = vote.add(digit)
                if triggered is not None and controller:
                    print(f">>> 確認觸發: {digit} <<<")
                    controller.execute(triggered)
            else:
                vote.no_detection()

            # FPS
            fps_count += 1
            if fps_count % 10 == 0:
                now = time.time()
                fps = 10 / max(now - fps_time, 0.001)
                fps_time = now

            if not args.headless:
                disp = engine.draw(frame, results)
                status = f"FPS:{fps:.0f}"
                if results:
                    status += f" | {results[0][0]}({results[0][1]:.0%})"
                streak_info = f"{vote.streak_digit}" if vote.streak_digit >= 0 else "-"
                status += f" | 連續:{streak_info}x{vote.streak_count}/{vote.required}"
                cv2.putText(disp, status, (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("DOGZILLA", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and controller:
                    controller.reset()
                    vote.clear()
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if controller and controller.connected:
            controller.reset()
        print("結束。")


if __name__ == "__main__":
    main()
