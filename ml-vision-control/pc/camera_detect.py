"""
camera_detect.py - PC 即時數字偵測
=====================================
開啟攝影機，即時偵測畫面中的數字 (0~9)。

使用方式：
    cd ml-vision-control/pc
    python camera_detect.py

操作鍵：
    q     - 退出
    d     - 切換除錯視窗 (顯示預處理結果)
    +/-   - 調整最小面積閾值
    空白鍵 - 暫停/繼續

需要先執行 train_model.py 訓練模型。
"""

import cv2
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digit_engine import DigitEngine


def create_debug_view(frame, thresh, results):
    """
    建立除錯視窗：原圖 + 二值圖 + 各偵測區域放大。
    """
    h, w = frame.shape[:2]

    # 二值圖轉 BGR
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # 在二值圖上畫輪廓
    for (x, y, bw, bh), digit, conf in results:
        color = (0, 255, 0) if conf >= 0.8 else (0, 255, 255)
        cv2.rectangle(thresh_bgr, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(
            thresh_bgr, f"{digit}", (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # 左右拼接
    combined = np.hstack([frame, thresh_bgr])

    # 底部顯示偵測到的各個 ROI 放大圖
    if results:
        roi_strip_h = 80
        roi_strip = np.zeros((roi_strip_h, combined.shape[1], 3), dtype=np.uint8)
        x_offset = 10

        for (rx, ry, rw, rh), digit, conf in results[:10]:  # 最多顯示 10 個
            roi = thresh[ry:ry + rh, rx:rx + rw]
            if roi.size == 0:
                continue

            # 放大到固定高度
            scale = (roi_strip_h - 20) / max(rh, 1)
            new_w = max(1, int(rw * scale))
            new_h = max(1, int(rh * scale))
            roi_resized = cv2.resize(roi, (new_w, new_h))
            roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)

            # 放入 strip
            if x_offset + new_w + 10 > roi_strip.shape[1]:
                break
            y_start = (roi_strip_h - new_h) // 2
            roi_strip[y_start:y_start + new_h, x_offset:x_offset + new_w] = roi_bgr

            # 標籤
            cv2.putText(
                roi_strip, f"{digit}:{conf:.0%}",
                (x_offset, roi_strip_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )
            x_offset += new_w + 15

        combined = np.vstack([combined, roi_strip])

    return combined


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PC 即時數字偵測")
    parser.add_argument("--camera", type=int, default=0, help="攝影機索引 (預設: 0)")
    parser.add_argument("--width", type=int, default=640, help="影格寬度 (預設: 640)")
    parser.add_argument("--height", type=int, default=480, help="影格高度 (預設: 480)")
    parser.add_argument("--model", type=str, default=None, help="模型路徑")
    parser.add_argument("--min-conf", type=float, default=0.65, help="最低信心閾值 (預設: 0.65)")
    parser.add_argument("--debug", action="store_true", help="啟動時顯示除錯視窗")
    parser.add_argument("--log", action="store_true", help="開啟詳細 log （輸出到終端）")
    parser.add_argument("--log-every", type=int, default=15, help="每 N 幀輸出一次 log (預設: 15)")
    args = parser.parse_args()

    # 載入模型
    print("載入模型...")
    engine = DigitEngine(model_path=args.model)
    if not engine.trained:
        print("\n❌ 模型未載入！請先執行: python train_model.py")
        return

    # 開啟攝影機
    print(f"開啟攝影機 {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    show_debug = args.debug
    show_log = args.log
    log_every = args.log_every
    paused = False
    min_area = 300
    frame_count = 0
    fps_time = time.time()
    fps = 0
    # 穩定幀緩衝：多數決確認最終數字 (只影響狀態列，不影響輪廓繪製)
    stable_buffer = []
    STABLE_FRAMES = 4          # 緩衝長度
    STABLE_MAJORITY = 3        # 4 幀中 3 幀一致就鎖定
    stable_digit = None
    stable_result = None

    print("\n操作說明:")
    print("  q     = 退出")
    print("  d     = 切換除錯視窗")
    print("  l     = 切換 log 輸出")
    print("  +/-   = 調整最小面積")
    print("  空白鍵 = 暫停/繼續")
    print()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ 讀取影格失敗")
                break

        # 偵測
        all_results, thresh, all_info = engine.detect_digits_verbose(frame, min_confidence=args.min_conf)

        # 選最佳候選 (面積 × 信心 加權)
        if all_results:
            best = max(all_results, key=lambda r: (r[0][2] * r[0][3]) * r[2])
        else:
            best = None

        # 穩定緩衝 (多數決：只影響狀態列的「確認數字」)
        if best is not None:
            stable_buffer.append(best[1])
        else:
            stable_buffer.append(None)
        if len(stable_buffer) > STABLE_FRAMES:
            stable_buffer.pop(0)

        # 多數決：STABLE_FRAMES 幀中 STABLE_MAJORITY 幀一致就鎖定
        if len(stable_buffer) >= STABLE_FRAMES:
            from collections import Counter
            counts = Counter(d for d in stable_buffer if d is not None)
            if counts:
                top_digit, top_count = counts.most_common(1)[0]
                if top_count >= STABLE_MAJORITY:
                    stable_digit = top_digit
                    stable_result = best if best and best[1] == top_digit else stable_result
                else:
                    stable_digit = None
                    stable_result = None
            else:
                stable_digit = None
                stable_result = None
        elif best is None and all(d is None for d in stable_buffer):
            stable_digit = None
            stable_result = None

        # ★ 畫面上顯示：當前幀 ALL 候選 (立刻畫輪廓) ★
        results = all_results if all_results else []

        # --- Log 輸出 ---
        if show_log and frame_count % log_every == 0:
            FILTER_TAG = {'area': 'AREA', 'aspect': 'ASPT', 'solidity': 'SLDT'}
            passed   = [i for i in all_info if i['filter_reason'] is None]
            rejected = [i for i in all_info if i['filter_reason'] is not None]
            print(f"\n╔{'='*62}")
            print(f"  Frame {frame_count:5d} | 通過過濾: {len(passed)} | 被過濾掉: {len(rejected)} | FPS: {fps:.0f}")
            print(f"  信心門溻: {args.min_conf:.0%}  第一名: {results[0][1] if results else '--'}")
            print(f"╠{'='*62}")
            if passed:
                print("  [通過形狀匹配的候選區]")
                for i in passed:
                    x,y,w,h = i['bbox']
                    ms = i.get('match_score', -1)
                    md = i.get('match_digit', -1)
                    if i['below_threshold']:
                        verdict = f"✖ CNN={i['confidence']:.0%} < 門檻  shape→{md}({ms:.3f})"
                    else:
                        verdict = f"✔ {i['digit']}  CNN={i['confidence']:.0%}  shape→{md}({ms:.3f})"
                    print(f"    ({x:3d},{y:3d}) {w:3d}x{h:3d}  area={i['area']:6.0f}  asp={i['aspect']:.2f}  {verdict}")
            if rejected:
                print("  [被拒絕]")
                for i in rejected[:8]:
                    x,y,w,h = i['bbox']
                    reason = i.get('filter_reason', '?')
                    ms = i.get('match_score', -1)
                    print(f"    ({x:3d},{y:3d}) {w:3d}x{h:3d}  area={i['area']:6.0f}  ❌ {reason} shape({ms:.3f})")
                if len(rejected) > 8:
                    print(f"    ... 另外 {len(rejected)-8} 項")
            print(f"╚{'='*62}")

        # FPS 計算
        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - fps_time)
            fps_time = now

        # 畫結果
        display = engine.draw_results(frame, results)

        # 狀態列
        if stable_digit is not None:
            d = stable_digit
            conf_str = f"{best[2]:.0%}" if best and best[1] == d else "--"
            digit_str = f">> CONFIRMED: {d} ({conf_str}) <<"
        elif best is not None:
            from collections import Counter
            counts = Counter(dd for dd in stable_buffer if dd is not None)
            top = counts.most_common(1)[0] if counts else (best[1], 0)
            digit_str = f"Detecting {best[1]} [{top[1]}/{STABLE_MAJORITY}]"
        else:
            digit_str = "-- No Digit --"
        status = f"FPS:{fps:.0f}  {digit_str}"
        if paused:
            status += "  [PAUSED]"
        cv2.putText(
            display, status, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

        cv2.imshow("Digit Detection", display)

        # 除錯視窗
        if show_debug and thresh is not None:
            debug_view = create_debug_view(frame, thresh, results)
            cv2.imshow("Debug", debug_view)
        elif not show_debug:
            try:
                cv2.destroyWindow("Debug")
            except Exception:
                pass

        # 鍵盤輸入
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"除錯視窗: {'開' if show_debug else '關'}")
        elif key == ord("l"):
            show_log = not show_log
            print(f"Log 輸出: {'開' if show_log else '關'}")
        elif key == ord("+") or key == ord("="):
            min_area += 100
            print(f"最小面積: {min_area}")
        elif key == ord("-"):
            min_area = max(100, min_area - 100)
            print(f"最小面積: {min_area}")
        elif key == 32:  # 空白鍵
            paused = not paused
            print(f"{'暫停' if paused else '繼續'}")

    cap.release()
    cv2.destroyAllWindows()
    print("結束。")


if __name__ == "__main__":
    main()
