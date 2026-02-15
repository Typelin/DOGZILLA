import time
import sys

# 1. 確保導入樹莓派上的官方驅動庫
sys.path.append('/home/pi/DOGZILLA/app_dogzilla/')
try:
    from DOGZILLALib import DOGZILLA
except ImportError:
    print("錯誤: 找不到 DOGZILLALib。請確保此腳本運行在機器狗的樹莓派上。")
    sys.exit(1)

def main():
    print("--- DOGZILLA 步伐與高度實機測試 (test_action_4) ---")
    bot = DOGZILLA()
    
    try:
        # --- 測試 A: 高度動態調整 ---
        print("階段 A: 調整高度中...")
        for h in [80, 108]:
            print(f"  -> 設定高度: {h}")
            bot.translation('z', h)
            time.sleep(2)
        
        # --- 測試 B: 步頻與模擬距離測試 ---
        print("\n階段 B: 步頻與模擬距離測試...")
        print("  -> 設定步頻為 Normal (標準)")
        bot.pace("normal")
        time.sleep(1)
        
        print("  -> 移動 5 秒 (速度 80)")
        bot.forward(80)
        time.sleep(5.0)
        bot.stop()
        print("  -> 停止移動")
        
        time.sleep(2)

    except Exception as e:
        print(f"測試異常: {e}")
    finally:
        # 最後安全歸位
        print("\n測試結束: 恢復預設姿勢 (ID: 255)")
        bot.action(255)
        time.sleep(2)
        del bot

if __name__ == "__main__":
    main()
