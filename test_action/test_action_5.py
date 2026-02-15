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
    print("--- DOGZILLA 原地旋轉測試 (test_action_5) ---")
    bot = DOGZILLA()
    
    try:
        # --- 初始設定 ---
        print("設定為標準步頻 (Normal) 與標準高度 (108)")
        bot.pace("normal")
        bot.translation('z', 108)
        time.sleep(2)
        
        # --- 測試 A: 向右旋轉 ---
        print("\n階段 A: 向右原地旋轉 (轉速 80, 持續 3 秒)")
        bot.turnright(80)
        time.sleep(3.0)
        bot.stop()
        print("停止旋轉")
        
        time.sleep(1)

        # --- 測試 B: 向左旋轉 ---
        print("\n階段 B: 向左原地旋轉 (轉速 80, 持續 3 秒)")
        bot.turnleft(80)
        time.sleep(3.0)
        bot.stop()
        print("停止旋轉")
        
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
