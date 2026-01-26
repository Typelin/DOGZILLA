import time
import sys

# 指向官方庫的路徑 [cite: 19]
sys.path.append('/home/pi/DOGZILLA/app_dogzilla/')
from DOGZILLALib import DOGZILLA

def main():
    # 初始化機器狗 [cite: 11]
    bot = DOGZILLA()
    
    # 設定速度
    bot.motor_speed(50) 
    
    print("--- 官方文檔動作測試開始 ---")

    try:
        # 動作 1：招手 (ID: 13) 
        print("執行動作 ID 13：招手 (Wave Hand)")
        bot.action(13) 
        time.sleep(4)

        # 動作 2：握手 (ID: 19) 
        print("執行動作 ID 19：握手 (Handshake)")
        bot.action(19) 
        time.sleep(4)

        # 結束測試：恢復初始姿態 (ID: 255) [cite: 12, 14]
        # 文檔標明 255 是 Reset，並非 0
        print("恢復初始姿態 (ID: 255)...")
        bot.action(255)
        time.sleep(2) 

    except Exception as e:
        print(f"錯誤: {e}")

    finally:
        # 確保清理物件
        print("測試結束")
        try:
            del bot
        except:
            pass

if __name__ == "__main__":
    main()