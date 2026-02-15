import time
import sys

# 指向官方庫的路徑
sys.path.append('/home/pi/DOGZILLA/app_dogzilla/')
from DOGZILLALib import DOGZILLA

def main():
    # 初始化機器狗
    bot = DOGZILLA()
    
    # 設定速度 (0-100)
    bot.motor_speed(50)
    
    print("--- 官方文檔動作測試 1 (單一動作) ---")

    try:
        # 動作 1：伸懶腰 (ID: 14)
        # 這是預設的開機動作
        print("執行動作 ID 14：伸懶腰 (Stretch)")
        bot.action(14)
        time.sleep(4)
        
        # 結束測試：恢復初始姿態 (ID: 255)
        print("恢復初始姿態 (ID: 255)...")
        bot.action(255) # 255 代表重置
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