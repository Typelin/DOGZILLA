import time
import sys

# 指向官方庫的路徑
sys.path.append('/home/pi/DOGZILLA/app_dogzilla/')
from DOGZILLALib import DOGZILLA

def main():
    # 初始化機器狗
    bot = DOGZILLA()
    
    # 設定速度
    bot.motor_speed(50) 
    
    print("--- 官方文檔動作測試 3 (運動組合) ---")

    try:
        # 動作 1：站起 (ID: 2)
        print("執行動作 ID 2：站起 (Stand Up)")
        bot.action(2) 
        time.sleep(3)

        # 動作 2：原地踏步 (ID: 5)
        print("執行動作 ID 5：原地踏步 (Mark Time)")
        bot.action(5)
        time.sleep(5) # 踏步需要多一點時間展示

        # 動作 3：蹲起 (ID: 6)
        print("執行動作 ID 6：蹲起 (Squat)")
        bot.action(6)
        time.sleep(4)

        # 動作 4：坐下 (ID: 12)
        print("執行動作 ID 12：坐下 (Sit Down)")
        bot.action(12)
        time.sleep(3)

        # 結束測試：恢復初始姿態 (ID: 255)
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
