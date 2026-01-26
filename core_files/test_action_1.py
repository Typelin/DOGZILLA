import time
from DOGZILLALib import DOGZILLA

# 1. 初始化機器狗實例
print("正在初始化 DOGZILLA...")
bot = DOGZILLA()

try:
    # 2. 設定電機速度 (0-100)
    bot.motor_speed(50)
    
    # 3. 執行預設動作組
    # 動作編號 14 是「伸懶腰」 (與 app_dogzilla 開機動作相同)
    print("執行動作：伸懶腰...")
    bot.action(14)
    
    # 等待動作完成 (給予 3 秒時間)
    time.sleep(3)
    
    # 4. 回到初始位置
    print("恢復初始姿勢...")
    bot.action(0xff) # 0xff 通常代表重置到標準站姿
    time.sleep(1)

except Exception as e:
    print(f"發生錯誤: {e}")

finally:
    # 5. 結束程式前務必釋放資源
    del bot
    print("測試結束，程式已安全關閉。")