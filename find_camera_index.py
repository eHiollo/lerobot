import cv2
import os
import sys

print("正在扫描摄像头索引 (可能需要几秒钟)...")
print("注意：深度相机通常会占用多个 /dev/video* 节点（RGB、深度、红外等）。")
print("我们需要找到能输出彩色图像 (RGB/BGR) 的那个索引。\n")

found_any = False

# 扫描前 20 个索引
for index in range(20):
    dev_path = f"/dev/video{index}"
    if not os.path.exists(dev_path):
        continue
        
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        # 尝试设置一个常见的 RGB 分辨率，看是否支持
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 尝试读一帧
        ret, frame = cap.read()
        if ret:
            found_any = True
            h, w, c = frame.shape
            print(f"✅ [索引 {index}] ({dev_path}) -> 成功读取! 分辨率: {w}x{h}")
            
            # 保存一张图片供确认
            img_name = f"test_cam_{index}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"   已保存测试图片: {img_name} (请查看此图确认是否为 RGB 画面)")
        else:
            print(f"❌ [索引 {index}] ({dev_path}) -> 无法读取画面 (可能是深度/红外流，或者格式不支持)")
        cap.release()
    else:
        print(f"❌ [索引 {index}] ({dev_path}) -> 无法打开")

if not found_any:
    print("\n未找到任何可用的摄像头。请检查连接。")
else:
    print("\n请查看生成的 .jpg 图片，确定哪个索引是你需要的 RGB 镜头。")
