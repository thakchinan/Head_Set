import numpy as np
from tensorflow.keras.models import load_model

# โหลดโมเดลที่ฝึกจากโน้ตบุ๊ก
model = load_model("model.h5")

# X_spec: สเปกโตรแกรมที่เตรียมตามโน้ตบุ๊ก (เช่น shape [N_epochs, time, mel, channels] หรือรูปแบบที่รีโปกำหนด)
# โหลด/สร้างให้เหมือน gen_spectrogram.ipynb
X_spec = np.load("X_test_spec.npy")     # ตัวอย่าง
y_pred_prob = model.predict(X_spec)     # [N_epochs, 3]
y_pred = y_pred_prob.argmax(axis=-1)    # 0=Wake, 1=NREM, 2=REM (ตรวจ mapping ในโน้ตบุ๊กอีกรอบ)

# สมมุติ 1 คืน = 8 ชม. (= 8*60*2 epochs ถ้า epoch 30 วิ) แบ่งเป็นต่อ “คืน”
# indices_per_night: list ของ slice/indices ที่ชี้ว่าแต่ละคืนประกอบด้วย epoch ไหนบ้าง
