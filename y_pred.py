EPOCH_SEC = 30

def stage_to_sleepmetrics(stages_epoch):
    # stages_epoch: array ของค่าคลาสต่อ epoch คืนเดียว (0=Wake,1=NREM,2=REM)
    n_epochs = len(stages_epoch)
    time_in_bed_min = n_epochs * EPOCH_SEC / 60.0

    # นับว่าเป็นหลับถ้าไม่ใช่ Wake
    asleep_mask = stages_epoch != 0
    TST_min = asleep_mask.sum() * EPOCH_SEC / 60.0

    # WASO: ช่วงตื่นที่เกิด "หลังจาก" เริ่มหลับครั้งแรก
    # หา epoch หลับครั้งแรก
    first_sleep = np.argmax(asleep_mask) if asleep_mask.any() else n_epochs
    waso_epochs = ((stages_epoch[first_sleep:] == 0).sum())
    WASO_min = waso_epochs * EPOCH_SEC / 60.0

    SE = 100.0 * TST_min / max(time_in_bed_min, 1e-6)
    return SE, WASO_min

def quality_label(SE, WASO):
    if SE >= 85 and WASO <= 30:
        return 2   # 2=ดี
    if (75 <= SE < 85) or (30 < WASO <= 60):
        return 1   # 1=กลาง
    return 0       # 0=ไม่ดี

# ตัวอย่าง: ประเมินหลายคืน
night_labels = []
for night_idx, idx_slice in enumerate(indices_per_night):
    stages = y_pred[idx_slice]            # hypnogram สำหรับคืนนี้
    SE, WASO = stage_to_sleepmetrics(stages)
    lbl = quality_label(SE, WASO)
    night_labels.append((SE, WASO, lbl))
