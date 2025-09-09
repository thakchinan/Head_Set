# ... หลัง config ถูกสร้างแล้ว
from data_loader.data_loaders import load_folds_data, load_folds_data_shhs
from data_loader.data_loaders import data_generator_np
from utils.util import calc_class_weight

# โหลดข้อมูล + folds
if "shhs" in args2.np_data_dir:
    (X, y), folds = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
else:
    (X, y), folds = load_folds_data(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

# ประกาศเป็น global ให้ data_generator_np ใช้
GLOBAL_X, GLOBAL_Y = X, y

folds_data = folds  # ให้ชื่อเหมือนที่คุณใช้ใน main()

main(config, fold_id)
