"""
config.py - Đường dẫn và các hằng số dùng chung cho cả dự án.
"""
import os
from pathlib import Path

# Phiên bản làm sạch: v1 (cách ban đầu) hoặc v2 (cách bổ sung, phiên bản chính).
# Đổi phiên bản trong cửa sổ lệnh trước khi chạy:  set LAM_SACH=v1   (PowerShell: $env:LAM_SACH="v1")
VERSIONS = ("v1", "v2")
VERSION = os.environ.get("LAM_SACH", "v2").strip().lower()

# Thư mục
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT.parent / "data" / "raw"      # dùng chung dữ liệu gốc với thư mục ngoài
DATA_PROCESSED = ROOT / "data" / "processed" / VERSION
RESULTS = ROOT / "results" / VERSION
MODELS_DIR = ROOT / "models" / VERSION
RESULTS_COMMON = ROOT / "results" / "chung"      # kết quả không phụ thuộc phiên bản (E01, E06, E07)

RAW_FILES = {
    "train": "train.csv",                 # nhãn LOS của (đoạn, ngày, khung 30 phút)
    "status": "segment_status.csv",       # từng lượt người dùng báo tốc độ
    "segments": "segments.csv",           # các đoạn đường
    "nodes": "nodes.csv",                 # các nút (tọa độ)
    "streets": "streets.csv",             # các tuyến đường
}

SEED = 42
UTC_OFFSET_HOURS = 7          # dữ liệu gốc ghi giờ UTC, giờ Việt Nam = UTC + 7
SLOT_MINUTES = 30

# Ngưỡng của các quy tắc làm sạch (giải thích trong src/cleaning.py)
DAY_BULK = "2020-08-03"       # QT2: ngày dữ liệu sinh ngẫu nhiên
V1_CAP_KMH = 80               # v1, QT3: lượt trên 80 km/h cắt về 80
V2_MAX_KMH = 120              # v2, QT3: bỏ lượt trên 120 km/h
STATIONARY_MAX_KMH = 3          # v2, QT5: mọi lượt trong chuỗi <= 3 km/h
STATIONARY_RATIO = 3         # v2, QT5: quãng đường theo tốc độ báo >= 3 lần chiều dài đoạn
FROZEN_MIN_RUN = 20               # v2, QT6: >= 20 lượt liền nhau cùng một giá trị
URBAN_MAX_KMH = 70         # v2, QT4: lượt >= 70 km/h trên đường không phải cao tốc, quốc lộ
HIGHWAY_TYPES = ("motorway", "motorway_link", "trunk", "trunk_link")
SESSION_GAP_S = 120           # hai lượt cách nhau quá 2 phút thì không tính là liền nhau

# Dự báo tốc độ được giới hạn trong [0, SPEED_CAP_KMH]
SPEED_CAP_KMH = V1_CAP_KMH if VERSION == "v1" else V2_MAX_KMH

LOS_LEVELS = ["A", "B", "C", "D", "E", "F"]
LOS_NAMES = {"A": "Thông thoáng", "B": "Khá thoáng", "C": "Đông xe",
             "D": "Đông, chậm", "E": "Ùn ứ", "F": "Kẹt cứng"}
WEEKDAY_VN = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]

# Bảng 1 của Mai-Tan và cs. (2020): ngưỡng dưới (km/h) của mức A, B, C, D, E;
# thấp hơn ngưỡng của E là F. Khóa là tốc độ tối đa cho phép của đoạn
# (không rõ hoặc <= 50 dùng cột 50; <= 60 dùng cột 60; còn lại cột 70).
LOS_THRESHOLDS = {
    50: (35, 30, 25, 20, 15),
    60: (40, 35, 30, 25, 20),
    70: (45, 40, 35, 30, 25),
}


def ensure_dirs():
    for p in (DATA_PROCESSED, RESULTS, MODELS_DIR, RESULTS_COMMON):
        p.mkdir(parents=True, exist_ok=True)
