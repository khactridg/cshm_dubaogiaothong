"""Đường dẫn, hằng số và cách vẽ hình dùng chung cho cả dự án."""
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
DU_LIEU_GOC = GOC / "data" / "raw"
DU_LIEU_SACH = GOC / "data" / "sach"
KET_QUA = GOC / "ket_qua"
MO_HINH = GOC / "mo_hinh"

TEP_GOC = {"luot_bao": "segment_status.csv", "nhan": "train.csv", "doan": "segments.csv",
           "nut": "nodes.csv", "duong": "streets.csv", "mua": "thoi_tiet_luoi.csv"}

PHIEN_BAN = ["v1", "v2"]
TEP_MO_HINH = {"Mốc: trung bình": "moc_trung_binh", "Mốc: lịch sử đoạn": "moc_lich_su_doan", "Hồi quy tuyến tính": "hoi_quy_tuyen_tinh",
               "Cây quyết định": "cay_quyet_dinh", "Random Forest": "random_forest", "XGBoost": "xgboost"}   # tên tệp trong mo_hinh/<v>/
SEED = 42
UTC_LECH_GIO = 7                  # dữ liệu gốc ghi giờ UTC, giờ Việt Nam = UTC + 7

# Ngưỡng làm sạch
NGAY_BAT_THUONG = "2020-08-03"    # ngày có tốc độ sinh ngẫu nhiên
V1_TRAN_KMH = 80                  # v1: lượt trên 80 km/h cắt về 80
V2_TRAN_KMH = 120                 # v2: bỏ lượt trên 120 km/h
NOI_THANH_KMH = 70                # v2: bỏ lượt từ 70 km/h trên đường không phải cao tốc, quốc lộ
DUONG_TOC_DO_CAO = ("motorway", "motorway_link", "trunk", "trunk_link")
DUNG_YEN_TOC_DO = 3               # v2: chuỗi đứng yên, mọi lượt không quá 3 km/h
DUNG_YEN_TI_LE = 3                # v2: quãng đường theo tốc độ báo từ 3 lần chiều dài đoạn
DONG_BANG_LUOT = 20               # v2: từ 20 lượt liền nhau cùng một giá trị
KHOANG_LIEN_NHAU_GIAY = 120       # hai lượt cách nhau quá 2 phút thì không tính là liền nhau
TRAN_DU_BAO = {"v1": V1_TRAN_KMH, "v2": V2_TRAN_KMH}   # dự báo được chặn trong [0, trần]

MUC_LOS = ["A", "B", "C", "D", "E", "F"]
TEN_LOS = {"A": "Thông thoáng", "B": "Khá thoáng", "C": "Đông xe", "D": "Đông, chậm", "E": "Ùn ứ", "F": "Kẹt cứng"}
# Ngưỡng dưới (km/h) của mức A, B, C, D, E theo tốc độ tối đa cho phép của đoạn; thấp hơn ngưỡng E là F.
# Bảng 1 của Mai-Tan và cs. (2020). Đoạn không rõ hoặc <= 50 km/h dùng cột 50, <= 60 dùng cột 60, còn lại cột 70.
NGUONG_LOS = {50: (35, 30, 25, 20, 15), 60: (40, 35, 30, 25, 20), 70: (45, 40, 35, 30, 25)}

# Màu và cỡ chữ của hình
XANH = "#0E4A75"
DO = "#D1373F"
XAM = "#9AA5AE"
CAM = "#E08A1E"
XANH_NHAT = "#8DB6D6"


def tao_thu_muc():
    for p in (DU_LIEU_SACH, KET_QUA, MO_HINH):
        p.mkdir(parents=True, exist_ok=True)


def thiet_lap_hinh():
    """Cỡ chữ đủ đọc khi in A4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "figure.dpi": 100,
                         "axes.spines.top": False, "axes.spines.right": False})


def luu_hinh(fig, ten):
    fig.tight_layout()
    fig.savefig(KET_QUA / ten, dpi=200)
    print("  đã ghi", ten)


def so_vn(x, n=2):
    """Số theo kiểu Việt Nam: dấu phẩy thập phân."""
    return f"{x:.{n}f}".replace(".", ",")
