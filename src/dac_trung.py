"""Tạo đặc trưng cho mô hình.
Nhóm A, thời gian: giờ, thứ, cuối tuần, buổi trong ngày.
Nhóm B, đoạn đường: loại, cấp, tốc độ tối đa, chiều dài, vị trí, số đoạn nối vào và nối ra.
Nhóm C, lịch sử: tốc độ trung bình của chính đoạn (hoặc tuyến đường) trong các ngày thuộc tập huấn luyện, có co rút.
Nhóm A, B chỉ dùng thông tin biết trước (lịch, bản đồ). Nhóm C dùng đáp án nên chỉ học từ tập huấn luyện;
trên chính tập huấn luyện, mỗi mẫu được tính từ các ngày khác (ngoài mẫu), để mô hình không thấy đáp án của chính nó."""
import numpy as np
import pandas as pd

from . import doc_du_lieu as DL

LOAI_DUONG_CHINH = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
                    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]
KAPPA = 5      # số "mẫu ảo": đoạn ít mẫu bị kéo về trung bình chung


def buoi_trong_ngay(gio):
    if 6 <= gio < 9:
        return "cao_diem_sang"
    if 9 <= gio < 16:
        return "ban_ngay"
    if 16 <= gio < 19:
        return "cao_diem_chieu"
    if 19 <= gio < 22:
        return "toi"
    return "dem"


def dac_trung_nen(df, doan=None):
    """Nhóm A và B. doan: bảng đoạn đường (để tính số đoạn nối vào, nối ra); None thì đọc từ dữ liệu gốc."""
    if doan is None:
        doan = DL.doc_doan()
    vao = doan.groupby("e_node_id").size()       # số đoạn đi vào mỗi nút
    ra = doan.groupby("s_node_id").size()        # số đoạn đi ra từ mỗi nút
    X = pd.DataFrame(index=df.index)
    X["hour"] = df["hour_vn"]                    # 7.5 = 7 giờ 30
    X["hour_int"] = df["hour_vn"].astype(int)    # giờ nguyên, hồi quy tuyến tính mã hóa one-hot
    X["weekday"] = df["weekday"]
    X["is_weekend"] = df["is_weekend"]
    X["day_part"] = df["hour_vn"].apply(buoi_trong_ngay)
    X["street_type"] = df["street_type"].where(df["street_type"].isin(LOAI_DUONG_CHINH), "khac")
    X["street_level"] = df["street_level"]
    X["max_velocity"] = df["max_velocity"]       # thiếu ở khoảng 85% mẫu, để trống
    X["max_velocity_missing"] = df["max_velocity"].isna().astype(int)
    X["log_length"] = np.log1p(df["length"])
    X["lat"] = df["lat"]
    X["lon"] = df["lon"]
    X["n_in"] = df["s_node_id"].map(vao).fillna(0).astype(int)
    X["n_out"] = df["e_node_id"].map(ra).fillna(0).astype(int)
    return X


def trung_binh_co_rut(khoa_hl, y_hl, khoa_ap, cot):
    """Trung bình tốc độ theo nhóm `cot`, có co rút:
    (tổng tốc độ của nhóm + KAPPA x trung bình chung) / (số mẫu của nhóm + KAPPA).
    Học từ khoa_hl, y_hl và áp lên khoa_ap. Trả về (giá trị, số mẫu của nhóm)."""
    bang = khoa_hl[cot].copy()
    bang["y"] = y_hl.to_numpy()
    thong_ke = bang.groupby(cot)["y"].agg(["sum", "count"]).reset_index()
    chung = y_hl.mean()
    ghep = khoa_ap[cot].merge(thong_ke, on=cot, how="left")
    tong = ghep["sum"].fillna(0).to_numpy()
    dem = ghep["count"].fillna(0).to_numpy()
    return (tong + KAPPA * chung) / (dem + KAPPA), dem


def khoa_lich_su(df):
    k = df[["segment_id", "street_id"]].copy()
    k["day_part"] = df["hour_vn"].apply(buoi_trong_ngay)
    return k


def khoi_lich_su(khoa_hl, y_hl, khoa_ap):
    """Bốn đặc trưng nhóm C cho khoa_ap, học từ khoa_hl."""
    ra = pd.DataFrame(index=khoa_ap.index)
    gia_tri, dem = trung_binh_co_rut(khoa_hl, y_hl, khoa_ap, ["segment_id"])
    ra["hist_segment"] = gia_tri                         # trung bình của đoạn
    ra["hist_segment_log_n"] = np.log1p(dem)             # đoạn đã có bao nhiêu mẫu
    ra["hist_segment_daypart"], _ = trung_binh_co_rut(khoa_hl, y_hl, khoa_ap, ["segment_id", "day_part"])
    ra["hist_street"], _ = trung_binh_co_rut(khoa_hl, y_hl, khoa_ap, ["street_id"])
    return ra[["hist_segment", "hist_segment_daypart", "hist_street", "hist_segment_log_n"]]


def dac_trung_lich_su(df_hl, y_hl, df_ap=None, nhom_trong=None):
    """Nhóm C. Có df_ap: học từ toàn bộ df_hl rồi áp lên df_ap (tập kiểm tra).
    Không có df_ap: tính cho chính df_hl theo kiểu ngoài mẫu, mỗi phần trong nhom_trong được tính từ các phần còn lại."""
    khoa_hl = khoa_lich_su(df_hl)
    if df_ap is not None:
        return khoi_lich_su(khoa_hl, y_hl, khoa_lich_su(df_ap))
    cac_phan = []
    for g in np.unique(nhom_trong):
        trong = nhom_trong == g
        cac_phan.append(khoi_lich_su(khoa_hl[~trong], y_hl[~trong], khoa_hl[trong]))
    return pd.concat(cac_phan).loc[df_hl.index]


def dac_trung_mot_lan_chia(df, nen, chi_so_hl, chi_so_kt, nhom_trong):
    """Bảng đặc trưng cho phần huấn luyện và phần kiểm tra của một lần chia (nhóm C tính riêng cho lần chia này)."""
    y = df["speed"]
    hl, kt = df.iloc[chi_so_hl], df.iloc[chi_so_kt]
    c_hl = dac_trung_lich_su(hl, y.iloc[chi_so_hl], nhom_trong=nhom_trong)
    c_kt = dac_trung_lich_su(hl, y.iloc[chi_so_hl], df_ap=kt)
    return pd.concat([nen.iloc[chi_so_hl], c_hl], axis=1), pd.concat([nen.iloc[chi_so_kt], c_kt], axis=1)


def dac_trung_huan_luyen(hl):
    """Đặc trưng cho toàn tập huấn luyện hl (có cột phan). Nhóm C tính ngoài mẫu theo 4 phần."""
    lich_su = dac_trung_lich_su(hl, hl["speed"], nhom_trong=hl["phan"].to_numpy())
    return pd.concat([dac_trung_nen(hl), lich_su], axis=1)


def dac_trung_cho(df_hl, df_ap):
    """Đặc trưng của df_ap cho một mô hình học từ df_hl (nhóm C chỉ học từ df_hl)."""
    return pd.concat([dac_trung_nen(df_ap), dac_trung_lich_su(df_hl, df_hl["speed"], df_ap=df_ap)], axis=1)
