"""Các quy tắc làm sạch dữ liệu, đánh mã QT1-QT10 theo thứ tự áp dụng.

  Mã    Quy tắc                                       v1 (làm sạch ban đầu)     v2 (làm sạch bổ sung)
  QT1   lượt báo trùng lặp                             bỏ                        bỏ
  QT2   ngày 03/08/2020 (tốc độ sinh ngẫu nhiên)       giữ, đánh dấu             bỏ toàn bộ
  QT3   tốc độ quá cao                                 cắt lượt trên 80 về 80    bỏ lượt trên 120
  QT4   từ 70 km/h trên đường không phải cao tốc, QL   giữ                       bỏ
  QT5   thiết bị đứng yên (chuỗi lượt liền nhau trên   giữ                       bỏ
        một đoạn, mọi lượt <= 3 km/h, quãng đường theo
        tốc độ báo >= 3 lần chiều dài đoạn)
  QT6   tốc độ đóng băng (>= 20 lượt liền nhau cùng    giữ                       bỏ
        một giá trị, khác 1 km/h)
  QT7   lượt báo 0 km/h                                giữ                       giữ
  QT8   lượt báo đúng 1 km/h                           giữ                       giữ
  QT9   mẫu không còn lượt báo nào                     không xảy ra              bỏ mẫu
  QT10  ngày có dưới 10 mẫu                            giữ                       giữ
"Liền nhau" nghĩa là đứng cạnh nhau khi xếp mọi lượt báo theo thời gian và cách nhau không quá 2 phút."""
import pandas as pd

from . import cau_hinh as CH


def ghi_nhat_ky(nhat_ky, ma, quy_tac, so_luong, tong, cach_xu_ly):
    nhat_ky.append({"ma": ma, "quy_tac": quy_tac, "so_bi_anh_huong": int(so_luong), "tong": int(tong),
                    "ty_le": round(so_luong / tong, 4) if tong else 0.0, "cach_xu_ly": cach_xu_ly})


def sap_xep_luot_bao(rep):
    """Xếp các lượt báo theo thời gian. Thêm cột t (thời điểm), dt (giây từ lượt liền trước),
    ngat (True nếu cách lượt trước quá 2 phút)."""
    rep = rep.copy()
    rep["t"] = pd.to_datetime(rep["updated_at"], utc=True, format="ISO8601")
    rep = rep.sort_values(["t", "report_id"], kind="stable").reset_index(drop=True)
    rep["dt"] = rep["t"].diff().dt.total_seconds()
    rep["ngat"] = rep["dt"].isna() | (rep["dt"] > CH.KHOANG_LIEN_NHAU_GIAY)
    return rep


def danh_so_chuoi(rep, cot):
    """Các lượt liền nhau có cùng giá trị ở cột `cot` thuộc cùng một chuỗi. Ví dụ đoạn = 5,5,5,8,8,5 -> chuỗi 1,1,1,2,2,3."""
    chuoi_moi = (rep[cot] != rep[cot].shift()) | rep["ngat"]
    return chuoi_moi.cumsum()


def la_dung_yen(rep, chieu_dai_doan):
    """QT5: True cho các lượt thuộc chuỗi thiết bị đứng yên."""
    chuoi = danh_so_chuoi(rep, "segment_id")
    g = rep.groupby(chuoi)
    so_luot = g["velocity"].transform("size")
    v_max = g["velocity"].transform("max")
    v_tb = g["velocity"].transform("mean")
    giay = (g["t"].transform("max") - g["t"].transform("min")).dt.total_seconds()
    quang_duong = v_tb / 3.6 * giay                       # mét, nếu thiết bị chạy đúng tốc độ báo
    dai = rep["segment_id"].map(chieu_dai_doan).fillna(0).clip(lower=1)
    return (so_luot >= 2) & (v_max <= CH.DUNG_YEN_TOC_DO) & (quang_duong >= CH.DUNG_YEN_TI_LE * dai)


def la_dong_bang(rep):
    """QT6: True cho các lượt thuộc chuỗi từ 20 lượt liền nhau cùng một giá trị (khác 1 km/h)."""
    chuoi = danh_so_chuoi(rep, "velocity")
    so_luot = rep.groupby(chuoi)["velocity"].transform("size")
    return (so_luot >= CH.DONG_BANG_LUOT) & (rep["velocity"] != 1)


def la_qua_nhanh_noi_thanh(rep, loai_doan):
    """QT4: True cho lượt từ 70 km/h trên đường không phải cao tốc, quốc lộ."""
    cao_toc = rep["segment_id"].map(loai_doan).isin(CH.DUONG_TOC_DO_CAO)
    return (rep["velocity"] >= CH.NOI_THANH_KMH) & ~cao_toc


def lam_sach_luot_bao(luot_bao, doan, phien_ban):
    """Làm sạch ở mức lượt báo. Trả về (bảng lượt báo còn lại, nhật ký)."""
    nhat_ky = []
    trung = luot_bao.duplicated(["segment_id", "updated_at", "velocity"])
    ghi_nhat_ky(nhat_ky, "QT1", "Lượt báo trùng lặp", trung.sum(), len(luot_bao), "bỏ")
    rep = luot_bao[~trung]

    if phien_ban == "v1":
        ghi_nhat_ky(nhat_ky, "QT7", "Lượt báo 0 km/h", (rep["velocity"] == 0).sum(), len(rep), "giữ nguyên")
        ghi_nhat_ky(nhat_ky, "QT8", "Lượt báo đúng 1 km/h", (rep["velocity"] == 1).sum(), len(rep), "giữ nguyên")
        qua_cao = rep["velocity"] > CH.V1_TRAN_KMH
        ghi_nhat_ky(nhat_ky, "QT3", f"Lượt báo trên {CH.V1_TRAN_KMH} km/h", qua_cao.sum(), len(rep),
                    f"cắt về {CH.V1_TRAN_KMH} km/h")
        rep = rep.copy()
        rep["velocity_raw"] = rep["velocity"]
        rep["velocity"] = rep["velocity"].clip(upper=CH.V1_TRAN_KMH)
        return rep, nhat_ky

    # v2: đánh dấu từng quy tắc trên toàn bộ lượt báo, rồi bỏ lần lượt theo thứ tự
    d = doan.set_index("segment_id")
    rep = sap_xep_luot_bao(rep)
    ngay = rep["ts_vn"].dt.strftime("%Y-%m-%d")
    quy_tac = [
        ("QT2", f"Ngày {CH.NGAY_BAT_THUONG} (tốc độ sinh ngẫu nhiên)", ngay == CH.NGAY_BAT_THUONG),
        ("QT3", f"Lượt báo trên {CH.V2_TRAN_KMH} km/h", rep["velocity"] > CH.V2_TRAN_KMH),
        ("QT4", f"Lượt báo từ {CH.NOI_THANH_KMH} km/h trên đường không phải cao tốc, quốc lộ",
         la_qua_nhanh_noi_thanh(rep, d["street_type"])),
        ("QT5", "Thiết bị đứng yên", la_dung_yen(rep, d["length"])),
        ("QT6", f"Tốc độ đóng băng (từ {CH.DONG_BANG_LUOT} lượt liền nhau cùng giá trị)", la_dong_bang(rep)),
    ]
    da_bo = pd.Series(False, index=rep.index)
    for ma, ten, co in quy_tac:
        moi = co & ~da_bo                                 # chỉ đếm các lượt chưa bị quy tắc trước bỏ
        ghi_nhat_ky(nhat_ky, ma, ten, moi.sum(), len(rep), "bỏ")
        da_bo = da_bo | co
    rep = rep[~da_bo].drop(columns=["t", "dt", "ngat"]).reset_index(drop=True)
    ghi_nhat_ky(nhat_ky, "QT7", "Lượt báo 0 km/h", (rep["velocity"] == 0).sum(), len(rep), "giữ nguyên")
    ghi_nhat_ky(nhat_ky, "QT8", "Lượt báo đúng 1 km/h", (rep["velocity"] == 1).sum(), len(rep), "giữ nguyên")
    return rep, nhat_ky


def lam_sach_mau(mau, nhat_ky, phien_ban):
    """Làm sạch ở mức mẫu (sau khi gộp theo khung 30 phút)."""
    df = mau.copy()
    if phien_ban == "v1":
        df["is_bulk_day"] = (df["date_vn"] == CH.NGAY_BAT_THUONG).astype(int)
        ghi_nhat_ky(nhat_ky, "QT2", f"Ngày {CH.NGAY_BAT_THUONG}", df["is_bulk_day"].sum(), len(df),
                    "giữ, đánh dấu cột is_bulk_day")
    else:
        trong = df["speed"].isna()
        ghi_nhat_ky(nhat_ky, "QT9", "Mẫu không còn lượt báo hợp lệ", trong.sum(), len(df), "bỏ mẫu")
        df = df[~trong].reset_index(drop=True)
        df["is_bulk_day"] = 0
    moi_ngay = df.groupby("date_vn")["segment_id"].transform("size")
    ghi_nhat_ky(nhat_ky, "QT10", "Ngày có dưới 10 mẫu", (moi_ngay < 10).sum(), len(df), "giữ nguyên")
    return df
