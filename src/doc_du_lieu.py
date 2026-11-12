"""Đọc dữ liệu gốc, đổi giờ UTC sang giờ Việt Nam, gộp lượt báo thành mẫu, chia ngày huấn luyện và kiểm tra.
Mỗi mẫu = một đoạn đường x một khung 30 phút x một ngày. Tốc độ của mẫu = trung bình các lượt báo trong khung."""
import numpy as np
import pandas as pd

from . import cau_hinh as CH


def doc_tep(ten):
    """Đọc một tệp gốc: luot_bao, nhan, doan, nut, duong, mua."""
    return pd.read_csv(CH.DU_LIEU_GOC / CH.TEP_GOC[ten], encoding="utf-8-sig")


def khoang_cach_m(lat1, lon1, lat2, lon2):
    """Khoảng cách đường thẳng (m) giữa hai tọa độ, công thức haversine."""
    lat1, lon1, lat2, lon2 = [np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2)]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def doc_luot_bao():
    """Từng lượt báo tốc độ, thêm ngày và khung 30 phút theo giờ UTC (để ghép với nhãn) và giờ Việt Nam."""
    st = doc_tep("luot_bao").rename(columns={"_id": "report_id"})
    t = pd.to_datetime(st["updated_at"], utc=True, format="ISO8601")
    st["date"] = t.dt.strftime("%Y-%m-%d")
    st["period"] = "period_" + t.dt.hour.astype(str) + "_" + ((t.dt.minute // 30) * 30).astype(str).str.zfill(2)
    st["ts_vn"] = t.dt.tz_convert(None) + pd.Timedelta(hours=CH.UTC_LECH_GIO)
    return st


def doc_doan():
    """Toàn bộ đoạn đường, kèm tọa độ hai đầu, trung điểm và khoảng cách đường thẳng."""
    doan = doc_tep("doan").rename(columns={"_id": "segment_id"})
    doan = doan.drop(columns=["created_at", "updated_at"], errors="ignore")
    nut = doc_tep("nut").rename(columns={"_id": "node_id", "long": "lon"}).set_index("node_id")[["lat", "lon"]]
    doan = doan.join(nut.add_suffix("_s"), on="s_node_id").join(nut.add_suffix("_e"), on="e_node_id")
    doan["lat"] = (doan["lat_s"] + doan["lat_e"]) / 2
    doan["lon"] = (doan["lon_s"] + doan["lon_e"]) / 2
    doan["dist_m"] = khoang_cach_m(doan["lat_s"], doan["lon_s"], doan["lat_e"], doan["lon_e"])
    return doan


def quy_ra_los(toc_do, vmax=None):
    """Quy các tốc độ (km/h) ra mức LOS theo bảng ngưỡng. vmax: tốc độ tối đa cho phép của từng đoạn (có thể thiếu)."""
    toc_do = np.atleast_1d(np.asarray(toc_do, dtype=float))
    vmax = np.full(len(toc_do), np.nan) if vmax is None else np.atleast_1d(np.asarray(vmax, dtype=float))
    cot = np.where(np.isnan(vmax) | (vmax <= 50), 50, np.where(vmax <= 60, 60, 70))
    nguong = np.array([CH.NGUONG_LOS[c] for c in cot])
    dat = toc_do[:, None] >= nguong                       # True nếu tốc độ đạt ngưỡng của mức A..E
    bac = np.where(dat.any(axis=1), dat.argmax(axis=1), 5)  # mức đầu tiên đạt ngưỡng, không có thì F
    ket_qua = np.array(CH.MUC_LOS, dtype=object)[bac]
    ket_qua[np.isnan(toc_do)] = None
    return ket_qua


def bac_los(muc):
    """A..F -> 0..5."""
    return pd.Series(muc).map({m: i for i, m in enumerate(CH.MUC_LOS)}).to_numpy()


def tai_tao_nhan(luot_bao, vmax_doan):
    """Tạo lại nhãn LOS theo cách của tác giả bộ dữ liệu: quy từng lượt ra mức (A=0 ... F=5),
    lấy trung bình chỉ số mức trong khung rồi làm tròn."""
    vmax = luot_bao["segment_id"].map(vmax_doan)
    bac = pd.Series(bac_los(quy_ra_los(luot_bao["velocity"], vmax)), index=luot_bao.index)
    tb = bac.groupby([luot_bao["segment_id"], luot_bao["date"], luot_bao["period"]]).mean()
    return np.round(tb).astype(int).clip(0, 5).map(dict(enumerate(CH.MUC_LOS)))


def gop_mau(luot_bao):
    """Gộp lượt báo theo (đoạn, ngày, khung 30 phút), ghép với nhãn trong train.csv, thêm các cột thời gian."""
    nhan = doc_tep("nhan")
    khoa = ["segment_id", "date", "period"]
    goc = luot_bao["velocity_raw"] if "velocity_raw" in luot_bao else luot_bao["velocity"]   # tốc độ trước khi cắt (v1)
    lb = luot_bao.assign(v_goc=goc, la_1=(goc == 1), la_0=(goc == 0))
    tong = lb.groupby(khoa, sort=False).agg(
        speed=("velocity", "mean"), speed_raw=("v_goc", "mean"), speed_min=("v_goc", "min"),
        speed_max=("v_goc", "max"), n_reports=("velocity", "size"), n_1kmh=("la_1", "sum"),
        n_0kmh=("la_0", "sum")).reset_index()
    df = nhan.merge(tong, on=khoa, how="left")

    gio = df["period"].str.extract(r"period_(\d+)_(\d+)").astype(int)       # giờ UTC của khung
    dt_utc = pd.to_datetime(df["date"], format="%Y-%m-%d") + pd.to_timedelta(gio[0] * 60 + gio[1], unit="min")
    dt_vn = dt_utc + pd.Timedelta(hours=CH.UTC_LECH_GIO)
    df["dt_vn"] = dt_vn
    df["date_vn"] = dt_vn.dt.strftime("%Y-%m-%d")
    df["hour_vn"] = dt_vn.dt.hour + dt_vn.dt.minute / 60                       # 7.5 = 7 giờ 30
    df["weekday"] = dt_vn.dt.weekday                                           # 0 = Thứ hai
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["los_level"] = df["LOS"].map({m: i for i, m in enumerate(CH.MUC_LOS)})
    df = df.rename(columns={"_id": "sample_id", "long_snode": "lon_s", "lat_snode": "lat_s",
                            "long_enode": "lon_e", "lat_enode": "lat_e"})
    df["lat"] = (df["lat_s"] + df["lat_e"]) / 2
    df["lon"] = (df["lon_s"] + df["lon_e"]) / 2
    cot = ["sample_id", "segment_id", "street_id", "street_name", "street_type", "street_level", "max_velocity",
           "length", "s_node_id", "e_node_id", "lat", "lon", "lat_s", "lon_s", "lat_e", "lon_e", "date", "period",
           "dt_vn", "date_vn", "hour_vn", "weekday", "is_weekend", "LOS", "los_level", "speed", "speed_raw",
           "speed_min", "speed_max", "n_reports", "n_1kmh", "n_0kmh"]
    return df[cot].sort_values(["dt_vn", "segment_id"]).reset_index(drop=True)


def chia_ngay(ngay, so_phan):
    """Chia các ngày thành so_phan phần có số mẫu gần bằng nhau: ngày nhiều mẫu xét trước,
    mỗi ngày vào phần đang có ít mẫu nhất. Trả về {ngày: số phần}."""
    dem = ngay.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    tai = [0] * so_phan
    phan_cua = {}
    for n in dem.index:
        k = tai.index(min(tai))
        phan_cua[n] = k
        tai[k] = tai[k] + dem[n]
    return phan_cua


def doc_mau(phien_ban, kem_chia=True):
    """Đọc bảng mẫu đã làm sạch (b2), thêm cột tap (huan_luyen / test) và phan (0-3, -1 nếu là test) (b3)."""
    df = pd.read_csv(CH.DU_LIEU_SACH / f"mau_{phien_ban}.csv", encoding="utf-8-sig")
    df["dt_vn"] = pd.to_datetime(df["dt_vn"], format="%Y-%m-%d %H:%M")
    if kem_chia:
        chia = pd.read_csv(CH.DU_LIEU_SACH / "chia_du_lieu.csv", encoding="utf-8-sig")
        chia = chia[chia["phien_ban"] == phien_ban].set_index("date_vn")
        df["tap"] = df["date_vn"].map(chia["tap"])
        df["phan"] = df["date_vn"].map(chia["phan"])
    return df
