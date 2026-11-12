"""
data.py - Đọc dữ liệu gốc và tạo bảng mẫu.

Mỗi mẫu = 1 đoạn đường x 1 khung 30 phút x 1 ngày. Tốc độ của mẫu (cột speed) là
trung bình các lượt báo trong khung đó. Thời gian trong dữ liệu gốc là giờ UTC.
"""
import numpy as np
import pandas as pd

from . import config as C


def read_raw(name):
    """Đọc một tệp gốc: train, status, segments, nodes, streets."""
    return pd.read_csv(C.DATA_RAW / C.RAW_FILES[name], encoding="utf-8")


def haversine_m(lat1, lon1, lat2, lon2):
    """Khoảng cách đường thẳng (m) giữa hai điểm tọa độ (công thức haversine)."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def los_of_speed(speed, max_velocity):
    """Mức LOS của một tốc độ (km/h), theo bảng ngưỡng C.LOS_THRESHOLDS."""
    if pd.isna(speed):
        return None
    if pd.isna(max_velocity) or max_velocity <= 50:
        cuts = C.LOS_THRESHOLDS[50]
    elif max_velocity <= 60:
        cuts = C.LOS_THRESHOLDS[60]
    else:
        cuts = C.LOS_THRESHOLDS[70]
    for level, cut in zip(["A", "B", "C", "D", "E"], cuts):
        if speed >= cut:
            return level
    return "F"


def speed_to_los(speed, max_velocity=None):
    """Quy cả một dãy tốc độ ra mức LOS. max_velocity: dãy tốc độ tối đa cho phép (có thể bỏ trống)."""
    speed = np.atleast_1d(np.asarray(speed, dtype=float))
    if max_velocity is None:
        max_velocity = np.full(len(speed), np.nan)
    max_velocity = np.atleast_1d(np.asarray(max_velocity, dtype=float))
    result = []
    for i in range(len(speed)):
        result.append(los_of_speed(speed[i], max_velocity[i]))
    return np.array(result, dtype=object)


def los_from_reports(reports, max_velocity_by_segment):
    """Tạo lại nhãn LOS theo cách của tác giả bộ dữ liệu: quy từng lượt báo ra mức (A=0 ... F=5),
    lấy trung bình chỉ số mức trong mỗi khung rồi làm tròn."""
    level_number = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    vmax = reports["segment_id"].map(max_velocity_by_segment)
    level = pd.Series(speed_to_los(reports["velocity"], vmax), index=reports.index).map(level_number)
    mean_level = level.groupby([reports["segment_id"], reports["date"], reports["period"]]).mean()
    rounded = np.round(mean_level).astype(int).clip(0, 5)
    return rounded.map({0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"})


def load_reports():
    """Từng lượt báo tốc độ, thêm ngày và khung 30 phút (giờ UTC) để ghép với nhãn."""
    st = read_raw("status").rename(columns={"_id": "report_id"})
    t = pd.to_datetime(st["updated_at"], utc=True, format="ISO8601")
    slot = (t.dt.minute // 30) * 30
    st["date"] = t.dt.strftime("%Y-%m-%d")
    st["period"] = "period_" + t.dt.hour.astype(str) + "_" + slot.astype(str).str.zfill(2)
    st["ts_vn"] = t.dt.tz_convert(None) + pd.Timedelta(hours=C.UTC_OFFSET_HOURS)   # giờ Việt Nam
    return st


def build_samples(reports):
    """Gộp các lượt báo theo (đoạn, ngày, khung 30 phút), ghép với nhãn trong train.csv."""
    train = read_raw("train")
    keys = ["segment_id", "date", "period"]

    # velocity_raw: tốc độ gốc trước khi cắt (chỉ có ở v1)
    raw = reports["velocity_raw"] if "velocity_raw" in reports else reports["velocity"]
    rep = reports.assign(v_raw=raw, is_1=(raw == 1), is_0=(raw == 0), over_80=(raw > 80))
    agg = rep.groupby(keys, sort=False).agg(
        speed=("velocity", "mean"), speed_raw=("v_raw", "mean"), speed_median=("velocity", "median"),
        speed_min=("v_raw", "min"), speed_max=("v_raw", "max"), n_reports=("velocity", "size"),
        n_1kmh=("is_1", "sum"), n_0kmh=("is_0", "sum"), n_over80=("over_80", "sum")).reset_index()
    df = train.merge(agg, on=keys, how="left")

    # đổi giờ UTC sang giờ Việt Nam
    hm = df["period"].str.extract(r"period_(\d+)_(\d+)").astype(int)
    dt_utc = pd.to_datetime(df["date"], format="%Y-%m-%d") + pd.to_timedelta(hm[0] * 60 + hm[1], unit="min")
    dt_vn = dt_utc + pd.Timedelta(hours=C.UTC_OFFSET_HOURS)
    df["dt_vn"] = dt_vn
    df["date_vn"] = dt_vn.dt.strftime("%Y-%m-%d")
    df["month_vn"] = dt_vn.dt.strftime("%Y-%m")
    df["hour_vn"] = dt_vn.dt.hour + dt_vn.dt.minute / 60          # 7.5 = 7 giờ 30
    df["slot_vn"] = dt_vn.dt.hour * 2 + dt_vn.dt.minute // 30      # 0..47
    df["weekday"] = dt_vn.dt.weekday                               # 0 = Thứ 2 ... 6 = Chủ nhật
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["los_level"] = df["LOS"].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5})
    df = df.rename(columns={"_id": "sample_id", "long_snode": "lon_s", "lat_snode": "lat_s",
                            "long_enode": "lon_e", "lat_enode": "lat_e"})
    df["lat"] = (df["lat_s"] + df["lat_e"]) / 2
    df["lon"] = (df["lon_s"] + df["lon_e"]) / 2

    cols = ["sample_id", "segment_id", "street_id", "street_name", "street_type", "street_level",
            "max_velocity", "length", "s_node_id", "e_node_id", "lat", "lon", "lat_s", "lon_s", "lat_e", "lon_e",
            "date", "period", "dt_vn", "date_vn", "month_vn", "hour_vn", "slot_vn", "weekday", "is_weekend",
            "LOS", "los_level", "speed", "speed_raw", "speed_median", "speed_min", "speed_max",
            "n_reports", "n_1kmh", "n_0kmh", "n_over80"]
    return df[cols].sort_values(["dt_vn", "segment_id"]).reset_index(drop=True)


def load_segments():
    """Toàn bộ đoạn đường, kèm tọa độ hai đầu, trung điểm và khoảng cách đường thẳng."""
    seg = read_raw("segments").rename(columns={"_id": "segment_id"})
    seg = seg.drop(columns=["created_at", "updated_at"], errors="ignore")
    nodes = read_raw("nodes").rename(columns={"_id": "node_id", "long": "lon"}).set_index("node_id")[["lat", "lon"]]
    seg = seg.join(nodes.add_suffix("_s"), on="s_node_id").join(nodes.add_suffix("_e"), on="e_node_id")
    seg["lat"] = (seg["lat_s"] + seg["lat_e"]) / 2
    seg["lon"] = (seg["lon_s"] + seg["lon_e"]) / 2
    seg["dist_m"] = haversine_m(seg["lat_s"], seg["lon_s"], seg["lat_e"], seg["lon_e"])
    return seg


SAMPLES_FILE = C.DATA_PROCESSED / "samples.csv"


def load_clean_samples():
    """Đọc bảng mẫu đã làm sạch (do tao_du_lieu_sach.py tạo)."""
    df = pd.read_csv(SAMPLES_FILE, encoding="utf-8-sig")
    df["dt_vn"] = pd.to_datetime(df["dt_vn"], format="%Y-%m-%d %H:%M")
    return df
