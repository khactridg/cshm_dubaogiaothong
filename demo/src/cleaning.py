"""
cleaning.py - Các quy tắc làm sạch dữ liệu (QT1–QT10, đánh số theo thứ tự áp dụng).

  Mã    Quy tắc                                        v1 (cách ban đầu)          v2 (cách bổ sung)
  QT1   lượt báo trùng lặp                              bỏ                         bỏ
  QT2   ngày 03/08/2020 (tốc độ sinh ngẫu nhiên)        giữ, đánh dấu is_bulk_day  bỏ toàn bộ
  QT3   tốc độ quá cao                                  cắt lượt > 80 về 80        bỏ lượt > 120
  QT4   từ 70 km/h trên đường không phải cao tốc, QL    giữ                        bỏ
  QT5   thiết bị đứng yên: các lượt liền nhau trên      giữ                        bỏ
        cùng đoạn, mọi lượt <= 3 km/h, nhưng quãng
        đường theo tốc độ báo >= 3 lần chiều dài đoạn
  QT6   tốc độ đóng băng: >= 20 lượt liền nhau cùng     giữ                        bỏ
        một giá trị (khác 1 km/h)
  QT7   lượt báo 0 km/h                                 giữ                        giữ
  QT8   lượt báo đúng 1 km/h                            giữ                        giữ
  QT9   mẫu không còn lượt báo nào                      không phát sinh            bỏ mẫu
  QT10  ngày có dưới 10 mẫu                             giữ                        giữ

"Liền nhau" nghĩa là đứng cạnh nhau khi xếp mọi lượt báo theo thời gian, cách nhau không quá 2 phút.
"""
import pandas as pd

from . import config as C
from . import data as D


def add_log(log, code, rule, count, total, action):
    """Ghi một dòng vào nhật ký làm sạch."""
    log.append({"ma": code, "quy_tac": rule, "so_bi_anh_huong": int(count), "tong": int(total),
                "ty_le": round(count / total, 4) if total else 0.0, "cach_xu_ly": action})


def order_reports(rep):
    """Xếp mọi lượt báo theo thời gian. Thêm cột:
    t (thời điểm), dt (số giây từ lượt liền trước), brk (True nếu cách lượt trước quá 2 phút)."""
    rep = rep.copy()
    rep["t"] = pd.to_datetime(rep["updated_at"], utc=True, format="ISO8601")
    rep = rep.sort_values(["t", "report_id"], kind="stable").reset_index(drop=True)
    rep["dt"] = rep["t"].diff().dt.total_seconds()
    rep["brk"] = rep["dt"].isna() | (rep["dt"] > C.SESSION_GAP_S)
    return rep


def chain_id(rep, column):
    """Đánh số chuỗi: các lượt liền nhau có cùng giá trị ở cột `column` thuộc cùng một chuỗi.
    Ví dụ cột segment_id = 5, 5, 5, 8, 8, 5  ->  số chuỗi = 1, 1, 1, 2, 2, 3.
    Lượt cách lượt trước quá 2 phút (brk = True) luôn mở chuỗi mới."""
    new_chain = (rep[column] != rep[column].shift()) | rep["brk"]
    return new_chain.cumsum()


def flag_stationary(rep, seg_length, max_speed=C.STATIONARY_MAX_KMH, ratio=C.STATIONARY_RATIO):
    """QT5. rep đã qua order_reports. Trả về True cho các lượt thuộc chuỗi thiết bị đứng yên."""
    chain = chain_id(rep, "segment_id")              # chuỗi = các lượt liền nhau trên cùng một đoạn
    n = rep.groupby(chain)["velocity"].transform("size")
    vmax = rep.groupby(chain)["velocity"].transform("max")
    vmean = rep.groupby(chain)["velocity"].transform("mean")
    t_start = rep.groupby(chain)["t"].transform("min")
    t_end = rep.groupby(chain)["t"].transform("max")
    seconds = (t_end - t_start).dt.total_seconds()
    distance = vmean / 3.6 * seconds                 # quãng đường (m) nếu thiết bị chạy đúng tốc độ báo
    length = rep["segment_id"].map(seg_length).fillna(0).clip(lower=1)
    return (n >= 2) & (vmax <= max_speed) & (distance >= ratio * length)


def flag_frozen(rep, min_run=C.FROZEN_MIN_RUN):
    """QT6. rep đã qua order_reports. True cho các lượt thuộc chuỗi >= min_run lượt cùng một giá trị."""
    chain = chain_id(rep, "velocity")                # chuỗi = các lượt liền nhau cùng một tốc độ
    n = rep.groupby(chain)["velocity"].transform("size")
    return (n >= min_run) & (rep["velocity"] != 1)


def flag_urban_too_fast(rep, seg_type, limit=C.URBAN_MAX_KMH):
    """QT4. True cho các lượt >= limit km/h trên đường không phải cao tốc, quốc lộ."""
    highway = rep["segment_id"].map(seg_type).isin(C.HIGHWAY_TYPES)
    return (rep["velocity"] >= limit) & ~highway


def clean_reports(reports, version=None, segments=None,
                  stationary_max_speed=C.STATIONARY_MAX_KMH, stationary_ratio=C.STATIONARY_RATIO,
                  frozen_min_run=C.FROZEN_MIN_RUN, urban_limit=C.URBAN_MAX_KMH, max_speed=C.V2_MAX_KMH):
    """Làm sạch ở mức lượt báo. Trả về (bảng lượt báo đã làm sạch, nhật ký).
    Các ngưỡng để mặc định; E06 đổi ngưỡng để xem độ nhạy."""
    version = version or C.VERSION
    log = []
    total = len(reports)

    # QT1: bỏ lượt trùng lặp (cùng đoạn, cùng thời điểm, cùng tốc độ)
    dup = reports.duplicated(["segment_id", "updated_at", "velocity"])
    add_log(log, "QT1", "Lượt báo trùng lặp", dup.sum(), total, "bỏ")
    rep = reports[~dup]

    if version == "v1":
        add_log(log, "QT7", "Lượt báo 0 km/h", (rep["velocity"] == 0).sum(), len(rep), "giữ nguyên")
        add_log(log, "QT8", "Lượt báo đúng 1 km/h", (rep["velocity"] == 1).sum(), len(rep), "giữ nguyên")
        over = rep["velocity"] > C.V1_CAP_KMH
        add_log(log, "QT3", f"Lượt báo trên {C.V1_CAP_KMH} km/h", over.sum(), len(rep), f"cắt về {C.V1_CAP_KMH} km/h")
        rep = rep.copy()
        rep["velocity_raw"] = rep["velocity"]                  # giữ số gốc để đối chiếu
        rep["velocity"] = rep["velocity"].clip(upper=C.V1_CAP_KMH)
        return rep, log

    # v2: đánh dấu từng quy tắc trên toàn bộ lượt báo, rồi bỏ lần lượt theo thứ tự
    if segments is None:
        segments = D.load_segments()
    seg = segments.set_index("segment_id")
    rep = order_reports(rep)
    day = rep["ts_vn"].dt.strftime("%Y-%m-%d")
    rules = [
        ("QT2", f"Ngày {C.DAY_BULK} (tốc độ sinh ngẫu nhiên)", day == C.DAY_BULK),
        ("QT3", f"Lượt báo trên {max_speed} km/h", rep["velocity"] > max_speed),
        ("QT4", f"Lượt báo từ {urban_limit} km/h trên đường không phải cao tốc, quốc lộ",
         flag_urban_too_fast(rep, seg["street_type"], urban_limit)),
        ("QT5", "Thiết bị đứng yên", flag_stationary(rep, seg["length"], stationary_max_speed, stationary_ratio)),
        ("QT6", f"Tốc độ đóng băng (từ {frozen_min_run} lượt liền nhau cùng giá trị)", flag_frozen(rep, frozen_min_run)),
    ]
    removed = pd.Series(False, index=rep.index)
    for code, rule, flag in rules:
        new = flag & ~removed                 # chỉ đếm các lượt chưa bị quy tắc trước bỏ
        add_log(log, code, rule, new.sum(), len(rep), "bỏ")
        removed = removed | flag
    rep = rep[~removed].drop(columns=["t", "dt", "brk"]).reset_index(drop=True)

    add_log(log, "QT7", "Lượt báo 0 km/h", (rep["velocity"] == 0).sum(), len(rep), "giữ nguyên")
    add_log(log, "QT8", "Lượt báo đúng 1 km/h", (rep["velocity"] == 1).sum(), len(rep), "giữ nguyên")
    return rep, log


def clean_samples(samples, log, version=None):
    """Làm sạch ở mức mẫu (sau khi gộp theo khung 30 phút)."""
    version = version or C.VERSION
    df = samples.copy()
    if version == "v1":
        df["is_bulk_day"] = (df["date_vn"] == C.DAY_BULK).astype(int)
        add_log(log, "QT2", f"Ngày {C.DAY_BULK}", df["is_bulk_day"].sum(), len(df), "giữ, đánh dấu cột is_bulk_day")
    else:
        empty = df["speed"].isna()
        add_log(log, "QT9", "Mẫu không còn lượt báo hợp lệ", empty.sum(), len(df), "bỏ mẫu")
        df = df[~empty].reset_index(drop=True)
        df["is_bulk_day"] = 0                       # ngày 03/08 đã bị bỏ
    per_day = df.groupby("date_vn")["segment_id"].transform("size")
    add_log(log, "QT10", "Ngày có dưới 10 mẫu", (per_day < 10).sum(), len(df), "giữ nguyên")
    return df
