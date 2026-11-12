"""
features.py - Tạo đặc trưng cho mô hình.

Nhóm A - thời gian:   giờ, thứ, cuối tuần, buổi trong ngày
Nhóm B - đoạn đường:  loại, cấp, tốc độ tối đa, chiều dài, vị trí, số đoạn nối vào / nối ra
Nhóm C - lịch sử:     tốc độ trung bình trong quá khứ của chính đoạn / tuyến đường

Nhóm A, B chỉ dùng thông tin biết trước (lịch, bản đồ).
Nhóm C dùng đáp án (tốc độ) nên chỉ được tính từ tập huấn luyện. Trên chính tập huấn luyện, mỗi mẫu
được tính từ các ngày KHÁC (cách "out-of-fold"), để mô hình không nhìn thấy đáp án của chính nó.
"""
import numpy as np
import pandas as pd

from . import data as D

# Các loại đường chính; loại khác (rất hiếm) gộp thành "khac"
MAIN_STREET_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                     "residential", "motorway_link", "trunk_link", "primary_link", "secondary_link",
                     "tertiary_link"]
SMOOTHING = 5     # số "mẫu ảo": đoạn ít mẫu bị kéo về trung bình chung


def day_part(hour):
    """Buổi trong ngày từ giờ (giờ Việt Nam)."""
    if 6 <= hour < 9:
        return "cao_diem_sang"
    if 9 <= hour < 16:
        return "ban_ngay"
    if 16 <= hour < 19:
        return "cao_diem_chieu"
    if 19 <= hour < 22:
        return "toi"
    return "dem"


def time_features(df):
    """Nhóm A."""
    h = df["hour_vn"]
    X = pd.DataFrame(index=df.index)
    X["hour"] = h                                      # 7.5 = 7 giờ 30
    X["hour_int"] = h.astype(int)                      # giờ nguyên, hồi quy tuyến tính dùng làm biến phân loại
    X["hour_sin"] = np.sin(2 * np.pi * h / 24)
    X["hour_cos"] = np.cos(2 * np.pi * h / 24)
    X["weekday"] = df["weekday"]
    X["is_weekend"] = df["is_weekend"]
    X["day_part"] = h.apply(day_part)
    return X


def road_features(df):
    """Nhóm B."""
    seg = D.load_segments()
    n_into = seg.groupby("e_node_id").size()           # số đoạn đi vào mỗi nút
    n_out = seg.groupby("s_node_id").size()            # số đoạn đi ra từ mỗi nút
    X = pd.DataFrame(index=df.index)
    X["street_type"] = df["street_type"].where(df["street_type"].isin(MAIN_STREET_TYPES), "khac")
    X["street_level"] = df["street_level"]
    X["max_velocity"] = df["max_velocity"]             # thiếu ở khoảng 85% mẫu, để trống
    X["max_velocity_missing"] = df["max_velocity"].isna().astype(int)
    X["log_length"] = np.log1p(df["length"])
    X["lat"] = df["lat"]
    X["lon"] = df["lon"]
    X["n_in"] = df["s_node_id"].map(n_into).fillna(0).astype(int)    # số đoạn đổ vào đoạn này
    X["n_out"] = df["e_node_id"].map(n_out).fillna(0).astype(int)    # số đoạn đi tiếp từ đoạn này
    return X


def base_features(df):
    """Nhóm A + B (không dùng đáp án)."""
    return pd.concat([time_features(df), road_features(df)], axis=1)


def smoothed_mean(fit_frame, y_fit, apply_frame, keys):
    """Trung bình tốc độ theo nhóm `keys` (ví dụ theo đoạn), có co rút:
        giá trị = (tổng tốc độ của nhóm + SMOOTHING x trung bình chung) / (số mẫu của nhóm + SMOOTHING)
    Học từ fit_frame / y_fit, áp lên apply_frame. Trả về (giá trị, số mẫu của nhóm)."""
    table = fit_frame[keys].copy()
    table["y"] = y_fit.to_numpy()
    stats = table.groupby(keys)["y"].agg(["sum", "count"]).reset_index()
    prior = y_fit.mean()
    merged = apply_frame[keys].merge(stats, on=keys, how="left")
    total = merged["sum"].fillna(0).to_numpy()
    count = merged["count"].fillna(0).to_numpy()
    return (total + SMOOTHING * prior) / (count + SMOOTHING), count


def history_block(fit_frame, y_fit, apply_frame):
    """Bốn đặc trưng nhóm C cho apply_frame, học từ fit_frame."""
    out = pd.DataFrame(index=apply_frame.index)
    value, count = smoothed_mean(fit_frame, y_fit, apply_frame, ["segment_id"])
    out["hist_segment"] = value                        # trung bình của đoạn
    out["hist_segment_log_n"] = np.log1p(count)        # đoạn đã có bao nhiêu mẫu
    value, _ = smoothed_mean(fit_frame, y_fit, apply_frame, ["segment_id", "day_part"])
    out["hist_segment_daypart"] = value                # trung bình của đoạn trong cùng buổi
    value, _ = smoothed_mean(fit_frame, y_fit, apply_frame, ["street_id"])
    out["hist_street"] = value                         # trung bình của cả tuyến đường
    return out[["hist_segment", "hist_segment_daypart", "hist_street", "hist_segment_log_n"]]


def key_columns(df):
    """Các cột dùng làm khóa cho nhóm C."""
    keys = df[["segment_id", "street_id"]].copy()
    keys["day_part"] = df["hour_vn"].apply(day_part)
    return keys


def history_features(df_train, y_train, df_apply=None, inner_groups=None):
    """Nhóm C.
    - Có df_apply: học từ toàn bộ df_train rồi áp lên df_apply (tập kiểm tra).
    - Không có df_apply: tính cho chính df_train theo kiểu out-of-fold: chia df_train theo inner_groups
      (các phần chia theo ngày), mỗi phần được tính từ các phần còn lại."""
    train_keys = key_columns(df_train)
    if df_apply is not None:
        return history_block(train_keys, y_train, key_columns(df_apply))
    parts = []
    for g in np.unique(inner_groups):
        in_part = inner_groups == g
        parts.append(history_block(train_keys[~in_part], y_train[~in_part], train_keys[in_part]))
    return pd.concat(parts).loc[df_train.index]
