"""
evaluate.py - Chia dữ liệu, chỉ số đánh giá và kiểm định chéo.

Cách chia (chia_du_lieu.py tạo ra data/processed/<v>/chia_du_lieu.csv):
- Chia theo NGÀY, để các mẫu cùng ngày không nằm ở cả hai phía.
- 30 ngày làm tập test cố định, chỉ dùng một lần ở kiem_tra_test.py.
- Các ngày còn lại là tập huấn luyện, chia thành 4 phần cho kiểm định chéo 4 vòng.
"""
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from . import config as C, data as D, features as F

SPLIT_FILE = C.DATA_PROCESSED / "chia_du_lieu.csv"


# ---------------------------------------------------------------- Chia dữ liệu
def split_days(dates, n_parts):
    """Chia các ngày thành n_parts phần có số mẫu gần bằng nhau.
    Xét ngày nhiều mẫu trước, mỗi ngày đưa vào phần đang có ít mẫu nhất. Trả về {ngày: số phần}."""
    counts = dates.value_counts()
    # xếp ngày theo số mẫu giảm dần; ngày bằng số mẫu thì ngày sớm hơn đứng trước
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    load = [0] * n_parts            # số mẫu hiện có của mỗi phần
    part_of = {}
    for day in counts.index:
        k = load.index(min(load))   # phần đang ít mẫu nhất
        part_of[day] = k
        load[k] = load[k] + counts[day]
    return part_of


def add_split(df, version=None):
    """Thêm hai cột vào bảng mẫu: tap ("huan_luyen" / "test") và phan (0–3, -1 nếu là test)."""
    path = C.DATA_PROCESSED / "chia_du_lieu.csv"
    if version is not None:
        path = C.ROOT / "data" / "processed" / version / "chia_du_lieu.csv"
    split = pd.read_csv(path, encoding="utf-8-sig").set_index("date_vn")
    df = df.copy()
    df["tap"] = df["date_vn"].map(split["tap"])
    df["phan"] = df["date_vn"].map(split["phan"])
    return df


def load_samples(version=None):
    """Bảng mẫu đã làm sạch kèm cách chia. version=None: phiên bản đang chọn (LAM_SACH)."""
    path = C.DATA_PROCESSED / "samples.csv"
    if version is not None:
        path = C.ROOT / "data" / "processed" / version / "samples.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["dt_vn"] = pd.to_datetime(df["dt_vn"], format="%Y-%m-%d %H:%M")
    return add_split(df, version)


def train_part(df):
    """Chỉ lấy tập huấn luyện (đánh lại chỉ số 0..n-1). Trả về (bảng, mảng số phần của từng mẫu)."""
    train = df[df["tap"] == "huan_luyen"].reset_index(drop=True)
    return train, train["phan"].to_numpy()


def inner_day_groups(dates):
    """Chia các ngày của một tập huấn luyện thành 4 phần (dùng khi tính nhóm C out-of-fold)."""
    part_of = split_days(dates, 4)
    return dates.map(part_of).to_numpy()


def random_folds(n_samples, seed=C.SEED):
    """Đối chứng: chia ngẫu nhiên từng mẫu thành 4 phần (bỏ qua ngày)."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n_samples) % 4)


# ---------------------------------------------------------------- Chỉ số
def metrics(y_true, y_pred, max_velocity, los_label):
    """Các chỉ số cho một tập kiểm tra.
    MAE, RMSE, R2: sai số tốc độ. acc, F1: mức LOS quy từ tốc độ dự báo so với nhãn gốc
    (_nhan_goc) và so với LOS quy từ tốc độ thật (_tu_toc_do)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    los_pred = D.speed_to_los(y_pred, max_velocity)
    los_true_speed = D.speed_to_los(y_true, max_velocity)
    los_label = np.asarray(los_label, dtype=object)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "acc_LOS_nhan_goc": accuracy_score(los_label, los_pred),
        "F1_LOS_nhan_goc": f1_score(los_label, los_pred, average="macro", labels=C.LOS_LEVELS, zero_division=0),
        "acc_LOS_tu_toc_do": accuracy_score(los_true_speed, los_pred),
        "F1_LOS_tu_toc_do": f1_score(los_true_speed, los_pred, average="macro", labels=C.LOS_LEVELS,
                                     zero_division=0),
    }


# ---------------------------------------------------------------- Đặc trưng cho một lần chia
def make_features(df, base, train_idx, test_idx, inner_groups):
    """Bảng đặc trưng cho phần huấn luyện và phần kiểm tra của một lần chia.
    base: nhóm A + B đã tính sẵn cho cả df. Nhóm C tính riêng cho lần chia này."""
    y = df["speed"]
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    hist_train = F.history_features(train, y.iloc[train_idx], inner_groups=inner_groups)
    hist_test = F.history_features(train, y.iloc[train_idx], df_apply=test)
    X_train = pd.concat([base.iloc[train_idx], hist_train], axis=1)
    X_test = pd.concat([base.iloc[test_idx], hist_test], axis=1)
    return X_train, X_test


def features_from_train(train_df, apply_df):
    """Đặc trưng của apply_df cho một mô hình học từ train_df (nhóm C chỉ học từ train_df)."""
    hist = F.history_features(train_df, train_df["speed"], df_apply=apply_df)
    return pd.concat([F.base_features(apply_df), hist], axis=1)


def train_test_features(df):
    """Đặc trưng cho bước huấn luyện cuối (toàn tập huấn luyện) và bước test.
    Trả về X_train, y_train, X_test, test (bảng mẫu của tập test)."""
    base = F.base_features(df)
    train_idx = np.where(df["tap"] == "huan_luyen")[0]
    test_idx = np.where(df["tap"] == "test")[0]
    inner = df["phan"].to_numpy()[train_idx]          # nhóm C của tập huấn luyện: out-of-fold theo 4 phần
    X_train, X_test = make_features(df, base, train_idx, test_idx, inner)
    return X_train, df["speed"].iloc[train_idx], X_test, df.iloc[test_idx]


# ---------------------------------------------------------------- Kiểm định chéo
def cross_validate(model, df, folds, name="", columns=None, verbose=True):
    """Kiểm định chéo: mỗi vòng huấn luyện trên các phần còn lại, đánh giá trên một phần.
    model: một mô hình chưa huấn luyện (mỗi vòng dùng một bản sao mới).
    columns: chỉ dùng các đặc trưng này (None = dùng tất cả).
    Trả về (bảng chỉ số từng vòng, dự báo out-of-fold cho mọi mẫu)."""
    y = df["speed"]
    base = F.base_features(df)
    oof = np.full(len(df), np.nan)
    rows = []
    for k in np.unique(folds):
        train_idx = np.where(folds != k)[0]
        test_idx = np.where(folds == k)[0]
        inner = inner_day_groups(df["date_vn"].iloc[train_idx])
        X_train, X_test = make_features(df, base, train_idx, test_idx, inner)
        if columns is not None:
            X_train, X_test = X_train[columns], X_test[columns]
        m = copy.deepcopy(model)                              # bản sao mới, chưa huấn luyện
        m.fit(X_train, y.iloc[train_idx])
        pred = np.clip(m.predict(X_test), 0, C.SPEED_CAP_KMH)  # tốc độ không âm, không vượt trần
        oof[test_idx] = pred
        scores = metrics(y.iloc[test_idx], pred, df["max_velocity"].iloc[test_idx], df["LOS"].iloc[test_idx])
        row = {"mo_hinh": name, "phan": int(k), "so_mau_kiem_tra": len(test_idx)}
        for key in scores:
            row[key] = round(float(scores[key]), 4)
        rows.append(row)
        if verbose:
            print(f"   {name:<34} phần {k}: MAE {scores['MAE']:.2f}  R2 {scores['R2']:.3f}")
    return pd.DataFrame(rows), oof


METRIC_NAMES = ["MAE", "RMSE", "R2", "acc_LOS_nhan_goc", "F1_LOS_nhan_goc", "acc_LOS_tu_toc_do", "F1_LOS_tu_toc_do"]


def summarize(per_fold):
    """Trung bình (_tb) và độ lệch chuẩn (_dlc) qua các vòng, cho từng mô hình."""
    groups = per_fold.groupby("mo_hinh", sort=False)[METRIC_NAMES]
    mean = groups.mean().round(4)
    std = groups.std(ddof=0).round(4)
    out = pd.DataFrame({"mo_hinh": mean.index})
    for col in METRIC_NAMES:
        out[col + "_tb"] = mean[col].to_numpy()
    for col in METRIC_NAMES:
        out[col + "_dlc"] = std[col].to_numpy()
    return out
