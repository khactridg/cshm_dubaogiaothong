"""
models.py - Hai mô hình mốc và ba mô hình học máy.

Mỗi hàm make_...() trả về một mô hình MỚI, chưa huấn luyện, có hai hàm fit(X, y) và predict(X).
X là bảng đặc trưng tạo bởi features.py.
"""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from . import config as C

# Các cột số của nhóm B và nhóm C
ROAD_NUM = ["street_level", "max_velocity", "max_velocity_missing", "log_length", "lat", "lon", "n_in", "n_out"]
HIST_NUM = ["hist_segment", "hist_segment_daypart", "hist_street", "hist_segment_log_n"]


# ---------------------------------------------------------------- Mô hình mốc
def make_mean():
    """Mốc 1: luôn đoán tốc độ trung bình của tập huấn luyện."""
    return DummyRegressor(strategy="mean")


class HistoryLookup:
    """Mốc 2: đoán bằng tốc độ trung bình trong quá khứ của chính đoạn, trong cùng buổi.
    Không học gì thêm: dự báo chính là cột hist_segment_daypart."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["hist_segment_daypart"].to_numpy()


def make_history_lookup():
    return HistoryLookup()


# ---------------------------------------------------------------- Hồi quy tuyến tính
def make_linear(hour_as_category=True, extra_num=(), huber=False, huber_epsilon=1.35):
    """Hồi quy tuyến tính.
    hour_as_category=True: giờ được mã hóa one-hot (24 cột); False: giờ là một số 0–24.
    extra_num: các cột số thêm vào (dùng ở phần cải tiến).
    huber=True: dùng hàm mất mát Huber thay cho bình phương."""
    categorical = ["street_type", "day_part", "weekday"]
    numeric = ["is_weekend"] + ROAD_NUM + HIST_NUM + list(extra_num)
    if hour_as_category:
        categorical = categorical + ["hour_int"]
    else:
        numeric = ["is_weekend", "hour"] + ROAD_NUM + HIST_NUM + list(extra_num)
    preprocess = ColumnTransformer([
        ("phan_loai", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical),
        ("so", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),   # điền thiếu, chuẩn hóa
    ])
    if huber:
        regressor = HuberRegressor(epsilon=huber_epsilon, alpha=0.0, max_iter=2000)
    else:
        regressor = LinearRegression()
    return Pipeline([("tien_xu_ly", preprocess), ("mo_hinh", regressor)])


# ---------------------------------------------------------------- Mô hình cây
TREE_CAT = ["street_type", "day_part"]
TREE_NUM = ["hour", "weekday", "is_weekend"] + ROAD_NUM + HIST_NUM


def tree_preprocess(extra_num=()):
    """Mô hình cây: đổi cột chữ thành số thứ tự, giữ nguyên cột số (cây tự xử lý giá trị thiếu)."""
    return ColumnTransformer([
        ("phan_loai", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), TREE_CAT),
        ("so", "passthrough", TREE_NUM + list(extra_num)),
    ])


def make_random_forest(n_estimators=300, min_samples_leaf=5, max_features=0.5, max_depth=None,
                       extra_num=(), criterion="squared_error"):
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                               max_features=max_features, max_depth=max_depth, criterion=criterion,
                               n_jobs=-1, random_state=C.SEED)
    return Pipeline([("tien_xu_ly", tree_preprocess(extra_num)), ("mo_hinh", rf)])


def make_xgboost(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=5, extra_num=(), objective="reg:squarederror", huber_slope=1.0):
    """objective="reg:pseudohubererror" dùng hàm mất mát pseudo-Huber với độ dốc huber_slope."""
    if objective == "reg:pseudohubererror":
        xgb = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                           subsample=subsample, colsample_bytree=colsample_bytree,
                           min_child_weight=min_child_weight, tree_method="hist", objective=objective,
                           huber_slope=huber_slope, n_jobs=-1, random_state=C.SEED)
    else:
        xgb = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                           subsample=subsample, colsample_bytree=colsample_bytree,
                           min_child_weight=min_child_weight, tree_method="hist", objective=objective,
                           n_jobs=-1, random_state=C.SEED)
    return Pipeline([("tien_xu_ly", tree_preprocess(extra_num)), ("mo_hinh", xgb)])
