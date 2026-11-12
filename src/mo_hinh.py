"""Các mô hình, chỉ số đánh giá và kiểm định chéo.
Hai mốc: trung bình chung, trung bình lịch sử của đoạn. Bốn mô hình học máy: hồi quy tuyến tính, cây quyết định,
Random Forest, XGBoost. Mỗi hàm tao_...() trả về một mô hình mới, chưa huấn luyện, có fit(X, y) và predict(X)."""
import copy
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from . import cau_hinh as CH, dac_trung as DT, doc_du_lieu as DL

COT_DOAN = ["street_level", "max_velocity", "max_velocity_missing", "log_length", "lat", "lon", "n_in", "n_out"]
COT_LICH_SU = ["hist_segment", "hist_segment_daypart", "hist_street", "hist_segment_log_n"]
COT_CAY_CHU = ["street_type", "day_part"]
COT_CAY_SO = ["hour", "weekday", "is_weekend"] + COT_DOAN + COT_LICH_SU


# ---------------------------------------------------------------- Hai mốc
def tao_moc_trung_binh():
    """Luôn đoán tốc độ trung bình của tập huấn luyện."""
    return DummyRegressor(strategy="mean")


class MocLichSu:
    """Đoán bằng tốc độ trung bình của chính đoạn trong cùng buổi, tính từ tập huấn luyện (cột hist_segment_daypart)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["hist_segment_daypart"].to_numpy()


def tao_moc_lich_su():
    return MocLichSu()


# ---------------------------------------------------------------- Bốn mô hình học máy
def tao_hoi_quy_tuyen_tinh(cot_them=()):
    """Hồi quy tuyến tính. Cột chữ và giờ nguyên (24 giá trị) mã hóa one-hot; cột số điền thiếu bằng trung vị rồi chuẩn hóa."""
    chu = ["street_type", "day_part", "weekday", "hour_int"]
    so = ["is_weekend"] + COT_DOAN + COT_LICH_SU + list(cot_them)
    tien_xu_ly = ColumnTransformer([
        ("phan_loai", OneHotEncoder(handle_unknown="ignore", drop="first"), chu),
        ("so", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), so)])
    return Pipeline([("tien_xu_ly", tien_xu_ly), ("mo_hinh", LinearRegression())])


def tien_xu_ly_cay(cot_them=()):
    """Mô hình cây: cột chữ đổi thành số thứ tự, cột số giữ nguyên (cây tự xử lý giá trị thiếu)."""
    return ColumnTransformer([
        ("phan_loai", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), COT_CAY_CHU),
        ("so", "passthrough", COT_CAY_SO + list(cot_them))])


def tao_cay(max_depth=8, min_samples_leaf=20, cot_them=()):
    cay = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=CH.SEED)
    return Pipeline([("tien_xu_ly", tien_xu_ly_cay(cot_them)), ("mo_hinh", cay)])


def tao_random_forest(n_estimators=300, max_depth=None, min_samples_leaf=5, max_features=0.5, cot_them=()):
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                               max_features=max_features, n_jobs=-1, random_state=CH.SEED)
    return Pipeline([("tien_xu_ly", tien_xu_ly_cay(cot_them)), ("mo_hinh", rf)])


def tao_xgboost(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, cot_them=()):
    xgb = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample,
                       colsample_bytree=colsample_bytree, min_child_weight=min_child_weight, tree_method="hist",
                       objective="reg:squarederror", n_jobs=-1, random_state=CH.SEED)
    return Pipeline([("tien_xu_ly", tien_xu_ly_cay(cot_them)), ("mo_hinh", xgb)])


def tao_tu_tham_so(ten, p):
    """Tạo cây, Random Forest hoặc XGBoost từ một bộ siêu tham số (dict)."""
    if ten == "Cây quyết định":
        return tao_cay(**p)
    if ten == "Random Forest":
        return tao_random_forest(**p)
    return tao_xgboost(**p)


def doc_tham_so_tot_nhat(pb):
    """Bộ siêu tham số tốt nhất do b4 chọn cho phiên bản pb."""
    with open(CH.KET_QUA / f"b4_tham_so_tot_nhat_{pb}.json", encoding="utf-8") as f:
        return json.load(f)


def tao_bo_mo_hinh(pb):
    """Sáu mô hình của phiên bản pb (chưa huấn luyện), siêu tham số của cây, Random Forest, XGBoost lấy từ b4."""
    t = doc_tham_so_tot_nhat(pb)
    return {"Mốc: trung bình": tao_moc_trung_binh(), "Mốc: lịch sử đoạn": tao_moc_lich_su(),
            "Hồi quy tuyến tính": tao_hoi_quy_tuyen_tinh(), "Cây quyết định": tao_cay(**t["Cây quyết định"]["tham_so"]),
            "Random Forest": tao_random_forest(**t["Random Forest"]["tham_so"]),
            "XGBoost": tao_xgboost(**t["XGBoost"]["tham_so"])}


# ---------------------------------------------------------------- Chỉ số
def chi_so(y, du_bao, vmax):
    """Các chỉ số trên một tập kiểm tra. MAE, RMSE, R2, tỉ lệ sai không quá 5 km/h, sai số thời gian đi (phút/km);
    mức LOS quy từ tốc độ dự báo so với mức LOS quy từ tốc độ thật: tỉ lệ đúng, đúng hoặc lệch một mức, macro-F1."""
    y = np.asarray(y, dtype=float)
    du_bao = np.asarray(du_bao, dtype=float)
    los_db = DL.quy_ra_los(du_bao, vmax)
    los_that = DL.quy_ra_los(y, vmax)
    lech = np.abs(DL.bac_los(los_db) - DL.bac_los(los_that))
    phut_km = np.abs(60 / np.maximum(du_bao, 3) - 60 / np.maximum(y, 3)).mean()
    return {"MAE": mean_absolute_error(y, du_bao), "RMSE": float(np.sqrt(mean_squared_error(y, du_bao))),
            "R2": r2_score(y, du_bao), "trong_5kmh": float((np.abs(y - du_bao) <= 5).mean()), "phut_km": float(phut_km),
            "los_dung": float((lech == 0).mean()), "los_lech_1": float((lech <= 1).mean()),
            "macro_F1": f1_score(los_that.astype(str), los_db.astype(str), average="macro", labels=CH.MUC_LOS, zero_division=0)}


def chi_so_theo_tron(bang, nhom):
    """Trung bình (_tb) và độ lệch chuẩn (_dlc) của các chỉ số qua các vòng, theo cột `nhom` (thường là mo_hinh)."""
    cot = [c for c in bang.columns if c not in (nhom, "phan", "so_mau_kiem_tra")]
    tb = bang.groupby(nhom, sort=False)[cot].mean().add_suffix("_tb")
    dlc = bang.groupby(nhom, sort=False)[cot].std(ddof=0).add_suffix("_dlc")
    return pd.concat([tb, dlc], axis=1).round(4).reset_index()


# ---------------------------------------------------------------- Kiểm định chéo
def chia_ngau_nhien(so_mau):
    """Đối chứng: chia ngẫu nhiên từng mẫu thành 4 phần, bỏ qua ngày."""
    return np.random.default_rng(CH.SEED).permutation(np.arange(so_mau) % 4)


def kiem_dinh_cheo(mo_hinh, df, phan, ten="", tran=120, in_ra=True):
    """Mỗi vòng huấn luyện trên các phần còn lại, đánh giá trên một phần. Trả về (bảng chỉ số từng vòng, dự báo ngoài mẫu)."""
    nen = DT.dac_trung_nen(df)
    ngoai_mau = np.full(len(df), np.nan)
    dong = []
    for k in np.unique(phan):
        hl = np.where(phan != k)[0]
        kt = np.where(phan == k)[0]
        nhom_trong = df["date_vn"].iloc[hl].map(DL.chia_ngay(df["date_vn"].iloc[hl], 4)).to_numpy()
        X_hl, X_kt = DT.dac_trung_mot_lan_chia(df, nen, hl, kt, nhom_trong)
        m = copy.deepcopy(mo_hinh)
        m.fit(X_hl, df["speed"].iloc[hl])
        du_bao = np.clip(m.predict(X_kt), 0, tran)
        ngoai_mau[kt] = du_bao
        cs = chi_so(df["speed"].iloc[kt], du_bao, df["max_velocity"].iloc[kt])
        dong.append({"mo_hinh": ten, "phan": int(k), "so_mau_kiem_tra": len(kt), **{a: round(float(b), 4) for a, b in cs.items()}})
        if in_ra:
            print(f"   {ten:<24} phần {k}: MAE {cs['MAE']:.2f}  R2 {cs['R2']:.3f}")
    return pd.DataFrame(dong), ngoai_mau
