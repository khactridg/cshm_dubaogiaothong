"""Bước 7. Thêm thông tin giao thông vừa xảy ra gần đó và lượng mưa vào đặc trưng, xem MAE có giảm không.
Ba đặc trưng mới, chỉ dùng thông tin có trước lúc bắt đầu khung 30 phút cần dự báo:
  xung_quanh  độ lệch "tốc độ thật - mức bình thường" trung bình của các mẫu cách không quá 1 km, thuộc 2 khung ngay trước, cùng ngày.
  thanh_pho   như trên cho mọi mẫu ở bất kỳ đâu trong 4 khung trước (2 giờ).
  mua_mm      lượng mưa của giờ trước tại điểm lưới gần nhất (data/raw/thoi_tiet_luoi.csv, nguồn Open-Meteo).
"Mức bình thường" là đặc trưng hist_segment_daypart. Không có mẫu lân cận thì đặc trưng bằng 0.
Đầu vào:  data/sach, data/raw/thoi_tiet_luoi.csv, ket_qua/b4_tham_so_tot_nhat_v2.json
Đầu ra:   ket_qua/b7_ket_qua.csv, b7_nhom_co_lan_can.csv, b7_mua_va_toc_do.csv, b7_mae_theo_cau_hinh.png"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src import cau_hinh as CH, dac_trung as DT, doc_du_lieu as DL, mo_hinh as MH

warnings.filterwarnings("ignore", message="Found unknown categories")
CH.tao_thu_muc()
CH.thiet_lap_hinh()
BAN_KINH_M = 1000
KHUNG_GAN = 2          # 2 khung = 60 phút trước
KHUNG_THANH_PHO = 4    # 4 khung = 2 giờ trước
R_TRAI_DAT = 6_371_000
CAU_HINH = {"Nền": [], "Nền + xung quanh": ["xung_quanh"], "Nền + thành phố": ["thanh_pho"], "Nền + mưa": ["mua_mm"],
            "Nền + cả ba": ["xung_quanh", "thanh_pho", "mua_mm"]}
MO_HINH = ["Hồi quy tuyến tính", "Random Forest", "XGBoost"]

# 1. Dữ liệu mưa theo lưới điểm
mua = pd.read_csv(CH.DU_LIEU_GOC / CH.TEP_GOC["mua"])
mua["diem"] = mua["lat"].round(2).astype(str) + "," + mua["lon"].round(2).astype(str)
bang_mua = mua.pivot_table(index=pd.to_datetime(mua["gio_vn"]), columns="diem", values="mua_mm", aggfunc="first")  # hàng = giờ, cột = điểm
diem_luoi = mua.drop_duplicates("diem").set_index("diem").loc[bang_mua.columns, ["lat", "lon"]]
cay_luoi = BallTree(np.radians(diem_luoi[["lat", "lon"]].to_numpy()), metric="haversine")


def luong_mua(df):
    """Mưa của giờ trước khung tại điểm lưới gần nhất: khung 17:30 lấy mốc 17:00 (mưa từ 16:00 đến 17:00)."""
    _, gan = cay_luoi.query(np.radians(df[["lat", "lon"]].to_numpy()), k=1)
    cot = bang_mua.columns.get_indexer(diem_luoi.index[gan[:, 0]])
    hang = bang_mua.index.get_indexer(df["dt_vn"].dt.floor("h"))
    gia_tri = bang_mua.to_numpy()[np.maximum(hang, 0), cot]
    gia_tri[hang < 0] = 0
    return pd.Series(np.nan_to_num(gia_tri), index=df.index, name="mua_mm")


def thong_tin_vua_xay_ra(df, binh_thuong):
    """xung_quanh và thanh_pho cho từng mẫu của df, chỉ dùng mẫu của các khung trước, cùng ngày."""
    lech = (df["speed"] - binh_thuong).to_numpy()                       # tốc độ thật lệch so với mức bình thường
    khung = (df["hour_vn"] * 2).astype(int).to_numpy()
    toa_do = np.radians(df[["lat", "lon"]].to_numpy())
    gan = np.zeros(len(df))
    tp = np.zeros(len(df))
    for _, idx in df.groupby("date_vn").indices.items():                # idx: vị trí các mẫu của một ngày
        s = khung[idx]
        cay = BallTree(toa_do[idx], metric="haversine")
        lan_can = cay.query_radius(toa_do[idx], r=BAN_KINH_M / R_TRAI_DAT)
        for k in range(len(idx)):
            nb = lan_can[k]
            nb = nb[(s[nb] < s[k]) & (s[nb] >= s[k] - KHUNG_GAN)]
            if len(nb) > 0:
                gan[idx[k]] = lech[idx[nb]].mean()
            truoc = (s < s[k]) & (s >= s[k] - KHUNG_THANH_PHO)
            if truoc.any():
                tp[idx[k]] = lech[idx[truoc]].mean()
    return pd.DataFrame({"xung_quanh": gan, "thanh_pho": tp}, index=df.index)


def dac_trung_moi(df_hl, df_ap, nhom_trong):
    """Ba đặc trưng mới cho phần huấn luyện và phần chấm của một lần chia."""
    bt_hl = DT.dac_trung_lich_su(df_hl, df_hl["speed"], nhom_trong=nhom_trong)["hist_segment_daypart"]
    bt_ap = DT.dac_trung_lich_su(df_hl, df_hl["speed"], df_ap=df_ap)["hist_segment_daypart"]
    khoi_hl = thong_tin_vua_xay_ra(df_hl, bt_hl)
    khoi_hl["mua_mm"] = luong_mua(df_hl)
    khoi_ap = thong_tin_vua_xay_ra(df_ap, bt_ap)
    khoi_ap["mua_mm"] = luong_mua(df_ap)
    return khoi_hl, khoi_ap


def tao_mo_hinh(ten, cot_them, tham_so):
    if ten == "Hồi quy tuyến tính":
        return MH.tao_hoi_quy_tuyen_tinh(cot_them=cot_them)
    if ten == "Random Forest":
        return MH.tao_random_forest(**tham_so[ten]["tham_so"], cot_them=cot_them)
    return MH.tao_xgboost(**tham_so[ten]["tham_so"], cot_them=cot_them)


# 2. Đặc trưng cho tập huấn luyện và tập kiểm tra
df = DL.doc_mau("v2")
hl = df[df["tap"] == "huan_luyen"].reset_index(drop=True)
kt = df[df["tap"] == "test"].reset_index(drop=True)
X_hl = DT.dac_trung_huan_luyen(hl)
X_kt = DT.dac_trung_cho(hl, kt)
them_hl, them_kt = dac_trung_moi(hl, kt, hl["phan"].to_numpy())
X_hl = pd.concat([X_hl, them_hl], axis=1)
X_kt = pd.concat([X_kt, them_kt], axis=1)
tham_so = MH.doc_tham_so_tot_nhat("v2")
print(f"Huấn luyện {len(hl)} mẫu, kiểm tra {len(kt)} mẫu")

# 3. Năm cấu hình x ba mô hình
du_bao = {}
for ten in MO_HINH:
    for ch, cot in CAU_HINH.items():
        m = tao_mo_hinh(ten, cot, tham_so)
        m.fit(X_hl, hl["speed"])
        du_bao[(ten, ch)] = np.clip(m.predict(X_kt), 0, CH.TRAN_DU_BAO["v2"])
    print("  xong", ten)
y = kt["speed"].to_numpy()
co_lan_can = (them_kt["xung_quanh"] != 0).to_numpy()
dong, dong_nhom = [], []
for (ten, ch), p in du_bao.items():
    dong.append({"mo_hinh": ten, "cau_hinh": ch, "so_mau": len(y), "MAE": round(float(np.abs(p - y).mean()), 3)})
    dong_nhom.append({"mo_hinh": ten, "cau_hinh": ch, "so_mau": int(co_lan_can.sum()), "so_ngay": int(kt.loc[co_lan_can, "date_vn"].nunique()),
                      "MAE": round(float(np.abs(p - y)[co_lan_can].mean()), 3)})
ket_qua = pd.DataFrame(dong)
nhom = pd.DataFrame(dong_nhom)
ket_qua.to_csv(CH.KET_QUA / "b7_ket_qua.csv", index=False, encoding="utf-8-sig")
nhom.to_csv(CH.KET_QUA / "b7_nhom_co_lan_can.csv", index=False, encoding="utf-8-sig")
print(ket_qua.pivot(index="cau_hinh", columns="mo_hinh", values="MAE").loc[list(CAU_HINH)].to_string())
print(f"\nMẫu kiểm tra có số liệu lân cận trong 1 km: {co_lan_can.sum()} ({co_lan_can.mean():.1%})")
print(nhom.pivot(index="cau_hinh", columns="mo_hinh", values="MAE").loc[list(CAU_HINH)].to_string())

# 4. Mưa và độ lệch tốc độ (trên tập huấn luyện)
lech = hl["speed"] - DT.dac_trung_lich_su(hl, hl["speed"], nhom_trong=hl["phan"].to_numpy())["hist_segment_daypart"]
muc = pd.cut(luong_mua(hl), [-0.01, 0.0999, 0.5, 2, 5, 1000], labels=["không mưa", "0,1-0,5 mm", "0,5-2 mm", "2-5 mm", "từ 5 mm"])
theo_mua = pd.DataFrame({"muc": muc, "lech": lech, "ngay": hl["date_vn"]}).groupby("muc", observed=False).agg(
    so_mau=("lech", "size"), so_ngay=("ngay", "nunique"), lech_so_voi_binh_thuong=("lech", "mean")).round(2).reset_index()
theo_mua.to_csv(CH.KET_QUA / "b7_mua_va_toc_do.csv", index=False, encoding="utf-8-sig")
print(theo_mua.to_string(index=False))

# 5. Hình
fig, ax = plt.subplots(figsize=(10, 4.6))
cau_hinh = list(CAU_HINH)
rong = 0.8 / len(cau_hinh)
mau_cot = [CH.XAM, "#5E9BC9", CH.XANH_NHAT, "#B9C4CC", CH.XANH]
vi_tri = np.arange(len(MO_HINH))
for j, ch in enumerate(cau_hinh):
    gia_tri = [ket_qua[(ket_qua["mo_hinh"] == t) & (ket_qua["cau_hinh"] == ch)]["MAE"].iloc[0] for t in MO_HINH]
    xs = vi_tri - 0.4 + rong * (j + 0.5)
    ax.bar(xs, gia_tri, width=rong * 0.92, color=mau_cot[j], label=ch)
    for x, v in zip(xs, gia_tri):
        ax.text(x, v + 0.05, CH.so_vn(v), ha="center", fontsize=8.5)
ax.set_xticks(vi_tri)
ax.set_xticklabels(MO_HINH)
ax.set_ylabel("MAE trên tập kiểm tra (km/h)")
ax.set_title("MAE khi thêm thông tin giao thông gần nhất và lượng mưa")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=9.5, frameon=False)
CH.luu_hinh(fig, "b7_mae_theo_cau_hinh.png")
