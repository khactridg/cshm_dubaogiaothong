"""Bước 3. Chia dữ liệu theo ngày thành tập huấn luyện và tập kiểm tra.
Đầu vào:  data/sach/mau_v1.csv, mau_v2.csv
Đầu ra:   data/sach/chia_du_lieu.csv (mỗi dòng một ngày của một phiên bản: tap, phan)
          ket_qua/b3_chia.csv, b3_dai_lich.png (các ngày thuộc phần nào), b3_chia_ngay_va_ngau_nhien.png
- v1: các ngày chia thành 5 phần có số mẫu gần bằng nhau. Phần 4 là tập kiểm tra (30 ngày),
  phần 0-3 là tập huấn luyện và cũng là 4 vòng của kiểm định chéo.
- v2: dùng đúng các ngày kiểm tra của v1, các ngày còn lại chia thành 4 phần."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import cau_hinh as CH, doc_du_lieu as DL

CH.tao_thu_muc()
CH.thiet_lap_hinh()

mau = {pb: DL.doc_mau(pb, kem_chia=False) for pb in CH.PHIEN_BAN}

# 1. Chia ngày
dong = []
phan_v1 = DL.chia_ngay(mau["v1"]["date_vn"], 5)
for n, k in phan_v1.items():
    dong.append({"phien_ban": "v1", "date_vn": n, "tap": "test" if k == 4 else "huan_luyen", "phan": -1 if k == 4 else k})
ngay_test = [n for n, k in phan_v1.items() if k == 4]
v2 = mau["v2"]
phan_v2 = DL.chia_ngay(v2.loc[~v2["date_vn"].isin(ngay_test), "date_vn"], 4)
for n, k in phan_v2.items():
    dong.append({"phien_ban": "v2", "date_vn": n, "tap": "huan_luyen", "phan": k})
for n in v2.loc[v2["date_vn"].isin(ngay_test), "date_vn"].unique():
    dong.append({"phien_ban": "v2", "date_vn": n, "tap": "test", "phan": -1})
chia = pd.DataFrame(dong).sort_values(["phien_ban", "date_vn"]).reset_index(drop=True)
chia.to_csv(CH.DU_LIEU_SACH / "chia_du_lieu.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

tom_tat = []
for pb in CH.PHIEN_BAN:
    df = DL.doc_mau(pb)
    for k in [0, 1, 2, 3, -1]:
        d = df[df["phan"] == k]
        tom_tat.append({"phien_ban": pb, "phan": "test" if k == -1 else k, "so_ngay": d["date_vn"].nunique(), "so_mau": len(d)})
tom_tat = pd.DataFrame(tom_tat)
tom_tat.to_csv(CH.KET_QUA / "b3_chia.csv", index=False, encoding="utf-8-sig")
print(tom_tat.to_string(index=False))

# 2. Dải lịch của v2: mỗi ô là một ngày
c2 = chia[chia["phien_ban"] == "v2"].set_index("date_vn")
ngay = pd.date_range("2020-07-01", "2021-12-31")
mau_phan = [CH.XANH, "#3E7CB1", CH.XANH_NHAT, "#C6DAEA"]
thang = sorted({(pd.Timestamp(n).year, pd.Timestamp(n).month) for n in c2.index})
fig, ax = plt.subplots(figsize=(11, 0.42 * len(thang) + 1.4))
for i, (nam, th) in enumerate(thang):
    for d in ngay[(ngay.year == nam) & (ngay.month == th)]:
        s = d.strftime("%Y-%m-%d")
        if s in c2.index:
            mau_o = CH.DO if c2.loc[s, "tap"] == "test" else mau_phan[int(c2.loc[s, "phan"])]
        else:
            mau_o = "#ECEFF2"
        ax.add_patch(plt.Rectangle((d.day - 1, -i), 0.9, 0.8, color=mau_o))
ax.set_xlim(0, 31)
ax.set_ylim(-len(thang) + 0.5, 1)
ax.set_yticks([-i + 0.4 for i in range(len(thang))])
ax.set_yticklabels([f"{th:02d}/{nam}" for nam, th in thang])
ax.set_xticks([d - 0.6 for d in range(1, 32, 2)])
ax.set_xticklabels(list(range(1, 32, 2)))
ax.set_xlabel("Ngày trong tháng")
for k in range(4):
    ax.bar(0, 0, color=mau_phan[k], label=f"Huấn luyện, phần {k}")
ax.bar(0, 0, color=CH.DO, label="Kiểm tra")
ax.bar(0, 0, color="#ECEFF2", label="Không có dữ liệu")
ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10, frameon=False)
ax.set_title("Các ngày có dữ liệu của v2 và phần mà mỗi ngày thuộc về")
CH.luu_hinh(fig, "b3_dai_lich.png")

# 3. Chia theo ngày so với chia ngẫu nhiên (kiểm định chéo 4 vòng, tham số mặc định, chỉ dùng tập huấn luyện của v2)
from src import mo_hinh as MH

df = DL.doc_mau("v2")
df = df[df["tap"] == "huan_luyen"].reset_index(drop=True)
mo_hinh = {"Hồi quy tuyến tính": MH.tao_hoi_quy_tuyen_tinh(), "Random Forest": MH.tao_random_forest(), "XGBoost": MH.tao_xgboost()}
cach_chia = {"Chia theo ngày": df["phan"].to_numpy(), "Chia ngẫu nhiên từng mẫu": MH.chia_ngau_nhien(len(df))}
ket_qua = []
for ten_chia, phan in cach_chia.items():
    print(ten_chia)
    for ten, m in mo_hinh.items():
        bang, _ = MH.kiem_dinh_cheo(m, df, phan, ten=ten)
        ket_qua.append({"cach_chia": ten_chia, "mo_hinh": ten, "MAE_tb": round(bang["MAE"].mean(), 2), "MAE_dlc": round(bang["MAE"].std(ddof=0), 2)})
ket_qua = pd.DataFrame(ket_qua)
ket_qua.to_csv(CH.KET_QUA / "b3_chia_ngay_va_ngau_nhien.csv", index=False, encoding="utf-8-sig")
print(ket_qua.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.2))
vi_tri = np.arange(len(mo_hinh))
for i, (ten_chia, mau_cot) in enumerate([("Chia theo ngày", CH.XANH), ("Chia ngẫu nhiên từng mẫu", CH.CAM)]):
    gia_tri = ket_qua[ket_qua["cach_chia"] == ten_chia].set_index("mo_hinh").loc[list(mo_hinh), "MAE_tb"]
    ax.bar(vi_tri - 0.2 + 0.4 * i, gia_tri, width=0.4, color=mau_cot, label=ten_chia)
    for x, val in zip(vi_tri - 0.2 + 0.4 * i, gia_tri):
        ax.text(x, val + 0.1, CH.so_vn(val), ha="center")
ax.set_xticks(vi_tri)
ax.set_xticklabels(list(mo_hinh))
ax.set_ylabel("MAE trung bình 4 vòng (km/h)")
ax.set_title("Kiểm định chéo trên tập huấn luyện: hai cách chia dữ liệu")
ax.legend()
CH.luu_hinh(fig, "b3_chia_ngay_va_ngau_nhien.png")
