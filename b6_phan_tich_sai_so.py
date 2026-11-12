"""Bước 6. Phân tích sai số của Random Forest (v2) trên tập huấn luyện: mô hình sai ở đâu.
Dùng dự báo ngoài mẫu của kiểm định chéo 4 vòng theo ngày (mỗi mẫu được dự báo bởi mô hình chưa thấy ngày của nó).
Đầu vào:  data/sach, ket_qua/b4_tham_so_tot_nhat_v2.json
Đầu ra:   ket_qua/b6_*.csv và các hình b6_*.png"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import cau_hinh as CH, doc_du_lieu as DL, mo_hinh as MH

warnings.filterwarnings("ignore", message="Found unknown categories")
CH.tao_thu_muc()
CH.thiet_lap_hinh()

df = DL.doc_mau("v2")
df = df[df["tap"] == "huan_luyen"].reset_index(drop=True)
tham_so = MH.doc_tham_so_tot_nhat("v2")["Random Forest"]["tham_so"]
bang, ngoai_mau = MH.kiem_dinh_cheo(MH.tao_random_forest(**tham_so), df, df["phan"].to_numpy(), ten="Random Forest")
a = df[["sample_id", "segment_id", "date_vn", "hour_vn", "lat", "lon", "LOS", "speed", "phan"]].copy()
a["du_bao"] = np.round(ngoai_mau, 3)
a["lech"] = a["du_bao"] - a["speed"]              # dương: dự báo nhanh hơn thật
a["sai_so"] = a["lech"].abs()
a["gio"] = a["hour_vn"].astype(int)
a.to_csv(CH.KET_QUA / "b6_du_bao_ngoai_mau.csv", index=False, encoding="utf-8-sig")
print(f"MAE ngoài mẫu: {a['sai_so'].mean():.3f}")

# 1. Theo ngày: bao nhiêu phần phương sai của sai số nằm giữa các ngày
trung_binh_ngay = a.groupby("date_vn")["lech"].transform("mean")
ss_giua = ((trung_binh_ngay - a["lech"].mean()) ** 2).sum()
ss_tong = ((a["lech"] - a["lech"].mean()) ** 2).sum()
ty_le_ngay = ss_giua / ss_tong
print(f"Phần phương sai của sai số nằm giữa các ngày: {ty_le_ngay:.4f}")
theo_ngay = a.groupby("date_vn").agg(so_mau=("speed", "size"), that_tb=("speed", "mean"), du_bao_tb=("du_bao", "mean"),
                                     MAE=("sai_so", "mean")).round(3).reset_index()
theo_ngay.to_csv(CH.KET_QUA / "b6_theo_ngay.csv", index=False, encoding="utf-8-sig")
ngay_nhieu = theo_ngay[theo_ngay["so_mau"] >= 30]          # ngày có từ 30 mẫu
tq_ngay = np.corrcoef(ngay_nhieu["that_tb"], ngay_nhieu["du_bao_tb"] - ngay_nhieu["that_tb"])[0, 1]
pd.DataFrame([{"phuong_sai_sai_so_giua_cac_ngay": round(ty_le_ngay, 4), "so_ngay": a["date_vn"].nunique(), "so_mau": len(a),
               "so_ngay_tu_30_mau": len(ngay_nhieu), "tuong_quan_toc_do_ngay_va_lech_ngay": round(tq_ngay, 3)}]) \
    .to_csv(CH.KET_QUA / "b6_ty_le_giua_ngay.csv", index=False, encoding="utf-8-sig")

d = theo_ngay[theo_ngay["so_mau"] >= 30]
tran = max(d["that_tb"].max(), d["du_bao_tb"].max()) * 1.05
fig, ax = plt.subplots(figsize=(6, 5.6))
ax.plot([0, tran], [0, tran], color=CH.XAM, linestyle="--", label="dự báo bằng thực tế")
ax.scatter(d["that_tb"], d["du_bao_tb"], s=np.clip(d["so_mau"] / 8, 15, 300), color=CH.XANH, alpha=0.75)
ax.set_xlim(0, tran)
ax.set_ylim(0, tran)
ax.set_xlabel("Tốc độ thật trung bình của ngày (km/h)")
ax.set_ylabel("Tốc độ dự báo trung bình của ngày (km/h)")
ax.set_title("Mỗi chấm là một ngày có từ 30 mẫu")
ax.legend(loc="upper left")
CH.luu_hinh(fig, "b6_lech_theo_ngay.png")

# 2. Theo mức LOS thật: dự báo bị kéo về mức giữa
theo_los = a.groupby("LOS").agg(so_mau=("speed", "size"), toc_do_that_tb=("speed", "mean"), du_bao_tb=("du_bao", "mean"),
                                lech_tb=("lech", "mean"), MAE=("sai_so", "mean")).round(3).reindex(CH.MUC_LOS).reset_index()
theo_los.to_csv(CH.KET_QUA / "b6_theo_los.csv", index=False, encoding="utf-8-sig")
print(theo_los.to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(theo_los["LOS"], theo_los["lech_tb"], color=[CH.DO if x > 0 else CH.XANH for x in theo_los["lech_tb"]])
ax.axhline(0, color=CH.XAM)
for i, r in theo_los.iterrows():
    ax.text(i, r["lech_tb"] + (0.8 if r["lech_tb"] >= 0 else -2.2), f"{CH.so_vn(r['lech_tb'], 1)}", ha="center")
ax.set_xticks(range(6))
ax.set_xticklabels([f"{r['LOS']}\n{int(r['so_mau']):,} mẫu".replace(",", ".") for _, r in theo_los.iterrows()])
ax.set_xlabel("Mức LOS thật (nhãn gốc)")
ax.set_ylabel("Dự báo trừ thật, trung bình (km/h)")
ax.set_title("Dự báo nhanh hơn thật ở mức kẹt, chậm hơn thật ở mức thông thoáng")
CH.luu_hinh(fig, "b6_lech_theo_los.png")

# 3. Theo giờ
theo_gio = a.groupby("gio").agg(so_mau=("speed", "size"), that_tb=("speed", "mean"), du_bao_tb=("du_bao", "mean"),
                                MAE=("sai_so", "mean")).round(3).reset_index()
theo_gio.to_csv(CH.KET_QUA / "b6_theo_gio.csv", index=False, encoding="utf-8-sig")
g = theo_gio[theo_gio["so_mau"] >= 50]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(g["gio"], g["MAE"], marker="o", color=CH.XANH)
ax.set_xticks(g["gio"])
ax.set_xlabel("Giờ trong ngày (giờ Việt Nam)")
ax.set_ylabel("MAE (km/h)")
ax.set_title("Sai số theo giờ (các giờ có từ 50 mẫu)")
CH.luu_hinh(fig, "b6_sai_so_theo_gio.png")

# 4. Theo lượng lịch sử của đoạn: số mẫu của chính đoạn nằm ở ba phần huấn luyện còn lại của vòng dự báo
tong_doan = a["segment_id"].map(a["segment_id"].value_counts())
cung_phan = a.groupby(["segment_id", "phan"])["segment_id"].transform("size")
a["mau_lich_su"] = tong_doan - cung_phan
a["nhom_lich_su"] = pd.cut(a["mau_lich_su"], [-1, 0, 5, 20, 10 ** 9], labels=["chưa có", "1-5 mẫu", "6-20 mẫu", "trên 20 mẫu"])
theo_ls = a.groupby("nhom_lich_su", observed=True).agg(so_mau=("speed", "size"), MAE=("sai_so", "mean")).round(3).reset_index()
theo_ls.to_csv(CH.KET_QUA / "b6_theo_lich_su.csv", index=False, encoding="utf-8-sig")
print(theo_ls.to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(theo_ls["nhom_lich_su"].astype(str), theo_ls["MAE"], color=CH.XANH)
for i, r in theo_ls.iterrows():
    ax.text(i, r["MAE"] + 0.2, CH.so_vn(r["MAE"]), ha="center")
ax.set_xlabel("Số mẫu của chính đoạn trong dữ liệu huấn luyện của vòng dự báo")
ax.set_ylabel("MAE (km/h)")
ax.set_title("Đoạn càng có nhiều lịch sử, dự báo càng chính xác")
CH.luu_hinh(fig, "b6_sai_so_theo_lich_su.png")

# 5. Phần dư của các đoạn gần nhau: tương quan theo khoảng cách (cặp mẫu cùng ngày, cùng khung 30 phút)
a["phan_du"] = a["speed"] - a["du_bao"]
a["phan_du_trong_ngay"] = a["phan_du"] - a.groupby("date_vn")["phan_du"].transform("mean")
a["khung"] = (a["hour_vn"] * 2).astype(int)
cap = []
for _, g in a.groupby(["date_vn", "khung"]):
    if len(g) < 2:
        continue
    i, j = np.triu_indices(len(g), k=1)
    khoang = DL.khoang_cach_m(g["lat"].to_numpy()[i], g["lon"].to_numpy()[i], g["lat"].to_numpy()[j], g["lon"].to_numpy()[j])
    chi_so_g = g.index.to_numpy()
    cap.append(pd.DataFrame({"i": chi_so_g[i], "j": chi_so_g[j], "khoang": khoang}))
cap = pd.concat(cap, ignore_index=True)
bien = [0, 200, 500, 1000, 2000, 5000, 1e9]
nhan = ["dưới 200 m", "200-500 m", "0,5-1 km", "1-2 km", "2-5 km", "trên 5 km"]
nhom = pd.cut(cap["khoang"], bien, labels=nhan, right=False)
dong = []
for cot, ten in [("phan_du", "phần dư"), ("phan_du_trong_ngay", "phần dư đã trừ trung bình của ngày")]:
    r = a[cot].to_numpy()
    for t in nhan:
        p = cap[nhom == t]
        dong.append({"loai": ten, "khoang_cach": t, "so_cap": len(p), "tuong_quan": round(float(np.corrcoef(r[p["i"]], r[p["j"]])[0, 1]), 4)})
tq = pd.DataFrame(dong)
tq.to_csv(CH.KET_QUA / "b6_tuong_quan_theo_khoang_cach.csv", index=False, encoding="utf-8-sig")
print(tq.to_string(index=False))
fig, ax = plt.subplots(figsize=(9, 4.2))
for ten, mau_dg in [("phần dư", CH.XAM), ("phần dư đã trừ trung bình của ngày", CH.XANH)]:
    s = tq[tq["loai"] == ten]
    ax.plot(range(6), s["tuong_quan"], marker="o", color=mau_dg, label=ten)
ax.axhline(0, color=CH.XAM, linewidth=0.8)
ax.set_xticks(range(6))
ax.set_xticklabels(nhan)
ax.set_xlabel("Khoảng cách giữa hai đoạn (cùng ngày, cùng khung 30 phút)")
ax.set_ylabel("Hệ số tương quan của phần dư")
ax.set_title("Phần dư của hai đoạn gần nhau có xu hướng giống nhau")
ax.legend()
CH.luu_hinh(fig, "b6_tuong_quan_theo_khoang_cach.png")
