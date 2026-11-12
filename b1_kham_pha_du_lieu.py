"""Bước 1. Khám phá dữ liệu gốc.
Đầu vào:  data/raw
Đầu ra:   ket_qua/b1_tom_tat.csv, b1_phan_bo.png (phân bố tốc độ và số mẫu theo ngày), b1_mang_duong.png"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src import cau_hinh as CH, doc_du_lieu as DL

CH.tao_thu_muc()
CH.thiet_lap_hinh()
dong = []


def so(x):
    """Số nguyên có dấu chấm ngăn cách hàng nghìn."""
    return f"{int(x):,}".replace(",", ".")


def them(nhom, chi_so, gia_tri, ghi_chu=""):
    dong.append({"nhom": nhom, "chi_so": chi_so, "gia_tri": gia_tri, "ghi_chu": ghi_chu})


print("Đọc dữ liệu gốc...")
luot = DL.doc_luot_bao()
doan = DL.doc_doan()
mau = DL.gop_mau(luot)
v = luot["velocity"]

# 1. Quy mô và nhãn LOS
them("Quy mô", "Số lượt báo tốc độ", len(luot))
them("Quy mô", "Số mẫu (đoạn, ngày, khung 30 phút) có nhãn", len(mau))
them("Quy mô", "Tỉ lệ mẫu ghép được với lượt báo", round(mau["speed"].notna().mean(), 4))
tao_lai = DL.tai_tao_nhan(luot, doan.set_index("segment_id")["max_velocity"]).rename("los_tao_lai").reset_index()
kiem = mau.merge(tao_lai, on=["segment_id", "date", "period"])
them("Nhãn LOS", "Tỉ lệ nhãn tạo lại đúng (quy từng lượt ra mức, lấy trung bình, làm tròn)",
     round((kiem["los_tao_lai"] == kiem["LOS"]).mean(), 4))
los_tb = DL.quy_ra_los(mau["speed"], mau["max_velocity"])
them("Nhãn LOS", "Tỉ lệ nhãn trùng khi quy tốc độ trung bình ra mức", round((los_tb == mau["LOS"].to_numpy()).mean(), 4))

# Kiểm tra nhất quán của bản đồ: chiều dài ghi trong dữ liệu so với khoảng cách đường thẳng giữa hai đầu đoạn
d = doan.dropna(subset=["lat_s", "lat_e"])
tinh = np.floor(DL.khoang_cach_m(d["lat_s"], d["lon_s"], d["lat_e"], d["lon_e"]))
them("Bản đồ", "Tỉ lệ đoạn có chiều dài ghi đúng bằng khoảng cách haversine giữa hai đầu (làm tròn xuống mét)",
     round((np.abs(tinh - d["length"]) <= 1).mean(), 4), f"{len(d)} đoạn có tọa độ hai đầu")

# 2. Phân bố tốc độ
them("Tốc độ", "Tỉ lệ lượt báo đúng 1 km/h", round((v == 1).mean(), 4))
them("Tốc độ", "Tỉ lệ lượt báo 0 km/h", round((v == 0).mean(), 4))
them("Tốc độ", "Tỉ lệ lượt báo trên 80 km/h", round((v > 80).mean(), 4), f"{(v > 120).sum()} lượt trên 120 km/h, lớn nhất {v.max()} km/h")

# 3. Phân bố theo thời gian
theo_ngay = mau["date_vn"].value_counts().sort_index()
ngay_dong = theo_ngay.idxmax()
n = mau[mau["date_vn"] == ngay_dong]
khoang = (pd.Timestamp(theo_ngay.index.max()) - pd.Timestamp(theo_ngay.index.min())).days + 1
them("Thời gian", f"Tỉ trọng mẫu của ngày {ngay_dong}", round(theo_ngay[ngay_dong] / len(mau), 4))
them("Thời gian", f"Tỉ lệ mẫu của ngày {ngay_dong} rơi vào 6h-8h sáng", round(((n["hour_vn"] >= 6) & (n["hour_vn"] < 8)).mean(), 4))
them("Thời gian", "Số ngày có dữ liệu / số ngày trong khoảng", f"{len(theo_ngay)}/{khoang}",
     f"từ {theo_ngay.index.min()} đến {theo_ngay.index.max()}")
them("Thời gian", "Tỉ lệ mẫu từ 23h đến 5h sáng", round(mau["hour_vn"].astype(int).isin([23, 0, 1, 2, 3, 4]).mean(), 4))

# 4. Thiên lệch theo mức LOS
ty_le = mau["LOS"].value_counts(normalize=True)
luot_tb = mau.groupby("LOS")["n_reports"].mean()
them("Thiên lệch theo LOS", "Tỉ lệ mẫu mức A / mức F", f"{ty_le['A']:.4f} / {ty_le['F']:.4f}")
them("Thiên lệch theo LOS", "Số lượt báo trung bình mỗi mẫu mức A / mức F", f"{luot_tb['A']:.2f} / {luot_tb['F']:.2f}")
them("Thiên lệch theo LOS", "Tỉ lệ mẫu chỉ có 1 lượt báo", round((mau["n_reports"] == 1).mean(), 4))

# 5. Độ phủ và chênh lệch trong khung
moi_doan = mau["segment_id"].value_counts()
them("Độ phủ", "Tỉ lệ đoạn của mạng có ít nhất 1 mẫu", round(len(moi_doan) / len(doan), 4), f"{len(moi_doan)}/{len(doan)} đoạn")
them("Độ phủ", "Trung vị số mẫu mỗi đoạn có dữ liệu", int(moi_doan.median()))
them("Độ phủ", "Tỉ lệ mẫu thiếu tốc độ tối đa cho phép", round(mau["max_velocity"].isna().mean(), 4))
nhieu = mau[mau["n_reports"] >= 2]
lech = nhieu["speed_max"] - nhieu["speed_min"]
them("Chênh lệch trong khung", "Số khung có từ 2 lượt báo", len(nhieu))
them("Chênh lệch trong khung", "Trung vị chênh lệch giữa lượt nhanh nhất và chậm nhất (km/h)", float(lech.median()))
them("Chênh lệch trong khung", "Tỉ lệ khung lệch từ 20 km/h trở lên", round((lech >= 20).mean(), 4))

# 6. Hình phân bố
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
toc = np.arange(0, 101)
dem = v.value_counts().reindex(toc, fill_value=0)
a1.bar(toc, dem / len(v) * 100, color=[CH.CAM if t == 1 else CH.XANH for t in toc], width=0.8)
a1.set_xlabel("Tốc độ báo (km/h)")
a1.set_ylabel("Tỉ lệ lượt báo (%)")
a1.set_title(f"Phân bố tốc độ của {len(v):,} lượt báo".replace(",", "."))
a1.annotate(f"1 km/h: {CH.so_vn((v == 1).mean() * 100, 1)}%", (1, (v == 1).mean() * 100), xytext=(8, -4),
            textcoords="offset points", va="top")
ngay = pd.to_datetime(theo_ngay.index)
a2.bar(ngay, theo_ngay.values, color=[CH.CAM if d == ngay_dong else CH.XANH for d in theo_ngay.index], width=1.5)
a2.set_xlabel("Ngày")
a2.set_ylabel("Số mẫu")
a2.set_title(f"Số mẫu theo ngày ({len(theo_ngay)}/{khoang} ngày có dữ liệu)")
a2.annotate(f"03/08/2020: {theo_ngay[ngay_dong]:,} mẫu".replace(",", "."), (pd.Timestamp(ngay_dong), theo_ngay[ngay_dong]),
            xytext=(8, -4), textcoords="offset points", va="top")
a2.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
a2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%Y"))
CH.luu_hinh(fig, "b1_phan_bo.png")

# 6b. Ba biểu đồ khảo sát bổ sung: tỉ trọng ngày 03/08, số lượt báo mỗi mẫu theo mức LOS, số mẫu theo giờ
fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [0.8, 1, 1.2]})
ty = theo_ngay[ngay_dong] / len(mau)
b1.pie([ty, 1 - ty], colors=[CH.CAM, CH.XANH], startangle=90, counterclock=False,
       labels=[f"Ngày 03/08/2020\n{CH.so_vn(ty * 100, 1)}%", f"{len(theo_ngay) - 1} ngày còn lại\n{CH.so_vn((1 - ty) * 100, 1)}%"],
       wedgeprops=dict(edgecolor="white"), textprops=dict(fontsize=10.5))
b1.set_title("Tỉ trọng số mẫu", fontsize=12)
muc = list(CH.MUC_LOS)
b2.bar(muc, [luot_tb[m] for m in muc], color=CH.XANH)
for i, m in enumerate(muc):
    b2.text(i, luot_tb[m] + 0.1, CH.so_vn(luot_tb[m], 2), ha="center", fontsize=10.5)
b2.set_xlabel("Mức LOS của mẫu")
b2.set_ylabel("Số lượt báo trung bình mỗi mẫu")
b2.set_title("Đường kẹt được báo cáo dày hơn", fontsize=12)
gio = mau["hour_vn"].astype(int).value_counts().reindex(range(24), fill_value=0)
b3.bar(gio.index, gio.values, color=[CH.CAM if h in (23, 0, 1, 2, 3, 4) else CH.XANH for h in gio.index])
b3.set_xlabel("Giờ trong ngày")
b3.set_ylabel("Số mẫu")
b3.set_title("Số mẫu theo giờ", fontsize=12)
b3.set_xticks(range(0, 24, 2))
CH.luu_hinh(fig, "b1_khao_sat_them.png")

# 7. Bản đồ mạng đường: toàn bộ đoạn màu xám, đoạn có mẫu màu đỏ
fig, ax = plt.subplots(figsize=(7, 5.6))
duong = [[(r.lon_s, r.lat_s), (r.lon_e, r.lat_e)] for r in doan.itertuples()]
ax.add_collection(LineCollection(duong, colors="#C9D1D8", linewidths=0.3))
co_mau = doan["segment_id"].isin(moi_doan.index)
duong_co = [[(r.lon_s, r.lat_s), (r.lon_e, r.lat_e)] for r in doan[co_mau].itertuples()]
ax.add_collection(LineCollection(duong_co, colors=CH.DO, linewidths=0.8))
ax.autoscale()
ax.set_aspect(1 / np.cos(np.radians(10.8)))
ax.set_xlabel("Kinh độ")
ax.set_ylabel("Vĩ độ")
ax.set_title(f"Mạng đường {so(len(doan))} đoạn, trong đó {so(co_mau.sum())} đoạn (màu đỏ) có dữ liệu", fontsize=11)
CH.luu_hinh(fig, "b1_mang_duong.png")

# 8. Đồ thị đường (cần networkx)
import networkx as nx

G = nx.MultiDiGraph()
for r in doan.itertuples():
    G.add_edge(int(r.s_node_id), int(r.e_node_id), key=int(r.segment_id))
lon_nhat = max(nx.strongly_connected_components(G), key=len)
trong = sum(1 for u, w in G.edges() if u in lon_nhat and w in lon_nhat)
them("Đồ thị đường", "Số nút / số cạnh", f"{G.number_of_nodes()} / {G.number_of_edges()}")
them("Đồ thị đường", "Tỉ lệ nút nằm trong thành phần liên thông mạnh lớn nhất", round(len(lon_nhat) / G.number_of_nodes(), 4),
     f"{trong} đoạn nằm trong thành phần này")

tom_tat = pd.DataFrame(dong)
tom_tat.to_csv(CH.KET_QUA / "b1_tom_tat.csv", index=False, encoding="utf-8-sig")
print(tom_tat[["chi_so", "gia_tri"]].to_string(index=False))
