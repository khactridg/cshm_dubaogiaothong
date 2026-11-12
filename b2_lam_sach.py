"""Bước 2. Làm sạch dữ liệu theo hai phiên bản v1 và v2, gộp thành mẫu.
Đầu vào:  data/raw
Đầu ra:   data/sach/mau_v1.csv, mau_v2.csv
          ket_qua/b2_quy_tac.csv (số lượt bị loại ở từng quy tắc), b2_so_sanh.csv (v1 và v2),
          b2_bang_chung_*.csv và các hình b2_*.png (bằng chứng cho các nhóm lượt báo bất thường)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import cau_hinh as CH, doc_du_lieu as DL, lam_sach as LS

CH.tao_thu_muc()
CH.thiet_lap_hinh()

luot = DL.doc_luot_bao()
doan = DL.doc_doan()
d = doan.set_index("segment_id")
print("Số lượt báo gốc:", len(luot))

# 1. Làm sạch và gộp mẫu cho từng phiên bản
nhat_ky_chung, so_sanh = [], []
for pb in CH.PHIEN_BAN:
    sach, nhat_ky = LS.lam_sach_luot_bao(luot, doan, pb)
    mau = LS.lam_sach_mau(DL.gop_mau(sach), nhat_ky, pb)
    mau["los_from_speed"] = DL.quy_ra_los(mau["speed"], mau["max_velocity"])
    out = mau.copy()
    out["dt_vn"] = out["dt_vn"].dt.strftime("%Y-%m-%d %H:%M")
    out.to_csv(CH.DU_LIEU_SACH / f"mau_{pb}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    tao_lai = DL.tai_tao_nhan(sach, d["max_velocity"]).rename("los_tao_lai").reset_index()
    khop = mau.merge(tao_lai, on=["segment_id", "date", "period"])
    ty_le = mau["los_from_speed"].value_counts(normalize=True)
    dong = {"phien_ban": pb, "so_luot_bao_con_lai": len(sach), "so_mau": len(mau), "so_ngay": mau["date_vn"].nunique(),
            "so_doan": mau["segment_id"].nunique(), "toc_do_tb_kmh": round(mau["speed"].mean(), 2),
            "do_lech_chuan_toc_do": round(mau["speed"].std(), 2),
            "ty_le_mau_chi_1_luot_bao": round((mau["n_reports"] == 1).mean(), 4),
            "nhan_tao_lai_trung_nhan_goc": round((khop["los_tao_lai"] == khop["LOS"]).mean(), 4)}
    nhieu = mau[mau["n_reports"] >= 2]
    chenh = nhieu["speed_max"] - nhieu["speed_min"]          # chênh lệch giữa lượt nhanh nhất và chậm nhất trong khung
    dong["so_khung_tu_2_luot"] = len(nhieu)
    dong["trung_vi_chenh_lech_kmh"] = chenh.median()
    dong["ty_le_khung_lech_tu_20kmh"] = round((chenh >= 20).mean(), 4)
    dong["los_tu_toc_do_tb_trung_nhan_goc"] = round((mau["los_from_speed"] == mau["LOS"]).mean(), 4)
    for m in CH.MUC_LOS:
        dong["ty_le_LOS_" + m] = round(ty_le.get(m, 0), 4)
    so_sanh.append(dong)
    for r in nhat_ky:
        r["phien_ban"] = pb
    nhat_ky_chung += nhat_ky
    print(f"\n{pb}: {len(sach)} lượt báo, {len(mau)} mẫu")
    print(pd.DataFrame(nhat_ky)[["ma", "quy_tac", "so_bi_anh_huong", "cach_xu_ly"]].to_string(index=False))

pd.DataFrame(nhat_ky_chung).to_csv(CH.KET_QUA / "b2_quy_tac.csv", index=False, encoding="utf-8-sig")
so_sanh = pd.DataFrame(so_sanh)
so_sanh.to_csv(CH.KET_QUA / "b2_so_sanh.csv", index=False, encoding="utf-8-sig")
print("\n", so_sanh.T.to_string(header=False))

# 2. Bằng chứng cho các nhóm bất thường (tính trên toàn bộ lượt báo gốc)
rep = LS.sap_xep_luot_bao(luot)
ngay = rep["ts_vn"].dt.strftime("%Y-%m-%d")

# 2.1 Ngày 03/08/2020: tốc độ 1-39 km/h xuất hiện gần như đều nhau
la = rep[ngay == CH.NGAY_BAT_THUONG]
con_lai = rep[ngay != CH.NGAY_BAT_THUONG]
dem_la = la["velocity"].value_counts()
dem_con_lai = con_lai["velocity"].value_counts()
trong_la = dem_la.reindex(range(1, 40), fill_value=0)
trong_con_lai = dem_con_lai.reindex(range(1, 40), fill_value=0)
b = pd.DataFrame([
    ["Số lượt báo", len(la), len(con_lai)],
    ["Khoảng giá trị tốc độ (km/h)", f"{la['velocity'].min()}-{la['velocity'].max()}", f"{con_lai['velocity'].min()}-{con_lai['velocity'].max()}"],
    ["Số lần xuất hiện mỗi giá trị 1-39 km/h (nhỏ nhất - lớn nhất)", f"{trong_la.min()}-{trong_la.max()}", f"{trong_con_lai.min()}-{trong_con_lai.max()}"],
    ["Tỉ lệ lượt trùng thời điểm với lượt liền trước", round((la["dt"] == 0).mean(), 3), round((con_lai["dt"] == 0).mean(), 3)],
    ["Tỉ lệ lượt trong 6h-8h giờ Việt Nam", round(la["ts_vn"].dt.hour.isin([6, 7]).mean(), 3), round(con_lai["ts_vn"].dt.hour.isin([6, 7]).mean(), 3)],
], columns=["chi_so", "ngay_03_08_2020", "cac_ngay_con_lai"])
b.to_csv(CH.KET_QUA / "b2_bang_chung_ngay_03_08.csv", index=False, encoding="utf-8-sig")
print("\n", b.to_string(index=False))

toc = np.arange(0, 41)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
a1.bar(toc, dem_la.reindex(toc, fill_value=0) / len(la) * 100, color=CH.CAM)
a1.set_title(f"Ngày 03/08/2020 ({len(la):,} lượt)".replace(",", "."))
a2.bar(toc, dem_con_lai.reindex(toc, fill_value=0) / len(con_lai) * 100, color=CH.XANH)
a2.set_title(f"Các ngày còn lại ({len(con_lai):,} lượt)".replace(",", "."))
for a in (a1, a2):
    a.set_xlabel("Tốc độ báo (km/h)")
a1.set_ylabel("Tỉ lệ lượt báo (%)")
CH.luu_hinh(fig, "b2_ngay_03_08.png")

# 2.2 Thiết bị đứng yên: chuỗi lượt liền nhau trên một đoạn, tốc độ rất thấp nhưng quãng đường theo tốc độ báo rất dài
rep["chuoi"] = LS.danh_so_chuoi(rep, "segment_id")
x = rep[LS.la_dung_yen(rep, d["length"])]
g = x.groupby("chuoi")
chuoi = pd.DataFrame({"ngay": g["ts_vn"].first().dt.strftime("%Y-%m-%d"), "bat_dau": g["ts_vn"].first().dt.strftime("%H:%M"),
                      "so_luot": g.size(), "thoi_gian_phut": ((g["t"].max() - g["t"].min()).dt.total_seconds() / 60).round(1),
                      "toc_do_tb_kmh": g["velocity"].mean().round(2), "segment_id": g["segment_id"].first()})
chuoi["chieu_dai_doan_m"] = chuoi["segment_id"].map(d["length"])
chuoi["quang_duong_theo_toc_do_bao_m"] = (chuoi["toc_do_tb_kmh"] / 3.6 * chuoi["thoi_gian_phut"] * 60).round(0)
chuoi["ten_duong"] = chuoi["segment_id"].map(d["street_name"])
chuoi = chuoi.sort_values("so_luot", ascending=False)
chuoi.to_csv(CH.KET_QUA / "b2_bang_chung_dung_yen.csv", index=False, encoding="utf-8-sig")
dai = chuoi.iloc[0]
print(f"\nThiết bị đứng yên: {len(chuoi)} chuỗi, {chuoi['so_luot'].sum()} lượt. Chuỗi dài nhất: {dai['so_luot']} lượt, "
      f"đoạn dài {dai['chieu_dai_doan_m']} m, quãng đường theo tốc độ báo {dai['quang_duong_theo_toc_do_bao_m']} m")

fig, ax = plt.subplots(figsize=(7, 3.2))
gia_tri = [dai["chieu_dai_doan_m"], dai["quang_duong_theo_toc_do_bao_m"]]
ax.barh(["Chiều dài đoạn đường", "Quãng đường tính từ\ntốc độ thiết bị báo"], gia_tri, color=[CH.XANH, CH.DO])
for i, val in enumerate(gia_tri):
    ax.text(val, i, f" {val:,.0f} m".replace(",", "."), va="center")
ax.set_xlim(0, max(gia_tri) * 1.2)
ax.set_xlabel("Mét")
ax.set_title(f"Chuỗi {dai['so_luot']} lượt báo liền nhau trên một đoạn,\ntốc độ trung bình {CH.so_vn(dai['toc_do_tb_kmh'])} km/h", fontsize=12)
CH.luu_hinh(fig, "b2_dung_yen.png")

# 2.3 Tốc độ đóng băng: độ dài các chuỗi lượt báo cùng một giá trị
rep["chuoi"] = LS.danh_so_chuoi(rep, "velocity")
rep["ngay"] = ngay
g = rep.groupby("chuoi")
tat_ca = pd.DataFrame({"so_luot": g.size(), "gia_tri": g["velocity"].first(), "so_doan": g["segment_id"].nunique(),
                       "ngay": g["ngay"].first()})
thuong = tat_ca[(tat_ca["ngay"] != CH.NGAY_BAT_THUONG) & (tat_ca["gia_tri"] != 1)]   # bỏ ngày 03/08 và giá trị 1 km/h
dai_nhat = thuong[thuong["so_luot"] >= CH.DONG_BANG_LUOT].sort_values("so_doan").iloc[-1]
b = pd.DataFrame([["Số chuỗi (bỏ ngày 03/08 và giá trị 1 km/h)", len(thuong)], ["Trung vị độ dài chuỗi", thuong["so_luot"].median()],
                  ["Phân vị 99,9% độ dài chuỗi", thuong["so_luot"].quantile(0.999)],
                  [f"Số chuỗi từ {CH.DONG_BANG_LUOT} lượt", int((thuong["so_luot"] >= CH.DONG_BANG_LUOT).sum())],
                  ["Chuỗi đi qua nhiều đoạn nhất: số lượt / số đoạn khác nhau", f"{dai_nhat['so_luot']} / {dai_nhat['so_doan']}"]],
                 columns=["chi_so", "gia_tri"])
b.to_csv(CH.KET_QUA / "b2_bang_chung_dong_bang.csv", index=False, encoding="utf-8-sig")
print("\n", b.to_string(index=False))

# 2.4 Lượt từ 70 km/h trên đường không phải cao tốc, quốc lộ theo giờ
nhanh = rep[LS.la_qua_nhanh_noi_thanh(rep, d["street_type"])]
theo_gio = nhanh["ts_vn"].dt.hour.value_counts().reindex(range(24), fill_value=0)
theo_gio.rename("so_luot").to_csv(CH.KET_QUA / "b2_bang_chung_qua_nhanh_theo_gio.csv", encoding="utf-8-sig")
print(f"\nLượt từ {CH.NOI_THANH_KMH} km/h trên đường nội thành: {len(nhanh)}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
dem = thuong["so_luot"].value_counts().sort_index()
a1.bar(dem.index, dem.values, color=[CH.CAM if k >= CH.DONG_BANG_LUOT else CH.XANH for k in dem.index])
a1.set_yscale("log")
a1.axvline(CH.DONG_BANG_LUOT - 0.5, color=CH.XAM, linestyle="--")
a1.set_xlabel("Số lượt liền nhau có cùng một giá trị tốc độ")
a1.set_ylabel("Số chuỗi (thang log)")
a1.set_title(f"Chuỗi từ {CH.DONG_BANG_LUOT} lượt (màu cam) bị loại")
a2.bar(theo_gio.index, theo_gio.values, color=CH.XANH)
a2.set_xticks(range(0, 24, 2))
a2.set_xlabel("Giờ trong ngày (giờ Việt Nam)")
a2.set_ylabel("Số lượt báo")
a2.set_title(f"Lượt báo từ {CH.NOI_THANH_KMH} km/h trên đường không phải cao tốc, quốc lộ")
CH.luu_hinh(fig, "b2_dong_bang_va_qua_nhanh.png")

# 3. Phân bố LOS của hai phiên bản
vi_tri = np.arange(6)
fig, ax = plt.subplots(figsize=(9, 4.2))
ten = {"v1": "v1 (làm sạch ban đầu)", "v2": "v2 (làm sạch bổ sung)"}
for i, pb in enumerate(CH.PHIEN_BAN):
    mau = DL.doc_mau(pb, kem_chia=False)
    ty_le = mau["los_from_speed"].value_counts(normalize=True).reindex(CH.MUC_LOS, fill_value=0)
    ax.bar(vi_tri - 0.2 + 0.4 * i, ty_le * 100, width=0.4, color=[CH.XAM, CH.XANH][i], label=f"{ten[pb]}: {len(mau):,} mẫu".replace(",", "."))
ax.set_xticks(vi_tri)
ax.set_xticklabels([f"{k}\n{CH.TEN_LOS[k]}" for k in CH.MUC_LOS])
ax.set_ylabel("Tỉ lệ mẫu (%)")
ax.set_title("Phân bố mức LOS quy từ tốc độ của hai phiên bản làm sạch")
ax.legend()
CH.luu_hinh(fig, "b2_phan_bo_los.png")
