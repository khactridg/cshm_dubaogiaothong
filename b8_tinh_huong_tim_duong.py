"""Bước 8. Tìm đường: so sánh đường nhanh nhất theo dự báo với đường ngắn nhất ở bốn tình huống, và kiểm tra cài đặt.
Điểm xuất phát là Trường ĐH Sư phạm TP.HCM, cơ sở An Dương Vương. Mô hình dự báo: Random Forest (v2).
Đầu vào:  mo_hinh/v2, data/sach
Đầu ra:   ket_qua/b8_tinh_huong.csv, b8_kiem_tra.csv, b8_ban_do_tinh_huong.png
Mọi thời gian trong bước này là thời gian THEO DỰ BÁO của mô hình, không phải thời gian đo được ngoài thực tế."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src import cau_hinh as CH, tim_duong as TD

warnings.filterwarnings("ignore", message="Found unknown categories")
CH.tao_thu_muc()
CH.thiet_lap_hinh()
XUAT_PHAT = (10.76116, 106.68328)
# (tên, đích (vĩ độ, kinh độ), thứ (0 = Thứ hai), giờ khởi hành)
TINH_HUONG = [("Đường Tân Sơn Nhì, Tân Phú, Thứ hai 7:30", (10.79862, 106.63177), 0, 7.5),
              ("Đường Liên khu 10-11, Bình Tân, Thứ hai 17:30", (10.76804, 106.61787), 0, 17.5),
              ("Đường Phan Đăng Lưu, Phú Nhuận, Thứ hai 7:30", (10.79923, 106.68040), 0, 7.5),
              ("Đường Hồng Bàng, Quận 6, Thứ sáu 17:30", (10.75421, 106.64398), 4, 17.5)]

print("Dựng đồ thị và nạp mô hình...")
td = TD.TimDuong("v2")
print(f"{len(td.doan)} đoạn, {td.do_thi.number_of_nodes()} nút, thành phần liên thông mạnh lớn nhất {len(td.lien_thong)} nút")


def do_trung(a, b):
    """Chiều dài phần chung / chiều dài phần hợp của hai lộ trình (0 = khác hẳn, 1 = trùng khít)."""
    dai = dict(zip(td.doan["segment_id"], td.doan["length"]))
    hop = sum(dai[s] for s in set(a) | set(b))
    chung = sum(dai[s] for s in set(a) & set(b))
    return chung / hop if hop else 1.0


# 1. Bốn tình huống
dong, ket_qua = [], []
for ten, den, thu, gio in TINH_HUONG:
    r = td.tim(XUAT_PHAT, den, thu, gio, TD.MAC_DINH)
    nhanh, ngan = r["nhanh_nhat"], r["ngan_nhat"]
    ct = nhanh["chi_tiet"]
    dong.append({"tinh_huong": ten, "km_nhanh_nhat": round(nhanh["km"], 2), "phut_nhanh_nhat_du_bao": round(nhanh["phut"], 1),
                 "km_ngan_nhat": round(ngan["km"], 2), "phut_ngan_nhat_du_bao": round(ngan["phut"], 1),
                 "tiet_kiem_phut_du_bao": round(ngan["phut"] - nhanh["phut"], 1),
                 "do_trung_hai_lo_trinh": round(do_trung(nhanh["chi_tiet"]["segment_id"], ngan["chi_tiet"]["segment_id"]), 3),
                 "ty_le_dai_co_lich_su": round(nhanh["ty_le_dai_co_lich_su"], 3),
                 "ty_le_dai_muc_E_F": round(ct.loc[ct["LOS_du_bao"].isin(["E", "F"]), "length"].sum() / ct["length"].sum(), 3)})
    ket_qua.append((ten, r))
    print(f"{ten}: nhanh nhất {nhanh['km']:.2f} km, {nhanh['phut']:.1f} phút; ngắn nhất {ngan['km']:.2f} km, {ngan['phut']:.1f} phút")
tinh_huong = pd.DataFrame(dong)
tinh_huong.to_csv(CH.KET_QUA / "b8_tinh_huong.csv", index=False, encoding="utf-8-sig")
print(tinh_huong.T.to_string(header=False))

# 2. Kiểm tra cài đặt
kiem_tra = []
# (1) FIFO: 300 đoạn chọn ngẫu nhiên, mỗi đoạn thử 20 thời điểm vào quanh ranh giới khung 7:30 (7:20 đến 7:39)
rng = np.random.default_rng(CH.SEED)
vao = 7 * 60 + 20 + np.arange(20)
vi_pham = 0
for vt in rng.choice(len(td.doan), 300, replace=False):
    ra = np.array([td.qua_doan(int(vt), t, TD.MAC_DINH, 0) for t in vao])
    vi_pham += int(np.sum(np.diff(ra) < -1e-9))          # vào muộn hơn mà ra sớm hơn
kiem_tra.append({"kiem_tra": "FIFO: 300 đoạn x 20 thời điểm vào quanh 7:30", "so_truong_hop": 300 * 19, "so_vi_pham": vi_pham})
# (2) Nhất quán: nhãn của nút đích trừ giờ khởi hành bằng tổng thời gian đi lại theo lộ trình tìm được
lech = max(abs(r["nhan_nut_den_phut"] - r["nhanh_nhat"]["phut"]) for _, r in ket_qua)
kiem_tra.append({"kiem_tra": "Nhất quán: nhãn nút đích so với tổng thời gian trên lộ trình (lệch lớn nhất, phút)",
                 "so_truong_hop": len(ket_qua), "so_vi_pham": int(lech > 1e-6)})
# (3) Tối ưu tương đối: đường nhanh nhất không chậm hơn đường ngắn nhất, thử thêm 20 cặp điểm ngẫu nhiên
nut = td.nut_chon
tren = []
for _ in range(20):
    a, b = rng.choice(nut, 2, replace=False)
    toa_do = [(td.do_thi.nodes[n]["lat"], td.do_thi.nodes[n]["lon"]) for n in (a, b)]
    r = td.tim(toa_do[0], toa_do[1], int(rng.integers(0, 7)), float(rng.choice(np.arange(6, 22, 0.5))), TD.MAC_DINH)
    tren.append(r["nhanh_nhat"]["phut"] <= r["ngan_nhat"]["phut"] + 1e-6)
tren += [r["nhanh_nhat"]["phut"] <= r["ngan_nhat"]["phut"] + 1e-6 for _, r in ket_qua]
kiem_tra.append({"kiem_tra": "Đường nhanh nhất không chậm hơn đường ngắn nhất (4 tình huống + 20 cặp ngẫu nhiên)",
                 "so_truong_hop": len(tren), "so_vi_pham": int(len(tren) - sum(tren))})
kiem_tra = pd.DataFrame(kiem_tra)
kiem_tra.to_csv(CH.KET_QUA / "b8_kiem_tra.csv", index=False, encoding="utf-8-sig")
print(kiem_tra.to_string(index=False))

# 3. Bản đồ bốn tình huống
nen = [[(r.lon_s, r.lat_s), (r.lon_e, r.lat_e)] for r in td.doan.itertuples()]
fig, truc = plt.subplots(2, 2, figsize=(11, 9))
for ax, (ten, r) in zip(truc.ravel(), ket_qua):
    ax.add_collection(LineCollection(nen, colors="#DDE3E8", linewidths=0.25))
    ngan = np.array([(lon, lat) for lat, lon in r["ngan_nhat"]["toa_do"]])
    nhanh = np.array([(lon, lat) for lat, lon in r["nhanh_nhat"]["toa_do"]])
    ax.plot(ngan[:, 0], ngan[:, 1], color=CH.XANH, linestyle="--", linewidth=2, label=f"Ngắn nhất: {CH.so_vn(r['ngan_nhat']['km'])} km, {CH.so_vn(r['ngan_nhat']['phut'], 1)} phút")
    ax.plot(nhanh[:, 0], nhanh[:, 1], color=CH.DO, linewidth=2, label=f"Nhanh nhất: {CH.so_vn(r['nhanh_nhat']['km'])} km, {CH.so_vn(r['nhanh_nhat']['phut'], 1)} phút")
    ax.scatter(*XUAT_PHAT[::-1], color="black", zorder=5, s=30)
    x = np.concatenate([ngan[:, 0], nhanh[:, 0]])
    y = np.concatenate([ngan[:, 1], nhanh[:, 1]])
    ax.set_xlim(x.min() - 0.01, x.max() + 0.01)
    ax.set_ylim(y.min() - 0.01, y.max() + 0.01)
    ax.set_aspect(1 / np.cos(np.radians(10.8)))
    ax.set_title(ten, fontsize=10.5)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
CH.luu_hinh(fig, "b8_ban_do_tinh_huong.png")
