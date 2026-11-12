"""Bước 4. Tinh chỉnh siêu tham số của cây, Random Forest và XGBoost bằng tìm kiếm ngẫu nhiên.
Đầu vào:  data/sach (mau_v1, mau_v2, chia_du_lieu)
Đầu ra:   ket_qua/b4_cac_bo_da_thu_<v>.csv, b4_tham_so_tot_nhat_<v>.json
Mỗi mô hình thử bộ mặc định và 20 bộ bốc ngẫu nhiên (seed 42). Mỗi bộ được chấm bằng kiểm định chéo 4 vòng theo ngày
trên tập huấn luyện. Tập kiểm tra không được dùng ở đây."""
import json
import warnings

import pandas as pd
from sklearn.model_selection import ParameterSampler

from src import cau_hinh as CH, doc_du_lieu as DL, mo_hinh as MH

warnings.filterwarnings("ignore", message="Found unknown categories")
CH.tao_thu_muc()
SO_LAN_THU = 20

# (tên, không gian tìm kiếm, bộ mặc định)
MO_HINH = [
    ("Cây quyết định", {"max_depth": [3, 4, 6, 8, 10, 12, 16, None], "min_samples_leaf": [1, 5, 10, 20, 50]},
     {"max_depth": 8, "min_samples_leaf": 20}),
    ("Random Forest", {"n_estimators": [200, 300, 500], "max_depth": [None, 10, 15, 20, 30],
                       "min_samples_leaf": [1, 2, 5, 10, 20], "max_features": [0.3, 0.5, 0.7, 1.0]},
     {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "max_features": 0.5}),
    ("XGBoost", {"n_estimators": [300, 600, 1000], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [3, 4, 6, 8],
                 "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10, 20]},
     {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
      "min_child_weight": 5}),
]

for pb in CH.PHIEN_BAN:
    df = DL.doc_mau(pb)
    df = df[df["tap"] == "huan_luyen"].reset_index(drop=True)
    print(f"\n{pb}: tập huấn luyện {len(df)} mẫu")
    dong, tot_nhat = [], {}
    for ten, khong_gian, mac_dinh in MO_HINH:
        cac_bo = [mac_dinh] + list(ParameterSampler(khong_gian, n_iter=SO_LAN_THU, random_state=CH.SEED))
        mae_tot = None
        for i, p in enumerate(cac_bo):
            nhan = "mặc định" if i == 0 else f"thử {i}"
            bang, _ = MH.kiem_dinh_cheo(MH.tao_tu_tham_so(ten, p), df, df["phan"].to_numpy(), ten=ten,
                                        tran=CH.TRAN_DU_BAO[pb], in_ra=False)
            mae = round(bang["MAE"].mean(), 4)
            dong.append({"mo_hinh": ten, "lan_thu": nhan, **{k: str(v) for k, v in p.items()}, "MAE_tb": mae,
                         "MAE_dlc": round(bang["MAE"].std(ddof=0), 4)})
            print(f"   {ten:<15} {nhan:<9} MAE {mae:.3f}   {p}")
            if i == 0:
                mae_mac_dinh = mae
            if mae_tot is None or mae < mae_tot:           # MAE nhỏ nhất, hòa thì giữ bộ thử trước
                mae_tot = mae
                tot_nhat[ten] = {"tham_so": p, "lan_thu": nhan, "MAE_kiem_dinh_cheo": mae, "MAE_mac_dinh": mae_mac_dinh}
        print(f"   -> {ten}: chọn {tot_nhat[ten]['lan_thu']}, MAE {tot_nhat[ten]['MAE_kiem_dinh_cheo']:.3f}\n")
    pd.DataFrame(dong).to_csv(CH.KET_QUA / f"b4_cac_bo_da_thu_{pb}.csv", index=False, encoding="utf-8-sig")
    with open(CH.KET_QUA / f"b4_tham_so_tot_nhat_{pb}.json", "w", encoding="utf-8") as f:
        json.dump(tot_nhat, f, ensure_ascii=False, indent=2)

# Hình: MAE kiểm định chéo của 21 bộ siêu tham số (v2) theo từng mô hình
import matplotlib.pyplot as plt

CH.thiet_lap_hinh()
bang = pd.read_csv(CH.KET_QUA / "b4_cac_bo_da_thu_v2.csv")
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (ten, _, _) in enumerate(MO_HINH):
    b = bang[bang["mo_hinh"] == ten]
    thu, mac_dinh = b[b["lan_thu"] != "mặc định"], b[b["lan_thu"] == "mặc định"]["MAE_tb"].iloc[0]
    tot = b["MAE_tb"].min()
    ax.scatter([i] * len(thu), thu["MAE_tb"], color=CH.XANH_NHAT, s=40, label="20 bộ thử" if i == 0 else None)
    ax.scatter([i], [mac_dinh], color=CH.XAM, s=110, marker="s", label="Bộ mặc định" if i == 0 else None, zorder=3)
    ax.scatter([i], [tot], color=CH.DO, s=130, marker="*", label="Bộ được chọn" if i == 0 else None, zorder=3)
    ax.text(i + 0.12, tot - 0.03, CH.so_vn(tot, 2), va="top", color=CH.DO)
    ax.text(i + 0.12, mac_dinh + 0.03, CH.so_vn(mac_dinh, 2), va="bottom", color=CH.XAM)
ax.set_xticks(range(len(MO_HINH)))
ax.set_xticklabels([m[0] for m in MO_HINH])
ax.set_xlim(-0.5, len(MO_HINH) - 0.4)
ax.set_ylim(8.85, 10.6)   # một số bộ có MAE cao hơn 10,6 nằm ngoài khung hình
ax.set_ylabel("MAE kiểm định chéo (km/h)")
ax.set_title("Kết quả tinh chỉnh siêu tham số trên tập huấn luyện của v2")
ax.legend(loc="upper center", fontsize=10, ncol=3)
CH.luu_hinh(fig, "b4_tinh_chinh.png")
