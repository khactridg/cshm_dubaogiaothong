"""Bước 5. Huấn luyện các mô hình trên toàn tập huấn luyện, chấm trên tập kiểm tra (30 ngày), kiểm định chéo cấu hình cuối.
Đầu vào:  data/sach, ket_qua/b4_tham_so_tot_nhat_<v>.json
Đầu ra:   mo_hinh/<v>/*.joblib
          ket_qua/b5_ket_qua_test.csv (v2), b5_v1_v2.csv, b5_kiem_dinh_cheo.csv
          ket_qua/b5_mae_cac_mo_hinh.png, b5_v1_so_voi_v2.png
Cả mô hình học từ v1 lẫn từ v2 đều được chấm trên cùng tập kiểm tra, với đáp án lấy theo v2."""
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import cau_hinh as CH, dac_trung as DT, doc_du_lieu as DL, mo_hinh as MH

warnings.filterwarnings("ignore", message="Found unknown categories")
CH.tao_thu_muc()
CH.thiet_lap_hinh()
TEP = CH.TEP_MO_HINH

v2 = DL.doc_mau("v2")
kiem_tra = v2[v2["tap"] == "test"].reset_index(drop=True)
print(f"Tập kiểm tra: {len(kiem_tra)} mẫu, {kiem_tra['date_vn'].nunique()} ngày")

# 1. Huấn luyện và chấm
dong = []
for pb in CH.PHIEN_BAN:
    df = DL.doc_mau(pb)
    hl = df[df["tap"] == "huan_luyen"].reset_index(drop=True)
    X_hl = DT.dac_trung_huan_luyen(hl)
    X_kt = DT.dac_trung_cho(hl, kiem_tra)
    (CH.MO_HINH / pb).mkdir(parents=True, exist_ok=True)
    print(f"\n{pb}: huấn luyện trên {len(hl)} mẫu")
    for ten, m in MH.tao_bo_mo_hinh(pb).items():
        m.fit(X_hl, hl["speed"])
        joblib.dump(m, CH.MO_HINH / pb / (TEP[ten] + ".joblib"))
        du_bao = np.clip(m.predict(X_kt), 0, CH.TRAN_DU_BAO[pb])
        cs = MH.chi_so(kiem_tra["speed"], du_bao, kiem_tra["max_velocity"])
        dong.append({"phien_ban": pb, "mo_hinh": ten, "so_mau_test": len(kiem_tra), **{k: round(float(v), 4) for k, v in cs.items()}})
        print(f"   {ten:<20} MAE {cs['MAE']:.2f}  RMSE {cs['RMSE']:.2f}  R2 {cs['R2']:.3f}  LOS đúng {cs['los_dung']:.3f}")
tat_ca = pd.DataFrame(dong)

# 2. Bảng kết quả chính (mô hình học từ v2) kèm mốc "luôn đoán mức phổ biến nhất"
chinh = tat_ca[tat_ca["phien_ban"] == "v2"].drop(columns="phien_ban")
los_that = DL.quy_ra_los(kiem_tra["speed"], kiem_tra["max_velocity"])
pho_bien = pd.Series(los_that).value_counts().idxmax()
lech = np.abs(DL.bac_los(np.full(len(kiem_tra), pho_bien, dtype=object)) - DL.bac_los(los_that))
moc_a = {"mo_hinh": f"Luôn đoán mức {pho_bien}", "so_mau_test": len(kiem_tra), "los_dung": round(float((lech == 0).mean()), 4),
         "los_lech_1": round(float((lech <= 1).mean()), 4),
         "macro_F1": round(float(f1_score(los_that.astype(str), np.full(len(kiem_tra), pho_bien), average="macro", labels=CH.MUC_LOS, zero_division=0)), 4)}
chinh = pd.concat([chinh, pd.DataFrame([moc_a])], ignore_index=True)
chinh.to_csv(CH.KET_QUA / "b5_ket_qua_test.csv", index=False, encoding="utf-8-sig")
print("\n", chinh.to_string(index=False))

# 3. v1 so với v2
v1_v2 = tat_ca.pivot(index="mo_hinh", columns="phien_ban", values="MAE").loc[list(TEP)]
v1_v2.columns = ["MAE_v1", "MAE_v2"]
v1_v2["chenh"] = (v1_v2["MAE_v2"] - v1_v2["MAE_v1"]).round(4)
v1_v2.reset_index().to_csv(CH.KET_QUA / "b5_v1_v2.csv", index=False, encoding="utf-8-sig")
print("\n", v1_v2.to_string())

# 4. Kiểm định chéo 4 vòng của cấu hình cuối (v2, chỉ tập huấn luyện)
hl = v2[v2["tap"] == "huan_luyen"].reset_index(drop=True)
cac_vong = []
for ten, m in MH.tao_bo_mo_hinh("v2").items():
    bang, _ = MH.kiem_dinh_cheo(m, hl, hl["phan"].to_numpy(), ten=ten)
    cac_vong.append(bang)
kdc = MH.chi_so_theo_tron(pd.concat(cac_vong), "mo_hinh")
kdc.to_csv(CH.KET_QUA / "b5_kiem_dinh_cheo.csv", index=False, encoding="utf-8-sig")
print("\n", kdc[["mo_hinh", "MAE_tb", "MAE_dlc", "R2_tb"]].to_string(index=False))

# 5. Hình MAE các mô hình
ten = list(TEP)
mae = chinh.set_index("mo_hinh").loc[ten, "MAE"]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(ten, mae, color=[CH.XAM, CH.XAM] + [CH.XANH] * 4)
for i, v in enumerate(mae):
    ax.text(v + 0.15, i, CH.so_vn(v), va="center")
ax.invert_yaxis()
ax.set_xlabel("MAE trên tập kiểm tra (km/h)")
ax.set_title("Sai số tốc độ của các mô hình (màu xám: hai mốc so sánh)")
ax.set_xlim(0, mae.max() * 1.12)
CH.luu_hinh(fig, "b5_mae_cac_mo_hinh.png")

# 6. Hình v1 so với v2 (hai mốc không học từ dữ liệu làm sạch nên không vẽ)
ml = ten[2:]
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, t in enumerate(ml):
    a, b = v1_v2.loc[t, "MAE_v1"], v1_v2.loc[t, "MAE_v2"]
    ax.plot([a, b], [i, i], color=CH.XAM, linewidth=2, zorder=1)
    ax.scatter([a], [i], color=CH.XAM, s=80, zorder=2, label="Học từ v1" if i == 0 else None)
    ax.scatter([b], [i], color=CH.XANH, s=80, zorder=2, label="Học từ v2" if i == 0 else None)
    ax.text(a, i - 0.22, CH.so_vn(a), ha="center", fontsize=10)
    ax.text(b, i + 0.38, CH.so_vn(b), ha="center", fontsize=10)
ax.set_yticks(range(len(ml)))
ax.set_yticklabels(ml)
ax.invert_yaxis()
ax.set_xlabel("MAE trên cùng tập kiểm tra, đáp án theo v2 (km/h)")
ax.set_title("Mô hình học từ dữ liệu v1 và từ dữ liệu v2")
ax.legend(loc="lower right")
CH.luu_hinh(fig, "b5_v1_so_voi_v2.png")
