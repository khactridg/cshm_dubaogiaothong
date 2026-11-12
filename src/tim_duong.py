"""Từ tốc độ dự báo đến thời gian đi và tìm đường.
1. Dự báo tốc độ của MỌI đoạn đường ở từng khung 30 phút bằng mô hình đã lưu trong mo_hinh/v2.
2. Thời gian đi phụ thuộc thời điểm: xe vào đoạn lúc nào thì đi với tốc độ của khung lúc đó; đang đi mà sang khung mới
   thì phần còn lại của đoạn đi với tốc độ của khung mới (mô hình tốc độ theo bậc của Ichoua, Gendreau và Potvin, 2003).
   Cách tính này bảo đảm tính chất FIFO (vào trước thì ra trước). Tốc độ được chặn dưới 3 km/h.
3. Đường nhanh nhất: Dijkstra phụ thuộc thời gian, nhãn của mỗi nút là thời điểm tới sớm nhất (Kaufman và Smith, 1993).
   Đường ngắn nhất: Dijkstra thường theo chiều dài; thời gian của nó cũng tính theo cách ở bước 2."""
import heapq
import math

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from . import cau_hinh as CH, dac_trung as DT, doc_du_lieu as DL

TOC_DO_TOI_THIEU = 3.0
KHUNG_PHUT = 30
MAC_DINH = "Random Forest"
MO_HINH_TIM_DUONG = ["Random Forest", "XGBoost", "Hồi quy tuyến tính"]
R_TRAI_DAT = 6_371_000


class TimDuong:
    """Giữ đồ thị, các mô hình đã nạp và tốc độ đã dự báo (dựng một lần, dùng nhiều lần)."""

    def __init__(self, phien_ban="v2"):
        self.pb = phien_ban
        self.doan = DL.doc_doan().dropna(subset=["lat_s", "lat_e"]).reset_index(drop=True)
        self.vi_tri = pd.Index(self.doan["segment_id"])
        self.chieu_dai_km = self.doan["length"].to_numpy(dtype=float) / 1000

        # đồ thị có hướng; MultiDiGraph vì có vài cặp nút được nối bởi hai đoạn
        self.do_thi = nx.MultiDiGraph()
        for r in self.doan.itertuples():
            self.do_thi.add_node(int(r.s_node_id), lat=float(r.lat_s), lon=float(r.lon_s))
            self.do_thi.add_node(int(r.e_node_id), lat=float(r.lat_e), lon=float(r.lon_e))
        for r in self.doan.itertuples():
            self.do_thi.add_edge(int(r.s_node_id), int(r.e_node_id), key=int(r.segment_id), length=float(r.length))
        self.lien_thong = max(nx.strongly_connected_components(self.do_thi), key=len)
        self.nut_chon = np.array(sorted(self.lien_thong))
        toa_do = [[self.do_thi.nodes[n]["lat"], self.do_thi.nodes[n]["lon"]] for n in self.nut_chon]
        self.cay_nut = BallTree(np.radians(np.array(toa_do)), metric="haversine")

        # danh sách kề cho Dijkstra: nút -> [(nút kế tiếp, vị trí đoạn)]
        self.ke = {}
        for vt, (u, v) in enumerate(zip(self.doan["s_node_id"], self.doan["e_node_id"])):
            self.ke.setdefault(int(u), []).append((int(v), vt))

        # tập huấn luyện (để tính đặc trưng lịch sử) và số mẫu lịch sử của từng đoạn
        mau = DL.doc_mau(phien_ban)
        self.huan_luyen = mau[mau["tap"] == "huan_luyen"].reset_index(drop=True)
        so_mau = self.huan_luyen.groupby("segment_id").size()
        self.doan["so_mau_lich_su"] = self.doan["segment_id"].map(so_mau).fillna(0).astype(int)
        self.mo_hinh = {}
        self.bo_nho = {}      # (mô hình, thứ, khung) -> tốc độ dự báo của mọi đoạn

    # ---------------------------------------------------------------- dự báo tốc độ
    def lay_mo_hinh(self, ten):
        if ten not in self.mo_hinh:
            self.mo_hinh[ten] = joblib.load(CH.MO_HINH / self.pb / (CH.TEP_MO_HINH[ten] + ".joblib"))
        return self.mo_hinh[ten]

    def du_bao_toc_do(self, ten, thu, gio):
        """Tốc độ dự báo (km/h) của mọi đoạn ở khung 30 phút chứa thời điểm `gio` (7:40 thuộc khung 7:30)."""
        khung = math.floor(gio * 60 / KHUNG_PHUT) * KHUNG_PHUT / 60
        khoa = (ten, int(thu), khung)
        if khoa not in self.bo_nho:
            f = self.doan.copy()
            f["hour_vn"] = float(khung)
            f["weekday"] = int(thu)
            f["is_weekend"] = int(thu >= 5)
            X = pd.concat([DT.dac_trung_nen(f, self.doan), DT.dac_trung_lich_su(self.huan_luyen, self.huan_luyen["speed"], df_ap=f)], axis=1)
            du_bao = self.lay_mo_hinh(ten).predict(X)
            self.bo_nho[khoa] = np.clip(np.asarray(du_bao, dtype=float), 0, CH.TRAN_DU_BAO[self.pb])
        return self.bo_nho[khoa]

    def toc_do_luc(self, ten, thu, phut):
        """Tốc độ của khung chứa thời điểm `phut` (tính từ 0:00 ngày khởi hành; qua 24:00 là sang thứ kế tiếp)."""
        ngay = int(phut // (24 * 60))
        return self.du_bao_toc_do(ten, (int(thu) + ngay) % 7, (phut - ngay * 24 * 60) / 60)

    # ---------------------------------------------------------------- thời gian đi
    def qua_doan(self, vt, vao_phut, ten, thu):
        """Thời điểm ra khỏi đoạn ở vị trí vt nếu vào lúc vao_phut (mô hình tốc độ theo bậc)."""
        con_lai = self.chieu_dai_km[vt]
        t = float(vao_phut)
        while True:
            v = max(float(self.toc_do_luc(ten, thu, t)[vt]), TOC_DO_TOI_THIEU)
            het_khung = (math.floor(t / KHUNG_PHUT) + 1) * KHUNG_PHUT
            di_duoc = v * (het_khung - t) / 60            # km đi được tới hết khung
            if di_duoc >= con_lai:
                return t + con_lai / v * 60
            con_lai = con_lai - di_duoc
            t = het_khung

    def di_theo_lo_trinh(self, ma_doan, khoi_hanh_phut, ten, thu):
        """Thời điểm vào và ra của từng đoạn khi đi lần lượt theo một lộ trình."""
        vao, ra = [], []
        t = float(khoi_hanh_phut)
        for vt in self.vi_tri.get_indexer(ma_doan):
            vao.append(t)
            t = self.qua_doan(int(vt), t, ten, thu)
            ra.append(t)
        return np.array(vao), np.array(ra)

    # ---------------------------------------------------------------- tìm đường
    def nut_gan_nhat(self, lat, lon):
        """Nút giao gần nhất (trong thành phần liên thông mạnh lớn nhất) và khoảng cách tới nó (m)."""
        kc, i = self.cay_nut.query(np.radians([[lat, lon]]), k=1)
        return int(self.nut_chon[i[0, 0]]), float(kc[0, 0] * R_TRAI_DAT)

    def dijkstra_theo_gio(self, nguon, khoi_hanh_phut, ten, thu, dich):
        """Dijkstra phụ thuộc thời gian từ nút nguon tới nút dich. Trả về (thời điểm tới nút dich, đoạn cuối dẫn tới mỗi nút)."""
        toi = {nguon: float(khoi_hanh_phut)}
        qua = {}
        xong = set()
        hang = [(float(khoi_hanh_phut), nguon)]
        while hang:
            t, u = heapq.heappop(hang)                    # nút có thời điểm tới nhỏ nhất
            if u in xong:
                continue
            xong.add(u)
            if u == dich:
                break
            for v, vt in self.ke.get(u, []):
                if v in xong:
                    continue
                tv = self.qua_doan(vt, t, ten, thu)
                if tv < toi.get(v, math.inf):
                    toi[v] = tv
                    qua[v] = vt
                    heapq.heappush(hang, (tv, v))
        return toi[dich], qua

    def dung_lo_trinh(self, qua, nguon, dich):
        """Dãy mã đoạn từ nguon tới dich, dựng ngược từ bảng "đoạn cuối dẫn tới mỗi nút"."""
        ma, nut = [], dich
        while nut != nguon:
            vt = qua[nut]
            ma.append(int(self.doan["segment_id"].iloc[vt]))
            nut = int(self.doan["s_node_id"].iloc[vt])
        return ma[::-1]

    def doan_ngan_nhat(self, nguon, dich):
        """Dãy mã đoạn của đường ngắn nhất theo chiều dài."""
        nut = nx.dijkstra_path(self.do_thi, nguon, dich, weight="length")
        ma = []
        for u, v in zip(nut[:-1], nut[1:]):
            canh = self.do_thi[u][v]                      # {mã đoạn: thuộc tính}, có thể có hai đoạn song song
            ma.append(min(canh, key=lambda k: canh[k]["length"]))
        return ma

    def mo_ta(self, ma_doan, khoi_hanh_phut, ten, thu):
        """Chiều dài, thời gian dự báo, tọa độ và bảng chi tiết của một lộ trình."""
        ct = self.doan.iloc[self.vi_tri.get_indexer(ma_doan)][["segment_id", "street_name", "length", "max_velocity", "lat_s", "lon_s",
                                                                  "lat_e", "lon_e", "so_mau_lich_su"]].copy()
        vao, ra = self.di_theo_lo_trinh(ma_doan, khoi_hanh_phut, ten, thu)
        ct["phut_vao"] = vao
        ct["phut"] = ra - vao
        ct["toc_do_du_bao"] = ct["length"] / 1000 / (ct["phut"] / 60).clip(lower=1e-9)     # tốc độ áp dụng trên đoạn
        ct["LOS_du_bao"] = DL.quy_ra_los(ct["toc_do_du_bao"], ct["max_velocity"])
        toa_do = [(ct["lat_s"].iloc[0], ct["lon_s"].iloc[0])] + list(zip(ct["lat_e"], ct["lon_e"]))
        return {"km": float(ct["length"].sum() / 1000), "phut": float(ra[-1] - khoi_hanh_phut), "so_doan": len(ct),
                "ty_le_dai_co_lich_su": float(ct.loc[ct["so_mau_lich_su"] > 0, "length"].sum() / max(ct["length"].sum(), 1)),
                "toa_do": toa_do, "chi_tiet": ct.reset_index(drop=True)}

    def tim(self, di, den, thu, gio, ten=MAC_DINH):
        """di, den: (vĩ độ, kinh độ); gio: giờ khởi hành (7.75 = 7 giờ 45). Trả về đường nhanh nhất theo dự báo và đường ngắn nhất."""
        s, lech_di = self.nut_gan_nhat(*di)
        d, lech_den = self.nut_gan_nhat(*den)
        if s == d:
            raise ValueError("Điểm đi và điểm đến rơi vào cùng một nút giao, hãy chọn hai điểm xa nhau hơn.")
        khoi_hanh = float(gio) * 60
        toi, qua = self.dijkstra_theo_gio(s, khoi_hanh, ten, thu, d)
        return {"mo_hinh": ten, "lech_di_m": lech_di, "lech_den_m": lech_den, "nhan_nut_den_phut": toi - khoi_hanh,
                "nhanh_nhat": self.mo_ta(self.dung_lo_trinh(qua, s, d), khoi_hanh, ten, thu),
                "ngan_nhat": self.mo_ta(self.doan_ngan_nhat(s, d), khoi_hanh, ten, thu)}


def tom_tat_theo_duong(ct):
    """Gộp các đoạn liên tiếp cùng tên đường: giờ vào, tên đường, chiều dài, thời gian, tốc độ trung bình."""
    ten = ct["street_name"].fillna("(đường không tên)")
    nhom = (ten != ten.shift()).cumsum()
    g = ct.assign(ten=ten, nhom=nhom).groupby("nhom", sort=True)
    dai, phut = g["length"].sum(), g["phut"].sum()
    gio_vao = [f"{int(m) // 60 % 24}:{int(m) % 60:02d}" for m in g["phut_vao"].first().round()]
    return pd.DataFrame({"Vào lúc": gio_vao, "Đường": g["ten"].first(), "Dài (m)": dai.round(0).astype(int),
                         "Thời gian (phút)": phut.round(1), "Tốc độ dự báo (km/h)": (dai / 1000 / (phut / 60)).round(0)}).reset_index(drop=True)
