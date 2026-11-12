"""
routing.py - Dự báo tốc độ cho toàn mạng đường và tìm đường.

Với một truy vấn (điểm đi, điểm đến, thứ, giờ khởi hành, mô hình):
  1. Dự báo tốc độ cho MỌI đoạn của mạng đường ở từng khung 30 phút, bằng mô hình đã lưu ở models/<v>.
     Đặc trưng tính giống hệt lúc huấn luyện (nhóm C học từ tập huấn luyện). Đoạn chưa từng có số liệu thì
     nhóm C bằng trung bình chung. Chỉ dự báo những khung mà chuyến đi đi qua, có lưu đệm.
  2. Thời gian đi phụ thuộc thời điểm: xe vào đoạn lúc nào thì dùng tốc độ của khung lúc đó; đang đi mà sang khung
     mới thì phần còn lại của đoạn đi với tốc độ của khung mới (Ichoua, Gendreau và Potvin, 2003). Cách tính này
     bảo đảm "vào trước thì ra trước" (FIFO). Tốc độ chặn dưới 3 km/h.
  3. Đưa điểm người dùng chọn về nút giao gần nhất trong thành phần liên thông mạnh lớn nhất.
  4. Đường NHANH NHẤT: Dijkstra phụ thuộc thời gian, nhãn của mỗi nút là thời điểm tới sớm nhất; với mạng có tính
     FIFO, cách này cho đúng đường tới sớm nhất (Kaufman và Smith, 1993).
     Đường NGẮN NHẤT: Dijkstra thường theo chiều dài; thời gian của nó tính theo cùng cách ở bước 2.

Ví dụ:
    from src import routing as R
    router = R.Router()
    res = router.route((10.7769, 106.7009), (10.8231, 106.6297), weekday=0, hour=8.0)
"""
import heapq
import json
import math

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from . import config as C, data as D, evaluate as E, features as F, graph as G

MIN_SPEED_KMH = 3.0
DEFAULT_MODEL = "Random Forest (tinh chỉnh)"
DEMO_MODELS = ["Random Forest (tinh chỉnh)", "XGBoost (tinh chỉnh)", "Hồi quy tuyến tính",
               "Mốc: lịch sử đoạn", "Mốc: trung bình"]
EARTH_RADIUS_M = 6_371_000
SLOT_MIN = 30                      # độ dài một khung (phút)


class Router:
    """Giữ đồ thị, bảng đoạn đường, các mô hình đã nạp và tốc độ đã dự báo (dựng một lần, dùng nhiều lần)."""

    def __init__(self, verbose=True):
        self.seg = D.load_segments().dropna(subset=["lat_s", "lat_e"]).reset_index(drop=True)
        self.seg_index = pd.Index(self.seg["segment_id"])
        self.graph = G.build_graph(self.seg)
        self.scc = G.largest_scc(self.graph)

        # cây tìm nút gần nhất, chỉ gồm các nút trong thành phần liên thông mạnh lớn nhất
        self.snap_nodes = np.array(sorted(self.scc))
        coords = []
        for n in self.snap_nodes:
            coords.append([self.graph.nodes[n]["lat"], self.graph.nodes[n]["lon"]])
        self.snap_tree = BallTree(np.radians(np.array(coords)), metric="haversine")

        # đặc trưng nhóm B (không đổi theo giờ) và tập huấn luyện để tính nhóm C
        self.road = F.road_features(self.seg)
        self.train, _ = E.train_part(E.load_samples())
        n_hist = self.train.groupby("segment_id").size()
        self.seg["so_mau_lich_su"] = self.seg["segment_id"].map(n_hist).fillna(0).astype(int)

        # danh sách mô hình đã lưu
        with open(C.MODELS_DIR / "danh_sach_mo_hinh.json", encoding="utf-8") as f:
            catalog = json.load(f)
        self.model_files = {}
        for item in catalog:
            self.model_files[item["ten"]] = C.MODELS_DIR / item["tep"]
        self.models = {}               # mô hình đã nạp
        self.speed_cache = {}          # (mô hình, thứ, khung) -> tốc độ dự báo của mọi đoạn

        # danh sách kề cho Dijkstra: nút -> [(nút kế tiếp, vị trí đoạn trong self.seg)]
        self.length_km = self.seg["length"].to_numpy(dtype=float) / 1000
        self.adj = {}
        s_nodes = self.seg["s_node_id"].to_numpy()
        e_nodes = self.seg["e_node_id"].to_numpy()
        for pos in range(len(self.seg)):
            u = int(s_nodes[pos])
            if u not in self.adj:
                self.adj[u] = []
            self.adj[u].append((int(e_nodes[pos]), pos))
        if verbose:
            print(f"Router sẵn sàng: {len(self.seg)} đoạn, {self.graph.number_of_nodes()} nút, "
                  f"thành phần liên thông mạnh lớn nhất {len(self.scc)} nút")

    # ---------------------------------------------------------------- mô hình và dự báo
    def available_models(self):
        """Các mô hình demo đã được lưu (theo thứ tự DEMO_MODELS)."""
        result = []
        for name in DEMO_MODELS:
            if name in self.model_files and self.model_files[name].exists():
                result.append(name)
        return result

    def get_model(self, name):
        if name not in self.models:
            self.models[name] = joblib.load(self.model_files[name])
        return self.models[name]

    def features(self, weekday, hour):
        """Bảng đặc trưng của mọi đoạn ở (thứ, giờ), cùng cách tính như lúc huấn luyện."""
        frame = self.seg[["segment_id", "street_id"]].copy()
        frame["hour_vn"] = float(hour)
        frame["weekday"] = int(weekday)
        frame["is_weekend"] = int(weekday >= 5)
        time_features = F.time_features(frame)
        hist = F.history_features(self.train, self.train["speed"], df_apply=frame)
        return pd.concat([time_features, self.road, hist], axis=1)

    def predict_speeds(self, model_name, weekday, hour):
        """Tốc độ dự báo (km/h) của mọi đoạn ở khung 30 phút CHỨA thời điểm `hour` (7:40 thuộc khung 7:30)."""
        slot = math.floor(hour * 60 / SLOT_MIN) * SLOT_MIN / 60
        key = (model_name, int(weekday), slot)
        if key not in self.speed_cache:
            pred = self.get_model(model_name).predict(self.features(weekday, slot))
            self.speed_cache[key] = np.clip(np.asarray(pred, dtype=float), 0, C.SPEED_CAP_KMH)
        return self.speed_cache[key]

    def speeds_at(self, model_name, weekday, minute):
        """Tốc độ của khung chứa thời điểm `minute` (phút tính từ 0:00 ngày khởi hành; qua 24:00 là sang thứ sau)."""
        day = int(minute // (24 * 60))
        minute_in_day = minute - day * 24 * 60
        return self.predict_speeds(model_name, (int(weekday) + day) % 7, minute_in_day / 60)

    def traverse(self, pos, enter_min, model_name, weekday):
        """Thời điểm ra khỏi đoạn ở vị trí `pos` nếu vào đoạn lúc `enter_min` (phút).
        Đi với tốc độ của khung hiện tại; chưa hết đoạn mà đã sang khung mới thì phần còn lại
        đi với tốc độ của khung mới."""
        remain = self.length_km[pos]                     # km còn phải đi
        t = float(enter_min)
        while True:
            v = max(float(self.speeds_at(model_name, weekday, t)[pos]), MIN_SPEED_KMH)    # km/h
            slot_end = (math.floor(t / SLOT_MIN) + 1) * SLOT_MIN
            reach = v * (slot_end - t) / 60                                                # km đi được tới hết khung
            if reach >= remain:
                return t + remain / v * 60
            remain = remain - reach
            t = slot_end

    def walk(self, segment_ids, depart_min, model_name, weekday):
        """Đi lần lượt theo một lộ trình: thời điểm vào và ra của từng đoạn (phút)."""
        positions = self.seg_index.get_indexer(segment_ids)
        enter = []
        leave = []
        t = float(depart_min)
        for p in positions:
            enter.append(t)
            t = self.traverse(int(p), t, model_name, weekday)
            leave.append(t)
        return np.array(enter), np.array(leave)

    # ---------------------------------------------------------------- tìm đường
    def snap(self, lat, lon):
        """Nút giao gần nhất (trong thành phần liên thông mạnh lớn nhất) và khoảng cách tới nó (m)."""
        dist, idx = self.snap_tree.query(np.radians([[lat, lon]]), k=1)
        return int(self.snap_nodes[idx[0, 0]]), float(dist[0, 0] * EARTH_RADIUS_M)

    def td_dijkstra(self, source, depart_min, model_name, weekday, targets=None):
        """Dijkstra phụ thuộc thời gian: nhãn của một nút là thời điểm tới sớm nhất (phút).
        targets: tập nút đích, dừng khi đã chốt hết (None = tính cho mọi nút).
        Trả về (thời điểm tới của các nút đã chốt, đoạn cuối cùng dẫn tới mỗi nút)."""
        arrive = {source: float(depart_min)}
        via = {}
        done = set()
        remaining = None
        if targets is not None:
            remaining = set(targets) - {source}
        heap = [(float(depart_min), source)]
        while heap:
            t, u = heapq.heappop(heap)                   # nút có thời điểm tới nhỏ nhất
            if u in done:
                continue
            done.add(u)                                  # chốt nhãn của u
            if remaining is not None:
                remaining.discard(u)
                if len(remaining) == 0:
                    break
            for v, pos in self.adj.get(u, []):
                if v in done:
                    continue
                tv = self.traverse(pos, t, model_name, weekday)
                if tv < arrive.get(v, math.inf):
                    arrive[v] = tv
                    via[v] = pos
                    heapq.heappush(heap, (tv, v))
        result = {}
        for n in done:
            result[n] = arrive[n]
        return result, via

    def path_from(self, via, source, target):
        """Dãy mã đoạn từ source tới target, dựng ngược từ bảng "đoạn cuối dẫn tới mỗi nút"."""
        seg_ids = []
        node = target
        sid = self.seg["segment_id"].to_numpy()
        s_node = self.seg["s_node_id"].to_numpy()
        while node != source:
            pos = via[node]
            seg_ids.append(int(sid[pos]))
            node = int(s_node[pos])
        seg_ids.reverse()
        return seg_ids

    def path_segments(self, nodes, weight):
        """Đổi dãy nút thành dãy đoạn; hai nút có nhiều đoạn song song thì lấy đoạn có trọng số nhỏ nhất."""
        segs = []
        for i in range(len(nodes) - 1):
            edges = self.graph[nodes[i]][nodes[i + 1]]   # {segment_id: thuộc tính}
            best_key = None
            for key in edges:
                if best_key is None or edges[key][weight] < edges[best_key][weight]:
                    best_key = key
            segs.append(best_key)
        return segs

    def describe_path(self, segment_ids, depart_min, model_name, weekday):
        """Tổng chiều dài, tổng thời gian dự báo và bảng chi tiết của một lộ trình."""
        positions = self.seg_index.get_indexer(segment_ids)
        part = self.seg.iloc[positions][["segment_id", "street_name", "street_type", "length", "max_velocity",
                                         "lat_s", "lon_s", "lat_e", "lon_e", "so_mau_lich_su"]].copy()
        enter, leave = self.walk(segment_ids, depart_min, model_name, weekday)
        part["phut_vao"] = enter
        part["phut"] = leave - enter
        # tốc độ áp dụng trên đoạn = chiều dài / thời gian đi qua đoạn
        part["toc_do_du_bao"] = part["length"] / 1000 / (part["phut"] / 60).clip(lower=1e-9)
        part["LOS_du_bao"] = D.speed_to_los(part["toc_do_du_bao"], part["max_velocity"])
        coords = []
        if len(part) > 0:
            coords.append((part["lat_s"].iloc[0], part["lon_s"].iloc[0]))
            for lat, lon in zip(part["lat_e"], part["lon_e"]):
                coords.append((lat, lon))
        total_minutes = float(leave[-1] - depart_min) if len(part) > 0 else 0.0
        with_history = part.loc[part["so_mau_lich_su"] > 0, "length"].sum()
        return {
            "km": float(part["length"].sum() / 1000),
            "phut": total_minutes,
            "so_doan": int(len(part)),
            "ty_le_chieu_dai_co_lich_su": float(with_history / max(part["length"].sum(), 1)),
            "toa_do": coords,
            "chi_tiet": part.reset_index(drop=True),
        }

    def route(self, origin, destination, weekday, hour, model_name=DEFAULT_MODEL):
        """origin, destination: (vĩ độ, kinh độ); hour: giờ khởi hành (7.75 = 7 giờ 45).
        Trả về đường nhanh nhất theo dự báo và đường ngắn nhất."""
        s, ds = self.snap(origin[0], origin[1])
        t, dt = self.snap(destination[0], destination[1])
        if s == t:
            raise ValueError("Điểm đi và điểm đến rơi vào cùng một nút giao; hãy chọn hai điểm xa nhau hơn.")
        depart = float(hour) * 60
        out = {"mo_hinh": model_name, "thu": int(weekday), "gio": float(hour),
               "nut_di": s, "nut_den": t, "lech_diem_di_m": ds, "lech_diem_den_m": dt}
        _, via = self.td_dijkstra(s, depart, model_name, weekday, targets={t})
        out["nhanh_nhat"] = self.describe_path(self.path_from(via, s, t), depart, model_name, weekday)
        nodes = nx.dijkstra_path(self.graph, s, t, weight="length")
        out["ngan_nhat"] = self.describe_path(self.path_segments(nodes, "length"), depart, model_name, weekday)
        return out


def streets_summary(detail):
    """Gộp các đoạn liên tiếp cùng tên đường: giờ vào, tên, chiều dài, thời gian, tốc độ trung bình."""
    name = detail["street_name"].fillna("(đường không tên)")
    group_id = (name != name.shift()).cumsum()            # các đoạn liền nhau cùng tên có cùng số nhóm
    data = detail.copy()
    data["ten"] = name
    data["nhom"] = group_id
    g = data.groupby("nhom", sort=True)
    length = g["length"].sum()
    minutes = g["phut"].sum()
    start = g["phut_vao"].first().round().astype(int) % (24 * 60)
    start_text = []
    for m in start:
        start_text.append(f"{m // 60}:{m % 60:02d}")
    out = pd.DataFrame({
        "Vào lúc": start_text,
        "Đường": g["ten"].first(),
        "Dài (m)": length.round(0).astype(int),
        "Thời gian (phút)": minutes.round(1),
        "Tốc độ dự báo (km/h)": (length / 1000 / (minutes / 60)).round(0),
    })
    return out.reset_index(drop=True)
