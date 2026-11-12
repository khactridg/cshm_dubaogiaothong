"""
graph.py - Mạng đường dưới dạng đồ thị có hướng.

Đỉnh = nút giao (có tọa độ), cạnh = đoạn đường đi từ s_node_id tới e_node_id.
Dùng MultiDiGraph (cho phép nhiều cạnh giữa hai đỉnh, khóa cạnh = segment_id) vì có vài cặp nút
được nối bởi hai đoạn khác nhau; đồ thị thường (DiGraph) sẽ làm mất một trong hai đoạn.
"""
import networkx as nx
import pandas as pd


def clean_value(x):
    """Giá trị trống (NaN) đổi thành None để lưu vào đồ thị."""
    if pd.isna(x):
        return None
    return x


def build_graph(seg):
    """Dựng đồ thị từ bảng đoạn đường (data.load_segments)."""
    G = nx.MultiDiGraph()
    for row in seg.itertuples(index=False):              # nút đầu của các đoạn
        if int(row.s_node_id) not in G:
            G.add_node(int(row.s_node_id), lat=float(row.lat_s), lon=float(row.lon_s))
    for row in seg.itertuples(index=False):              # nút cuối chưa có trong đồ thị
        if int(row.e_node_id) not in G:
            G.add_node(int(row.e_node_id), lat=float(row.lat_e), lon=float(row.lon_e))
    for row in seg.itertuples(index=False):
        G.add_edge(int(row.s_node_id), int(row.e_node_id), key=int(row.segment_id),
                   segment_id=int(row.segment_id), length=float(row.length), street_id=int(row.street_id),
                   street_name=clean_value(row.street_name), street_type=clean_value(row.street_type),
                   street_level=int(row.street_level), max_velocity=clean_value(row.max_velocity))
    return G


def largest_scc(G):
    """Thành phần liên thông mạnh lớn nhất: trong đó từ mọi đỉnh đều đi được tới mọi đỉnh."""
    return max(nx.strongly_connected_components(G), key=len)


def graph_stats(G, labeled_segments):
    """Các con số mô tả đồ thị. labeled_segments: các segment_id có nhãn."""
    n_edges = G.number_of_edges()
    pairs = set()
    for u, v in G.edges():
        pairs.add((u, v))
    reverse = 0
    for u, v in G.edges():
        if (v, u) in pairs:
            reverse = reverse + 1
    big = largest_scc(G)
    edges_in_big = set()
    for u, v, key in G.edges(keys=True):
        if u in big and v in big:
            edges_in_big.add(key)
    labeled = set(int(s) for s in labeled_segments)
    return {
        "so_dinh": G.number_of_nodes(),
        "so_canh": n_edges,
        "so_cap_nut_noi_boi_2_doan": n_edges - len(pairs),
        "ty_le_doan_co_chieu_nguoc_lai": reverse / n_edges,
        "tplt_manh_lon_nhat_ty_le_dinh": len(big) / G.number_of_nodes(),
        "ty_le_doan_co_nhan_trong_tplt_manh_lon_nhat": len(labeled & edges_in_big) / len(labeled),
    }
