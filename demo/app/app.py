"""
app.py - Giao diện demo: dự báo tình trạng giao thông và đề xuất tuyến đường tại TP.HCM.

Chạy từ thư mục gốc của dự án:   streamlit run app/app.py     (hoặc bấm đúp chay_demo.bat)
Cần: models/v2 (huan_luyen.py), data/processed/v2 (tao_du_lieu_sach.py, chia_du_lieu.py) và mạng Internet
để tải nền bản đồ, phông chữ và tra địa chỉ.
Giao diện (màu, phông chữ, bo góc) cấu hình ở .streamlit/config.toml; logo trường đặt ở app/assets/.

Cách dùng: chọn điểm đi, điểm đến (địa điểm có sẵn, gõ địa chỉ, hoặc bấm lên bản đồ), chọn thứ, giờ và cách chọn đường.
Ứng dụng vẽ lộ trình được chọn (tô màu theo tình trạng giao thông dự báo) và đường ngắn nhất (nét đứt xám) để so sánh.
"""
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import folium                                   # noqa: E402
import pandas as pd                             # noqa: E402
import streamlit as st                          # noqa: E402
from streamlit_folium import st_folium          # noqa: E402

from src import config as C, routing as R       # noqa: E402
import ui                                       # noqa: E402

st.set_page_config(page_title="Tìm đường TP.HCM", page_icon=":material/route:", layout="wide")
st.html(ui.CSS)

# ============================================================================= Hằng số
ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = ASSETS / "logo.png"
if not LOGO.exists():
    LOGO = None
# Nền bản đồ xám nhạt (Esri) để lộ trình màu nổi bật; vẫn có thể đổi sang OpenStreetMap ở góc bản đồ
ESRI_GRAY = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
ESRI_LABELS = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/"
               "MapServer/tile/{z}/{y}/{x}")
ESRI_ATTR = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"

WEEKDAYS = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
WEEKDAYS_SHORT = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
# Giờ được chọn: 6:00–22:30 (quyết định 03/11/2026). Từ 23h đến 6h tập huấn luyện gần như không có số liệu.
HOURS = []                                       # giờ khởi hành 6:00–22:30, bước 5 phút
for minute in range(6 * 60, 22 * 60 + 31, 5):
    HOURS.append(minute / 60)
PLACES = {                                       # tọa độ (vĩ độ, kinh độ) lấy đúng nút giao trong dữ liệu mạng đường
    # Điểm xuất phát: Trường ĐH Sư phạm TP.HCM
    "ĐH Sư phạm - cơ sở An Dương Vương (ngã tư Nguyễn Văn Cừ)": (10.76116, 106.68328),
    "ĐH Sư phạm - cơ sở Lê Văn Sỹ (ngã tư Trần Quang Diệu)": (10.78759, 106.67826),
    # Tình huống demo (E11): đi từ cơ sở An Dương Vương, thứ Hai, giờ ghi trong ngoặc; các cách chọn đường
    # đi những lộ trình khác hẳn nhau (kết quả ở results/v2/E11_B2_tinh_huong_tot_nhat.csv)
    "Tình huống 1 · Đường Tân Sơn Nhì, Tân Phú (7:30)": (10.79862, 106.63177),
    "Tình huống 2 · Quốc lộ 1, Quận 12 (7:30)": (10.85446, 106.63524),
    "Tình huống 3 · Đường Liên khu 10-11, Bình Tân (17:30)": (10.76804, 106.61787),
    "Tình huống 4 · Đường Phan Đăng Lưu, Phú Nhuận (7:30)": (10.79923, 106.68040),
    "Tình huống 5 · Đường Trần Đại Nghĩa, Bình Chánh (17:30)": (10.72244, 106.60017),
    # Giờ kẹt: đi từ cơ sở An Dương Vương lúc 17:30 thứ Sáu; theo Random Forest, 80–84% chiều dài lộ trình
    # ở mức "đông, chậm" (kiểm tra ngày 04/11/2026)
    "Giờ kẹt 1 · Đường Hồng Bàng, Quận 6 (T6 17:30)": (10.75421, 106.64398),
    "Giờ kẹt 2 · Đường Âu Cơ, Tân Bình (T6 17:30)": (10.78510, 106.64193),
    "Giờ kẹt 3 · Đường Vườn Lài, Tân Phú (T6 17:30)": (10.78780, 106.62664),
}
CLICKED = "(điểm bấm trên bản đồ)"
# Tra địa chỉ: dịch vụ Nominatim của OpenStreetMap (miễn phí, cần Internet, tối đa 1 yêu cầu mỗi giây).
GEOCODER_URL = "https://nominatim.openstreetmap.org/search"
GEOCODER_AGENT = "demo-du-bao-giao-thong-tphcm/1.0"           # Nominatim yêu cầu tên ứng dụng gọi tới
HCMC_VIEWBOX = "106.35,11.20,107.05,10.35"                    # trái, trên, phải, dưới: khung TP.HCM

LOS_COLORS = {"A": "#1a9850", "B": "#91cf60", "C": "#d9ef8b", "D": "#fee08b", "E": "#fc8d59", "F": "#d73027"}
LOS_TEXT = {"A": "thông thoáng", "B": "khá thông thoáng", "C": "ổn định", "D": "hơi đông",
            "E": "đông, chậm", "F": "kẹt"}
# Lựa chọn hiển thị -> tên mô hình đã lưu (None = đường ngắn nhất, không cần mô hình để chọn đường)
CHOICES = {"XGBoost": "XGBoost (tinh chỉnh)", "Random Forest": "Random Forest (tinh chỉnh)",
           "Hồi quy tuyến tính": "Hồi quy tuyến tính", "Đường ngắn nhất": None}
DEFAULT_CHOICE = "Random Forest"
FAR_SNAP_M = 300                                 # điểm chọn cách nút giao gần nhất hơn mức này thì cảnh báo
MAP_HEIGHT = 480


# ============================================================================= Tiện ích
def fmt_hour(h):
    m = int(round(h * 60))
    return f"{m // 60}:{m % 60:02d}"


def weekday_label(i):
    return WEEKDAYS_SHORT[i]


def fmt_num(x, digits=1):
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_resource(show_spinner="Đang nạp mạng đường và mô hình (chỉ lần đầu, khoảng 5–10 giây)...")
def load_router():
    return R.Router(verbose=False)


@st.cache_data
def test_errors():
    """Sai số trên tập test của từng mô hình (E08), để nhắc người xem rằng dự báo có sai số."""
    path = C.RESULTS / "E08_B1_chi_so_tong.csv"
    if not path.exists():
        return {}
    t = pd.read_csv(path, encoding="utf-8-sig")
    t = t[t["hoc_tu"] == C.VERSION]
    errors = {}
    for r in t.itertuples():
        errors[r.mo_hinh] = (r.MAE, r.sai_so_thoi_gian_phut_moi_km)
    return errors


def _nominatim(query: str, bounded: bool):
    """Một lần gọi Nominatim. Lỗi mạng thì ném ngoại lệ."""
    params = {"q": query, "format": "jsonv2", "limit": 5, "countrycodes": "vn", "accept-language": "vi"}
    if bounded:
        params.update({"viewbox": HCMC_VIEWBOX, "bounded": 1})
    req = urllib.request.Request(f"{GEOCODER_URL}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": GEOCODER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [(d["display_name"], float(d["lat"]), float(d["lon"])) for d in data]


def _in_hcmc(lat: float, lon: float) -> bool:
    left, top, right, bottom = (float(x) for x in HCMC_VIEWBOX.split(","))
    return left <= lon <= right and bottom <= lat <= top


@st.cache_data(show_spinner="Đang tra địa chỉ...", ttl=3600, max_entries=200)
def geocode(query: str):
    """Đổi địa chỉ gõ vào thành tọa độ. Trả về danh sách (tên đầy đủ, vĩ độ, kinh độ), tối đa 5 kết quả,
    chỉ trong khung TP.HCM. Nominatim hay không tìm thấy khi có số nhà hoặc thiếu tên thành phố, nên thử lần lượt:
    câu gõ nguyên văn, thêm "Hồ Chí Minh", bỏ số nhà ở đầu câu. Lỗi mạng thì ném ngoại lệ."""
    variants = [query]
    if "hồ chí minh" not in query.lower() and "hcm" not in query.lower():
        variants.append(f"{query}, Hồ Chí Minh")
    no_number = re.sub(r"^\s*(số\s*)?\d+[a-zA-Z]?([/\-]\d+[a-zA-Z]?)*[,\s]+", "", query, flags=re.IGNORECASE)
    if no_number and no_number != query:
        variants.append(no_number)
        if "hồ chí minh" not in no_number.lower() and "hcm" not in no_number.lower():
            variants.append(f"{no_number}, Hồ Chí Minh")
    for q in variants:
        found = _nominatim(q, bounded=True)
        if found:
            return found
    for q in variants:                                 # không giới hạn khung, rồi lọc lại theo khung TP.HCM
        found = [r for r in _nominatim(q, bounded=False) if _in_hcmc(r[1], r[2])]
        if found:
            return found
    return []


def now_vn():
    """Thứ (0 = Thứ 2) và giờ hiện tại ở Việt Nam (UTC+7, không đổi giờ theo mùa), lấy tới bội 5 phút (13:47 thành
    13:45) rồi đưa vào khoảng 6:00–22:30 được chọn. Trả về (thứ, giờ, giờ thật, có bị đưa vào khoảng không)."""
    t = datetime.now(timezone.utc) + timedelta(hours=7)
    real = t.hour + t.minute / 60
    m = (t.hour * 60 + t.minute) // 5 * 5
    i = min(max((m - 6 * 60) // 5, 0), len(HOURS) - 1)
    hour = HOURS[i]                                    # lấy đúng phần tử của HOURS để thanh chọn giờ nhận ra
    return t.weekday(), hour, real, int(round(hour * 60)) != m


# ============================================================================= Trạng thái giữa các lần chạy lại
def set_now():
    ss = st.session_state
    ss.weekday, ss.hour, real, clamped = now_vn()
    ss.now_note = (f"Bây giờ là {int(real)}:{int(round(real % 1 * 60)):02d}, ngoài khoảng 6:00–22:30 nên lấy "
                   f"{fmt_hour(ss.hour)}.") if clamped else None


def clear_now_note():
    st.session_state.now_note = None                   # người dùng tự đổi thứ/giờ: bỏ ghi chú "giờ hiện tại"


def init_state():
    ss = st.session_state
    if "weekday" not in ss or ss.pop("pending_now", False):
        set_now()                                      # lần mở đầu tiên, hoặc vừa bấm "Giờ hiện tại"
    ss.setdefault("origin_name", "ĐH Sư phạm - cơ sở An Dương Vương (ngã tư Nguyễn Văn Cừ)")
    ss.setdefault("dest_name", "Tình huống 1 · Đường Tân Sơn Nhì, Tân Phú (7:30)")
    ss.setdefault("origin_click", None)
    ss.setdefault("dest_click", None)
    ss.setdefault("last_click", None)
    ss.setdefault("click_target", "Điểm đi")
    ss.setdefault("choice", DEFAULT_CHOICE)
    # Điểm vừa bấm / vừa tra ở lần chạy trước: phải đổi ô chọn TRƯỚC khi ô chọn được vẽ ra
    # (Streamlit không cho sửa giá trị của một ô chọn sau khi nó đã được vẽ trong cùng lần chạy).
    pending = ss.pop("pending_click", None)
    if pending:
        which, xy = pending
        ss[f"{which}_click"] = xy
        ss[f"{which}_name"] = CLICKED


def point(which):
    ss = st.session_state
    name = ss[f"{which}_name"]
    if name == CLICKED:
        return ss[f"{which}_click"]
    if name in PLACES:
        return PLACES[name]
    found = lookup_address(name)                       # chữ người dùng tự gõ: tra địa chỉ
    return found[1] if found else None


def lookup_address(text):
    """Kết quả tra địa chỉ (tên đầy đủ, (vĩ độ, kinh độ)) của chữ đã gõ, hoặc None nếu không tra được."""
    if not text or not text.strip():
        return None
    st.session_state["geocode_error"] = None
    try:
        found = geocode(text.strip())
    except Exception as exc:                           # mất mạng, dịch vụ quá tải...
        st.session_state["geocode_error"] = f"{type(exc).__name__}: {exc}"
        return None
    if not found:
        return None
    return found[0][0], (found[0][1], found[0][2])


def handle_click(clicked):
    """Ghi điểm vừa bấm trên bản đồ vào điểm đi hoặc điểm đến (mỗi lần bấm chỉ xử lý một lần)."""
    if not clicked:
        return
    xy = (round(clicked["lat"], 6), round(clicked["lng"], 6))
    ss = st.session_state
    if xy == ss.last_click:
        return
    ss.last_click = xy
    ss.pending_click = ("origin" if ss.click_target == "Điểm đi" else "dest", xy)
    st.rerun()


# ============================================================================= Các khối giao diện
def place_picker(which, label, icon):
    """Một ô duy nhất: chọn địa điểm có sẵn, hoặc gõ địa chỉ mới rồi Enter (app tự tra địa chỉ)."""
    ss = st.session_state
    options = list(PLACES)
    if ss[f"{which}_click"]:
        options.append(CLICKED)
    current = ss[f"{which}_name"]
    if current not in options:                         # địa chỉ đã gõ ở lần trước: giữ lại trong danh sách
        options.append(current)
    st.selectbox(f"{icon} {label}", options, key=f"{which}_name", accept_new_options=True,
                 placeholder="Chọn địa điểm hoặc gõ địa chỉ rồi Enter")
    name = ss[f"{which}_name"]
    if name and name not in PLACES and name != CLICKED:
        found = lookup_address(name)
        if found:
            st.caption(f":material/check: {found[0]}")
        else:
            error = ss.get("geocode_error")
            if error:
                st.warning(f"Không kết nối được dịch vụ tra địa chỉ ({error}). Kiểm tra Internet rồi thử lại, "
                           "hoặc bấm trực tiếp lên bản đồ.", icon=":material/wifi_off:")
            else:
                st.warning("Không tìm thấy địa chỉ này trong TP.HCM. Thử gõ tên đường và quận, "
                           "vd: Lê Văn Sỹ, Quận 3, hoặc bấm trực tiếp lên bản đồ.", icon=":material/search_off:")


def sidebar(choices, errors):
    """Thanh bên: lộ trình, thời điểm, cách chọn đường, chú thích màu. Trả về (thứ, giờ, lựa chọn)."""
    with st.sidebar:
        st.header("Lộ trình", icon=":material/route:")
        place_picker("origin", "Điểm đi", ":material/trip_origin:")
        place_picker("dest", "Điểm đến", ":material/location_on:")
        st.segmented_control("Bấm lên bản đồ để đặt", ["Điểm đi", "Điểm đến"], key="click_target",
                             required=True, width="stretch")

        st.header("Thời điểm", icon=":material/schedule:")
        weekday = st.segmented_control("Thứ", range(7), format_func=weekday_label, key="weekday",
                                       required=True, width="stretch", on_change=clear_now_note)
        hour = st.select_slider("Giờ khởi hành", HOURS, format_func=fmt_hour, key="hour", on_change=clear_now_note)
        if st.button("Giờ hiện tại", icon=":material/update:", width="stretch"):
            st.session_state.pending_now = True
            st.rerun()
        if st.session_state.get("now_note"):
            st.caption(st.session_state.now_note)
        st.caption("Chỉ chọn được 6:00–22:30: từ 23h đến 6h tập huấn luyện gần như không có số liệu, "
                   "nên dự báo không đáng tin.")

        st.header("Lựa chọn mô hình", icon=":material/model_training:")
        choice = st.selectbox("Mô hình", choices, key="choice", label_visibility="collapsed")
        model = R.DEFAULT_MODEL if CHOICES[choice] is None else CHOICES[choice]
        if CHOICES[choice] is None:
            st.caption("Đường ngắn nhất chỉ dựa vào chiều dài; thời gian và màu tình trạng ước tính bằng Random Forest.")
        else:
            st.caption("Random Forest và XGBoost dùng bản đã tinh chỉnh siêu tham số.")
        if model in errors:
            mae, mpk = errors[model]
            st.caption(f"Sai số trên tập test: trung bình {fmt_num(mae)} km/h mỗi đoạn, "
                       f"tương đương {fmt_num(mpk, 2)} phút cho mỗi km.")

        st.header("Tình trạng giao thông", icon=":material/traffic:")
        items = []
        for level in LOS_COLORS:
            items.append((LOS_COLORS[level], LOS_TEXT[level].capitalize()))
        if CHOICES[choice] is not None:
            items.append(("repeating-linear-gradient(90deg,#9aa3b2 0 6px,transparent 6px 10px)",
                          "Đường ngắn nhất (để so sánh)"))
        st.html(ui.legend(items))
    return weekday, hour, choice


def view_of(points, width_px=900, height_px=480):
    """Tâm và mức phóng để khung bản đồ chứa hết các điểm.
    Tự tính thay cho fit_bounds, vì bản đồ trong Streamlit được vẽ trước khi giãn ra hết bề ngang
    nên fit_bounds canh theo khung nhỏ (lộ trình bị lệch sang trái)."""
    lats = []
    lons = []
    for p in points:
        lats.append(p[0])
        lons.append(p[1])
    center = ((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)
    span_lon = max(max(lons) - min(lons), 1e-4)
    span_lat = max(max(lats) - min(lats), 1e-4)
    # ở mức phóng z, 1 điểm ảnh ≈ 360 / (256 · 2^z) độ (gần xích đạo, vĩ độ và kinh độ gần như cùng tỷ lệ)
    z = min(math.log2(360 * width_px / (256 * span_lon)), math.log2(360 * height_px / (256 * span_lat)))
    return center, int(max(11, min(17, math.floor(z))))


PULSE_CSS = """
<style>
/* Điểm đi/đến: chấm tròn viền trắng, bóng đổ lệch xuống; một vòng mảnh lan ra rồi tắt dần để dễ thấy vị trí */
@keyframes ring { 0% { transform: scale(.7); opacity: .7; } 100% { transform: scale(2.6); opacity: 0; } }
.pin { position: relative; width: 18px; height: 18px; }
.pin b { position: absolute; inset: 0; border-radius: 50%; border: 3px solid #fff;
         box-shadow: 0 3px 6px rgba(12,51,84,.35); }
.pin i { position: absolute; inset: 0; border-radius: 50%; border: 2px solid; background: transparent;
         animation: ring 2.4s cubic-bezier(0.23, 1, 0.32, 1) infinite; }
/* Dòng chảy dọc lộ trình: các chấm trắng nhỏ trôi đều theo hướng đi (dịch stroke-dashoffset của SVG) */
@keyframes flow { to { stroke-dashoffset: -26; } }
.flow { stroke-dasharray: 0.1 26; animation: flow 1.6s linear infinite; pointer-events: none; }
@media (prefers-reduced-motion: reduce) { .pin i, .flow { animation: none; } .pin i { opacity: 0; } }
</style>"""


def pin(color, label):
    """Điểm đi/đến dạng chấm tròn có một vòng mảnh lan ra (hiệu ứng động bằng CSS trong bản đồ)."""
    html = f'<div class="pin" title="{label}"><i style="border-color:{color}"></i><b style="background:{color}"></b></div>'
    return folium.DivIcon(html=html, icon_size=(18, 18), icon_anchor=(9, 9))


def draw_map(route, origin, dest, other=None):
    """route: lộ trình chính (tô màu theo tình trạng dự báo); other: đường ngắn nhất để so (nét đứt xám) hoặc None."""
    pts = [origin, dest] + route["toa_do"]
    if other:
        pts = pts + other["toa_do"]
    center, zoom = view_of(pts)
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(ESRI_GRAY, attr=ESRI_ATTR, name="Bản đồ nền xám", max_native_zoom=16, max_zoom=19).add_to(m)
    # Nền mặc định là bản đồ xám tối giản (chỉ đường và tên đường); OpenStreetMap chi tiết hơn (nhà, cửa hàng...)
    # chỉ hiện khi chọn ở nút lớp bản đồ góc phải trên. show=False: không thì lớp thêm sau cùng sẽ đè lên.
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (chi tiết)", show=False).add_to(m)
    folium.TileLayer(ESRI_LABELS, attr=ESRI_ATTR, name="Tên đường", overlay=True,
                     max_native_zoom=16, max_zoom=19).add_to(m)
    m.get_root().header.add_child(folium.Element(PULSE_CSS))
    if other:
        folium.PolyLine(other["toa_do"], color="#6b7280", weight=4, opacity=0.8, dash_array="8 8",
                        tooltip=f"Đường ngắn nhất: {fmt_num(other['km'], 2)} km, "
                                f"dự báo {fmt_num(other['phut'])} phút").add_to(m)
    folium.PolyLine(route["toa_do"], color=ui.NAVY_DARK, weight=11, opacity=0.85).add_to(m)   # viền tối liền mạch
    for r in route["chi_tiet"].itertuples():
        color = LOS_COLORS.get(r.LOS_du_bao, "#2a78d6")
        folium.PolyLine([(r.lat_s, r.lon_s), (r.lat_e, r.lon_e)], color=color, weight=7, opacity=1,
                        tooltip=f"{r.street_name if isinstance(r.street_name, str) else '(không tên)'}: "
                                f"{r.toc_do_du_bao:.0f} km/h, {LOS_TEXT.get(r.LOS_du_bao, '')}"
                                f"{'' if r.so_mau_lich_su else ' (đoạn chưa từng có số liệu)'}").add_to(m)
    # Dòng chảy chạy dọc lộ trình theo hướng đi (hiệu ứng động)
    folium.PolyLine(route["toa_do"], color="#ffffff", weight=4, opacity=0.95, line_cap="round",
                    className="flow").add_to(m)
    folium.Marker(origin, tooltip="Điểm đi", icon=pin("#16a34a", "Điểm đi")).add_to(m)
    folium.Marker(dest, tooltip="Điểm đến", icon=pin(ui.RED, "Điểm đến")).add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    return m


def summary_cards(choice, short, fast, shortest_only):
    """Dải số liệu gọn phía trên bản đồ (thời điểm và mô hình đã hiện ở thanh bên)."""
    if shortest_only:
        st.html(ui.stats([("Đường ngắn nhất", f"{fmt_num(short['phut'])} phút", f"{fmt_num(short['km'], 2)} km"),
                          ("Thời gian ước tính bằng", "Random Forest", "chọn đường chỉ theo chiều dài")]))
        return
    saved = short["phut"] - fast["phut"]
    st.html(ui.stats([
        (f"Nhanh nhất theo {choice}", f"{fmt_num(fast['phut'])} phút", f"{fmt_num(fast['km'], 2)} km"),
        ("Đường ngắn nhất", f"{fmt_num(short['phut'])} phút", f"{fmt_num(short['km'], 2)} km"),
        ("Tiết kiệm theo dự báo", f"{fmt_num(saved)} phút",
         "hai đường trùng nhau" if abs(saved) < 0.05 and abs(fast["km"] - short["km"]) < 0.01
         else f"đi xa hơn {fmt_num(max(fast['km'] - short['km'], 0), 2)} km"),
    ]))


def details(router, choices, origin, dest, weekday, hour, shown):
    tab_streets, tab_compare, tab_about = st.tabs([":material/signpost: Các đường đi qua",
                                                   ":material/compare_arrows: So sánh các lựa chọn",
                                                   ":material/info: Về dự báo này"])
    with tab_streets:
        st.dataframe(R.streets_summary(shown["chi_tiet"]), hide_index=True,
                     alt="Danh sách các đường trên lộ trình với chiều dài, thời gian và tốc độ dự báo")
    with tab_compare:
        rows = []
        for c in choices:
            m = CHOICES[c] or R.DEFAULT_MODEL
            r = router.route(origin, dest, weekday, hour, m)
            path = r["ngan_nhat"] if CHOICES[c] is None else r["nhanh_nhat"]
            rows.append({"Lựa chọn mô hình": c, "Dài (km)": round(path["km"], 2),
                         "Thời gian dự báo (phút)": round(path["phut"], 1), "Dự báo bằng": m})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     alt="So sánh chiều dài và thời gian dự báo giữa các cách chọn đường")
        st.caption("Mỗi mô hình tự dự báo tốc độ rồi chọn đường nhanh nhất theo dự báo của chính nó, nên thời gian "
                   "giữa các dòng không so trực tiếp được: chúng khác nhau một phần do lộ trình, một phần do mô hình "
                   "đoán nhanh chậm khác nhau.")
    with tab_about:
        st.markdown(f"**{fmt_num(100 * shown['ty_le_chieu_dai_co_lich_su'], 0)}%** chiều dài lộ trình này đi qua các đoạn "
                    "từng có số liệu trong tập huấn luyện; phần còn lại được dự báo chỉ từ loại đường, vị trí và thời gian.")
        st.markdown(
            "- Mô hình dự báo tốc độ cho **toàn bộ 84.633 đoạn đường** ở thứ và giờ đã chọn, theo những gì "
            "thường xảy ra vào thứ, giờ đó trong dữ liệu 2020–2021; đây không phải tình hình giao thông đang diễn ra.\n"
            "- Chỉ khoảng **7% số đoạn** từng có số liệu; các đoạn còn lại được dự báo từ loại đường, vị trí và thời gian.\n"
            "- Thời gian đi được tính **theo giờ tới từng đoạn**: xe tới đoạn nào lúc mấy giờ thì dùng tốc độ dự báo "
            "của khung 30 phút lúc đó; đang đi mà sang khung mới thì phần còn lại của đoạn đi với tốc độ của khung mới "
            "(tốc độ tối thiểu 3 km/h). Đường nhanh nhất tìm bằng thuật toán Dijkstra phụ thuộc thời gian, cho thời điểm "
            "tới sớm nhất; đường ngắn nhất tìm theo chiều dài và cũng tính thời gian theo cách trên.\n"
            "- Trên tập test, các mô hình có xu hướng đoán **thời gian đi ít hơn thực tế**, vì hiếm khi dự báo "
            "tình trạng kẹt cứng; nên hiểu con số phút là ước lượng lạc quan.")


# ============================================================================= Trang chính
init_state()
router = load_router()
saved_models = router.available_models()
choices = []
for c in CHOICES:
    if CHOICES[c] is None or CHOICES[c] in saved_models:
        choices.append(c)
if R.DEFAULT_MODEL not in saved_models:
    st.error("Chưa có mô hình đã lưu trong models/v2. Hãy chạy chay_v2.bat trước.", icon=":material/error:")
    st.stop()
if st.session_state.choice not in choices:
    st.session_state.choice = DEFAULT_CHOICE

if LOGO:
    # Thanh bên nền xanh: dùng logo kèm tên trường chữ trắng; khi thu gọn thanh bên thì hiện logo nhỏ
    wide = ASSETS / "logo_chu_trang.png"
    st.logo(str(wide if wide.exists() else LOGO), size="large", icon_image=str(LOGO))
weekday, hour, choice = sidebar(choices, test_errors())
st.html(ui.hero())

origin, dest = point("origin"), point("dest")
if origin is None or dest is None:
    st.info("Hãy gõ địa chỉ hoặc bấm lên bản đồ để đặt điểm còn thiếu.", icon=":material/touch_app:")
    with st.container(key="map_card"):
        m = folium.Map(location=(10.78, 106.68), zoom_start=13, tiles=None)
        folium.TileLayer(ESRI_GRAY, attr=ESRI_ATTR, max_native_zoom=16, max_zoom=19).add_to(m)
        folium.TileLayer(ESRI_LABELS, attr=ESRI_ATTR, overlay=True, max_native_zoom=16, max_zoom=19).add_to(m)
        out = st_folium(m, height=MAP_HEIGHT, use_container_width=True, returned_objects=["last_clicked"],
                        key="ban_do_trong")
    handle_click(out.get("last_clicked") if out else None)
    st.stop()

shortest_only = CHOICES[choice] is None
model = R.DEFAULT_MODEL if shortest_only else CHOICES[choice]
start = time.time()
try:
    res = router.route(origin, dest, weekday, hour, model)
except ValueError as err:
    st.warning(str(err), icon=":material/warning:")
    st.stop()
seconds = time.time() - start
short, fast = res["ngan_nhat"], res["nhanh_nhat"]
shown = short if shortest_only else fast

summary_cards(choice, short, fast, shortest_only)
with st.sidebar:
    st.caption(f":material/speed: Dự báo toàn mạng và tìm đường trong {fmt_num(seconds, 2)} giây.")
    st.caption(f"{ui.COURSE} · **{ui.TEAM}**")

for which, d in (("đi", res["lech_diem_di_m"]), ("đến", res["lech_diem_den_m"])):
    if d > FAR_SNAP_M:
        st.warning(f"Điểm {which} cách nút giao gần nhất trên mạng đường {fmt_num(d, 0)} m; "
                   "lộ trình bắt đầu/kết thúc ở nút giao đó.", icon=":material/near_me_disabled:")

with st.container(key="map_card"):
    out = st_folium(draw_map(shown, origin, dest, None if shortest_only else short), height=MAP_HEIGHT,
                    use_container_width=True, returned_objects=["last_clicked"], key="ban_do")
handle_click(out.get("last_clicked") if out else None)

st.space("small")
with st.container(key="detail_card"):
    details(router, choices, origin, dest, weekday, hour, shown)
st.html(ui.footer())
