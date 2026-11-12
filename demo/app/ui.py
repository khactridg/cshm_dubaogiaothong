"""
ui.py - Phần trình bày của giao diện demo: CSS, khung đầu trang, dải số liệu, chân trang.

Theo bộ nhận diện thương hiệu của Trường (thư mục "HCMUE - Nhận diện thương hiệu"):
  - màu: xanh #124874 là màu nền của thương hiệu (khung đầu trang, thanh bên, chân trang),
         đỏ #CF373D là màu nhấn, chỉ dùng cho vạch dưới tên đề tài, điểm đến và kết quả chính;
  - chữ: Bree Serif (bản Việt hóa có trong bộ nhận diện) cho tiêu đề, Be Vietnam Pro cho nội dung.
Khung đầu trang có ba lớp: mạng đường TP.HCM vẽ từ chính dữ liệu của đề tài (lớp xa, mờ, chừa chỗ cho chữ),
Tòa nhà A (đặt trên mép dưới của khung) và tên đề tài. Ảnh nền ở app/static/ (bật enableStaticServing trong
.streamlit/config.toml); ảnh Tòa nhà A và logo ở app/assets/.
Chuyển động chỉ khi mở trang (hiện dần, lệch nhịp), không có hiệu ứng lặp vô hạn; khi máy bật chế độ giảm
chuyển động thì chỉ còn hiện dần, không dịch chuyển.
"""
import base64
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
NAVY, NAVY_DARK, RED, PAPER = "#124874", "#0c3354", "#CF373D", "#f3f4f8"
INK, INK_SOFT = "#14243a", "#56657d"          # chữ chính và chữ phụ (xám ngả xanh, không dùng xám trơn)

COURSE = "Đề tài môn học Cơ sở học máy và ứng dụng"
TEAM = "Nhóm Nước đến tai mới nhảy"
SCHOOL = "Trường Đại học Sư phạm Thành phố Hồ Chí Minh"
NETWORK_IMG = "app/static/mang_duong.webp"    # đường dẫn do Streamlit phục vụ (thư mục app/static)


def img_data_uri(name: str) -> str:
    """Ảnh trong app/assets dưới dạng data URI (nhúng thẳng vào HTML, không cần máy chủ tệp tĩnh)."""
    path = ASSETS / name
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


# Hạt nhiễu rất nhẹ trên nền xanh để nền không bị loang thành từng dải màu trên màn hình thật
GRAIN = ("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E"
         "%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/%3E"
         "%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.9'/%3E%3C/svg%3E")
# Ký tự < > được mã hóa (%3C, %3E): nếu để nguyên, thẻ <svg> nằm giữa khối <style> làm trình duyệt hiểu sai và bỏ cả khối CSS

CSS = f"""
<style>
:root {{
  --navy: {NAVY}; --navy-dark: {NAVY_DARK}; --red: {RED}; --ink: {INK}; --ink-soft: {INK_SOFT};
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  --display: 'Bree Serif VN', 'Bree Serif', Georgia, serif;
  /* ba mức nổi, bóng ngả màu nền xanh, có độ lệch xuống dưới */
  --e1: 0 1px 2px rgba(18,72,116,.06), 0 4px 12px -4px rgba(18,72,116,.14);
  --e2: 0 2px 4px rgba(18,72,116,.06), 0 12px 28px -10px rgba(18,72,116,.28);
  --e3: 0 4px 8px rgba(12,51,84,.10), 0 22px 44px -16px rgba(12,51,84,.40);
}}
@keyframes rise   {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes settle {{ from {{ opacity: 0; transform: scale(1.05); }} to {{ opacity: .62; transform: none; }} }}
@keyframes rule   {{ from {{ transform: scaleX(0); }} to {{ transform: none; }} }}
@keyframes appear {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

/* ---------------------------------------------------------------- Khung đầu trang: ba lớp */
.hero {{
  position: relative; overflow: hidden; isolation: isolate; border-radius: 18px;
  min-height: 156px; padding: 24px 32px 26px; display: flex; flex-direction: column; justify-content: center;
  background: radial-gradient(90% 140% at 78% 100%, #1a5a8f 0%, var(--navy) 46%, var(--navy-dark) 100%);
  color: #fff; box-shadow: var(--e3), inset 0 1px 0 rgba(255,255,255,.12);
}}
/* Lớp xa: mạng đường thật của TP.HCM (đoạn có số liệu tô hồng), mờ dần về phía chữ */
.hero .net {{
  position: absolute; z-index: -2; right: -2%; top: 50%; height: 190%; width: auto; translate: 0 -50%;
  opacity: .62; filter: blur(.2px);
  -webkit-mask-image: linear-gradient(to left, #000 40%, transparent 82%);
          mask-image: linear-gradient(to left, #000 40%, transparent 82%);
  animation: settle 1.4s var(--ease-out) both;
}}
.hero .grain {{
  position: absolute; inset: 0; z-index: -1; pointer-events: none; opacity: .05; mix-blend-mode: overlay;
  background-image: url("{GRAIN}");
}}
/* Lớp giữa: Tòa nhà A đứng trên mép dưới của khung, có bóng đổ xuống */
.hero .building {{
  position: absolute; right: 22px; bottom: 0; height: 90%; max-width: 38%; object-fit: contain;
  object-position: right bottom; filter: drop-shadow(0 10px 14px rgba(6,28,48,.45));
  animation: rise .9s .15s var(--ease-out) both;
}}
/* Lớp gần: tên đề tài */
.hero h1 {{
  font-family: var(--display); font-weight: 400; color: #fff; margin: 0 0 12px; padding: 0;
  font-size: clamp(1.45rem, 1.05rem + 1.3vw, 2.15rem); line-height: 1.12; letter-spacing: -.005em;
  max-width: 58%; text-wrap: balance; text-shadow: 0 2px 10px rgba(6,28,48,.35);
  animation: rise .7s .05s var(--ease-out) both;
}}
.hero .rule {{
  height: 4px; width: 72px; background: var(--red); border-radius: 2px; transform-origin: left;
  animation: rule .6s .3s var(--ease-out) both;
}}
.hero .by {{
  margin-top: 14px; display: flex; align-items: baseline; gap: 9px; color: #c6d4e6;
  animation: rise .7s .4s var(--ease-out) both;
}}
.hero .by i {{ font-family: var(--display); font-style: normal; font-size: 1rem; opacity: .85; }}
.hero .by b {{ font-weight: 700; font-size: .84rem; letter-spacing: .24em; text-transform: uppercase; color: #fff; }}

/* ---------------------------------------------------------------- Dải số liệu phía trên bản đồ */
.stats {{ display: grid; grid-template-columns: 1.35fr 1fr 1fr; gap: 12px; }}
.stats.n2 {{ grid-template-columns: 1.35fr 1fr; }}
.stat {{
  background: #fff; border-radius: 14px; padding: 10px 18px 11px; border: 1px solid #e2e6ee;
  box-shadow: var(--e1); display: grid; grid-template-columns: auto 1fr; align-items: baseline; column-gap: 10px;
  animation: rise .5s var(--ease-out) both; transition: transform .16s var(--ease-out), box-shadow .16s var(--ease-out);
}}
.stat:nth-child(2) {{ animation-delay: .06s; }}
.stat:nth-child(3) {{ animation-delay: .12s; }}
.stat .k {{ grid-column: 1 / -1; font-size: .8rem; color: var(--ink-soft); }}
.stat .v {{ font-size: 1.32rem; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; letter-spacing: -.01em; }}
.stat .s {{ font-size: .82rem; color: var(--ink-soft); font-variant-numeric: tabular-nums; }}
/* Kết quả chính (ô đầu): to hơn, có vạch đỏ trước nhãn */
.stat:first-child .v {{ font-size: 1.6rem; color: var(--navy); }}
.stat:first-child .k::before {{
  content: ""; display: inline-block; width: 14px; height: 3px; border-radius: 2px; background: var(--red);
  vertical-align: middle; margin: -2px 8px 0 0;
}}
@media (hover: hover) and (pointer: fine) {{
  .stat:hover {{ transform: translateY(-2px); box-shadow: var(--e2); }}
}}
@media (max-width: 900px) {{
  .stats, .stats.n2 {{ grid-template-columns: 1fr; }}
  .hero {{ padding: 22px 22px 26px; }}
  .hero h1 {{ max-width: 100%; }}
  .hero .building {{ display: none; }}
}}

/* ---------------------------------------------------------------- Thẻ bản đồ, thẻ chi tiết */
.st-key-map_card {{ animation: appear .5s .15s var(--ease-out) both; }}
.st-key-map_card iframe {{ border-radius: 16px; border: 1px solid #dde2ec; box-shadow: var(--e2); }}
.st-key-detail_card {{
  background: #fff; border-radius: 16px; border: 1px solid #e2e6ee; box-shadow: var(--e1);
  padding: 6px 18px 14px; animation: appear .5s .25s var(--ease-out) both;
}}

/* ---------------------------------------------------------------- Khung trang */
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 1rem !important; padding-bottom: 1.5rem; }}
[data-testid="stSidebarHeader"] {{ height: auto; padding-top: 1.1rem; padding-bottom: .4rem; }}
[data-testid="stSidebarHeader"] img[data-testid="stLogo"] {{ height: auto !important; width: 240px; max-width: 82%; }}
[data-testid="stSidebar"] h2 {{ letter-spacing: .005em; }}

/* Chú thích màu trong thanh bên */
.legend {{ display: grid; gap: 7px; }}
.legend span {{ display: flex; align-items: center; gap: 10px; font-size: .9rem; }}
.legend i {{ width: 30px; height: 8px; border-radius: 4px; display: inline-block; box-shadow: inset 0 0 0 1px rgba(255,255,255,.18); }}

.footer {{
  margin-top: 26px; padding: 16px 24px; border-radius: 16px; background: var(--navy); color: #d5e0ee;
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap; font-size: .86rem;
  box-shadow: var(--e1), inset 0 1px 0 rgba(255,255,255,.1);
}}
.footer img {{ height: 34px; }}
.footer b {{ color: #fff; }}
.footer .sep {{ opacity: .4; }}

/* ---------------------------------------------------------------- Bề mặt trình duyệt */
::selection {{ background: rgba(207,55,61,.2); }}
:focus-visible {{ outline: 2px solid var(--red); outline-offset: 2px; }}
html {{ scrollbar-color: #b9c3d3 transparent; accent-color: var(--navy); caret-color: var(--navy); }}

/* Giảm chuyển động: chỉ giữ hiện dần, bỏ mọi dịch chuyển */
@media (prefers-reduced-motion: reduce) {{
  .hero h1, .hero .by, .hero .building, .stat {{ animation: appear .3s ease-out both; }}
  .hero .net {{ animation: none; }}
  .hero .rule {{ animation: none; }}
  .stat {{ transition: none; }}
}}
</style>
"""


def hero():
    """Khung đầu trang: chỉ tên đề tài và tên nhóm (thông tin thời điểm, mô hình nằm ở thanh bên)."""
    return f"""
<div class="hero">
  <img class="net" src="{NETWORK_IMG}" alt="">
  <div class="grain"></div>
  <img class="building" src="{img_data_uri('toa_nha_a.png')}" alt="">
  <h1>Dự báo giao thông và đề xuất tuyến đường tại TP.&nbsp;Hồ Chí Minh</h1>
  <div class="rule"></div>
  <div class="by"><i>by</i><b>{TEAM.removeprefix("Nhóm ")}</b></div>
</div>"""


def stats(items):
    """Dải số liệu gọn. items: danh sách (nhãn, giá trị, ghi chú); ô đầu là kết quả chính."""
    cells = ""
    for k, v, s in items:
        cells += f'<div class="stat"><span class="k">{k}</span><span class="v">{v}</span><span class="s">{s}</span></div>'
    return f'<div class="stats n{len(items)}">{cells}</div>'


def legend(items):
    """items: danh sách (màu, chữ)."""
    rows = ""
    for c, t in items:
        rows += f'<span><i style="background:{c}"></i>{t}</span>'
    return f'<div class="legend">{rows}</div>'


def footer():
    return f"""
<div class="footer">
  <img src="{img_data_uri('logo_chu_trang.png')}" alt="">
  <span><b>{TEAM}</b></span><span class="sep">|</span>
  <span>{COURSE}</span><span class="sep">|</span>
  <span>Dữ liệu: bộ dữ liệu giao thông TP.HCM trên Kaggle (2020–2021)</span>
</div>"""
