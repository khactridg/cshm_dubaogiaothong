@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
for %%f in (b1_kham_pha_du_lieu b2_lam_sach b3_chia_du_lieu b4_tinh_chinh b5_huan_luyen_va_kiem_tra b6_phan_tich_sai_so b7_thong_tin_gan_nhat b8_tinh_huong_tim_duong) do (
  echo === %%f ===
  .venv\Scripts\python %%f.py || (echo Loi o %%f & pause & exit /b 1)
)
echo Da chay xong.
pause
