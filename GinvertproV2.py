import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

# ==========================================
# MODUL 1: GRIDDING, KERNEL & WEIGHTING
# ==========================================
def setup_3d_grid(x, y, anomaly, nx, ny, nz, z_top, z_bottom):
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    XI, YI = np.meshgrid(xi, yi)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    
    g_grid = griddata((x, y), anomaly, (XI, YI), method='cubic')
    g_grid = np.nan_to_num(g_grid, nan=np.nanmean(g_grid))
    
    z_bounds = np.linspace(z_top, z_bottom, nz + 1)
    return XI, YI, g_grid, dx, dy, z_bounds

def calculate_kernel_layer(XI, YI, dx, dy, z_layer_top, z_layer_bottom, rho_layer, beta_weight):
    GAMMA = 6.67430e-3 
    X_k = XI - np.mean(XI)
    Y_k = YI - np.mean(YI)
    R2 = X_k**2 + Y_k**2
    
    K = rho_layer * ( (-1 / np.sqrt(R2 + z_layer_bottom**2)) - (-1 / np.sqrt(R2 + z_layer_top**2)) )
    
    z_center = (z_layer_top + z_layer_bottom) / 2.0
    epsilon = dx / 2.0 
    depth_weight = 1.0 / (z_center + epsilon)**beta_weight
    
    K_weighted = K * depth_weight
    
    S = GAMMA * K_weighted.sum() * dx * dy 
    S_matrix = np.full_like(XI, S)
    return GAMMA * K_weighted, S_matrix

# ==========================================
# MODUL 2: FILTER SEPARASI LAYER
# ==========================================
def upward_continuation(field, dz, dx, dy):
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_wave = np.sqrt(KX**2 + KY**2)
    return np.real(np.fft.ifft2(np.fft.fft2(field) * np.exp(-dz * K_wave)))

def downward_continuation_lavrentiev(field, dz, kappa, dx, dy):
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_wave = np.sqrt(KX**2 + KY**2)
    down_filter_reg = 1.0 / (np.exp(-dz * K_wave) + kappa)
    return np.real(np.fft.ifft2(np.fft.fft2(field) * down_filter_reg))

def extract_layer_anomaly(g_obs, z_depth, kappa, dx, dy):
    if z_depth <= 0: return g_obs
    g_up = upward_continuation(g_obs, z_depth, dx, dy)
    g_down = downward_continuation_lavrentiev(g_up, 2 * z_depth, kappa, dx, dy)
    return upward_continuation(g_down, z_depth, dx, dy)

# ==========================================
# VISUALISASI PRE-INVERSI DENGAN TOPOGRAFI
# ==========================================
def plot_3d_grid_wireframe_with_topo(nx, ny, nz, z_top, z_bottom, XI, YI, elev_grid, g_grid):
    fig = go.Figure()
    
    x_min, x_max = XI.min(), XI.max()
    y_min, y_max = YI.min(), YI.max()
    
    zt = -abs(z_top)
    zb = -abs(z_bottom)

    x_lines = [x_min, x_max, x_max, x_min, x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_max, x_max, x_min, x_min]
    y_lines = [y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_max, y_max]
    z_lines = [zt, zt, zt, zt, zt, zb, zb, zb, zb, zb, zb, zt, zt, zb, zb, zt]

    fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color='blue', width=4), name='Batas Blok 3D'))
    
    z_layers = np.linspace(zt, zb, nz+1)
    for z in z_layers[1:-1]:
        fig.add_trace(go.Scatter3d(
            x=[x_min, x_max, x_max, x_min, x_min], y=[y_min, y_min, y_max, y_max, y_min], z=[z, z, z, z, z],
            mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False
        ))
        
    fig.add_trace(go.Surface(
        x=XI, y=YI, z=elev_grid,  
        surfacecolor=g_grid,      
        colorscale='Jet', 
        name='Topography & Anomaly',
        colorbar=dict(title="Anomali (mGal)", len=0.5, x=0.85)
    ))

    fig.update_layout(scene=dict(aspectmode='auto'), margin=dict(l=0, r=0, b=0, t=30))
    return fig

# ==========================================
# MODUL 3: INVERSI KOREKSI LOKAL ITERATIF (CONSTRAINED)
# ==========================================
def gravity_inversion_3d(XI, YI, g_obs, dx, dy, z_bounds, nx, ny, nz, rho_1d_model, beta_weight, max_iter, smooth_sigma, max_dev, well_data=None):
    g_mean = np.nanmean(g_obs)
    g_obs_shifted = g_obs - g_mean
    
    Density_Model_3D = np.zeros((ny, nx, nz))
    g_calc_total = np.zeros((ny, nx))
    
    kappas = np.logspace(-2, 0.5, nz + 1)
    g_bases = [extract_layer_anomaly(g_obs_shifted, z, k, dx, dy) for z, k in zip(z_bounds, kappas)]
    
    Well_Mask = np.zeros((ny, nx, nz), dtype=bool)
    Well_Phi = np.zeros((ny, nx, nz))
    Well_RHOB_True = np.zeros((ny, nx, nz))
    z_centers_all = (z_bounds[:-1] + z_bounds[1:]) / 2
    
    if well_data is not None:
        nama_sumur = well_data.iloc[:, 0].unique()
        for w_name in nama_sumur:
            df_w = well_data[well_data.iloc[:, 0] == w_name]
            wx, wy = df_w.iloc[0, 1], df_w.iloc[0, 2]
            
            idx_x = (np.abs(XI[0, :] - wx)).argmin()
            idx_y = (np.abs(YI[:, 0] - wy)).argmin()
            
            wz_log = df_w.iloc[:, 3].values
            rho_log = df_w.iloc[:, 4].values
            
            if len(wz_log) > 1:
                rho_well_interp = np.interp(z_centers_all, wz_log, rho_log)
                for idx_z in range(nz):
                    rho_bg = np.interp(z_centers_all[idx_z], rho_1d_model['Depth (m)'].values, rho_1d_model['Density (g/cc)'].values)
                    wrhob = rho_well_interp[idx_z]
                    
                    Well_Mask[idx_y, idx_x, idx_z] = True
                    Well_RHOB_True[idx_y, idx_x, idx_z] = wrhob
                    Well_Phi[idx_y, idx_x, idx_z] = (wrhob - rho_bg) / rho_bg
            elif len(wz_log) == 1:
                idx_z = (np.abs(z_centers_all - wz_log[0])).argmin()
                rho_bg = np.interp(z_centers_all[idx_z], rho_1d_model['Depth (m)'].values, rho_1d_model['Density (g/cc)'].values)
                Well_Mask[idx_y, idx_x, idx_z] = True
                Well_RHOB_True[idx_y, idx_x, idx_z] = rho_log[0]
                Well_Phi[idx_y, idx_x, idx_z] = (rho_log[0] - rho_bg) / rho_bg
        
    progress_bar = st.progress(0)
    for k in range(nz):
        z_top = z_bounds[k]
        z_bottom = z_bounds[k+1]
        z_center = (z_top + z_bottom) / 2
        
        rho_layer = np.interp(z_center, rho_1d_model['Depth (m)'].values, rho_1d_model['Density (g/cc)'].values)
        
        g_layer = g_bases[k] - g_bases[k+1]
        g_layer_initial = np.copy(g_layer)
        
        K, S = calculate_kernel_layer(XI, YI, dx, dy, z_top, z_bottom, rho_layer, beta_weight)
        Phi = np.zeros_like(g_layer)
        
        for theta in range(1, int(max_iter) + 1): 
            delta_U = fftconvolve(g_layer, K, mode='same') * dx * dy
            S_S, S_dU = np.sum(S*S)*dx*dy, np.sum(S*delta_U)*dx*dy
            dU_dU, dg_dU = np.sum(delta_U*delta_U)*dx*dy, np.sum(g_layer*delta_U)*dx*dy
            dg_S = np.sum(g_layer*S)*dx*dy
            
            Q = (dU_dU * S_S) - (S_dU**2)
            if Q == 0: break
                
            alpha = (S_S * dg_dU - S_dU * dg_S) / Q
            beta_corr = (dU_dU * dg_S - S_dU * dg_dU) / Q
            
            Phi = Phi + (alpha * g_layer) + beta_corr
            
            if well_data is not None:
                mask_layer = Well_Mask[:, :, k]
                Phi[mask_layer] = Well_Phi[:, :, k][mask_layer]
            
            if smooth_sigma > 0:
                Phi = gaussian_filter(Phi, sigma=smooth_sigma)
                
            Phi = np.clip(Phi, -max_dev/rho_layer, max_dev/rho_layer)
            
            if well_data is not None:
                Phi[mask_layer] = Well_Phi[:, :, k][mask_layer]
            
            g_layer = g_layer - (alpha * delta_U) - (beta_corr * S)
            if np.sqrt(np.sum(g_layer**2)*dx*dy) < 1e-3: break
                
        Density_Model_3D[:, :, k] = rho_layer + (rho_layer * Phi)
        g_calc_total += (g_layer_initial - g_layer)
        progress_bar.progress((k + 1) / nz)
        
    if smooth_sigma > 0:
        Density_Model_3D = gaussian_filter(Density_Model_3D, sigma=(smooth_sigma, smooth_sigma, 1.0))
        
    if well_data is not None:
        Density_Model_3D[Well_Mask] = Well_RHOB_True[Well_Mask]
        
    g_calc_total = g_calc_total + g_mean
    return Density_Model_3D, g_calc_total

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="G-Invert Pro 2.0", layout="wide")
st.title("🌏 G-Invert Pro 2.0: Interactive 3D Constrained Inversion")
st.markdown("*Dibuat oleh Hidayat untuk kemajuan geofisika Indonesia 🇮🇩*")

if 'inversion_done' not in st.session_state: st.session_state.inversion_done = False

# --- UI SIDEBAR: Parameter ---
st.sidebar.header("1. Model Geometry")
nx = st.sidebar.number_input("Nx (Blok X)", value=50, step=10)
ny = st.sidebar.number_input("Ny (Blok Y)", value=50, step=10)
nz = st.sidebar.number_input("Nz (Layer Z)", value=10, step=2)
z_top = st.sidebar.number_input("Z Top (m)", value=0.0)
z_bottom = st.sidebar.number_input("Z Bottom (m)", value=5000.0)

st.sidebar.header("2. Inversion Parameters")
beta_weight = st.sidebar.slider("Depth Weighting (Beta)", 0.0, 3.0, 1.5, 0.1)
smooth_sigma = st.sidebar.slider("Horizontal Smoothness", 0.0, 5.0, 2.0, 0.1)
max_dev = st.sidebar.slider("Max Density Deviation (+/- g/cc)", 0.05, 0.50, 0.20, 0.05)
max_iter = st.sidebar.number_input("Max Iterasi", value=15)

# --- MAIN AREA ---
st.header("Step 1: Upload Data & Well Controls")
uploaded_file = st.file_uploader("Upload Data Anomali (CSV/TXT: X, Y, Elev, Anomali mGal)", type=['csv', 'txt'])

st.markdown("### Optional: Well Control Data")
st.info("Format CSV Log Sumur: Well_Name, X, Y, Depth, RHOB")

# --- FITUR BARU: MULTI-WELL IMPORTER ---
well_files = st.file_uploader("Saya sarankan data-data sumur ddigabungkan aja dalam 1 file CSV dengan 5 kolom yang berisi (ID, X,Y,depth, density), pastikan tidak ada kolom yang kosong yes..", type=['csv'], accept_multiple_files=True)

well_data = None
if well_files: # Jika list file tidak kosong
    dfs = []
    for file in well_files:
        df_temp = pd.read_csv(file, sep=None, engine='python')
        dfs.append(df_temp)
    # Jahit semua file sumur menjadi satu dataframe master
    well_data = pd.concat(dfs, ignore_index=True)
    
    nama_sumur = well_data.iloc[:, 0].unique()
    st.success(f"✅ Data sumur berhasil dimuat! Mendeteksi {len(nama_sumur)} sumur kontrol: {', '.join(nama_sumur)}")
    with st.expander("🔍 Preview Gabungan Data Sumur"):
        st.dataframe(well_data, use_container_width=True)
# ---------------------------------------

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    cols = df.columns
    x_data, y_data, elev_data, anomaly_data = df[cols[0]].values, df[cols[1]].values, df[cols[2]].values, df[cols[3]].values

    XI, YI, g_grid, dx, dy, z_bounds = setup_3d_grid(x_data, y_data, anomaly_data, nx, ny, nz, z_top, z_bottom)
    elev_grid = griddata((x_data, y_data), elev_data, (XI, YI), method='cubic')

    st.markdown("### Data Preview & Mesh Geometry")
    col_map, col_mesh = st.columns(2)
    with col_map:
        st.write("2D Anomaly Map (Interpolated)")
        fig2d, ax2d = plt.subplots(figsize=(6, 5))
        c2d = ax2d.contourf(XI, YI, g_grid, levels=50, cmap='jet')
        plt.colorbar(c2d, ax=ax2d, label='Anomali (mGal)')
        
        if well_data is not None:
            nama_sumur = well_data.iloc[:, 0].unique()
            for w_name in nama_sumur:
                df_w = well_data[well_data.iloc[:, 0] == w_name]
                wx, wy = df_w.iloc[0, 1], df_w.iloc[0, 2] 
                ax2d.plot(wx, wy, '^', color='white', markeredgecolor='black', markersize=10)
                ax2d.text(wx, wy, f'  {w_name}', color='white', weight='bold', fontsize=9, 
                          path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
                          
        ax2d.set_xlabel("X (m)"); ax2d.set_ylabel("Y (m)")
        st.pyplot(fig2d)

    with col_mesh:
        st.write(f"3D Grid Design: {nx}x{ny}x{nz} Blocks")
        fig3d = plot_3d_grid_wireframe_with_topo(nx, ny, nz, z_top, z_bottom, XI, YI, elev_grid, g_grid)
        st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("### Step 2: Define 1D Initial Density Model $\\rho_0(z)$")
    default_depths = np.linspace(z_top, z_bottom, 5)
    default_rho = pd.DataFrame({"Depth (m)": default_depths, "Density (g/cc)": [2.2, 2.4, 2.6, 2.7, 2.8]})
    rho_1d_model = st.data_editor(default_rho, num_rows="dynamic", use_container_width=True)

    st.markdown("---")
    if st.button("🚀 Run 3D Inversion", use_container_width=True):
        with st.spinner("Executing Local Corrections & Matrix Inversion..."):
            Density_3D, g_calc = gravity_inversion_3d(
                XI, YI, g_grid, dx, dy, z_bounds, nx, ny, nz, rho_1d_model, beta_weight, max_iter, smooth_sigma, max_dev, well_data
            )
            
            st.session_state['Density_3D'] = Density_3D
            st.session_state['XI'] = XI
            st.session_state['YI'] = YI
            st.session_state['z_bounds'] = z_bounds
            st.session_state['g_obs'] = g_grid
            st.session_state['g_calc'] = g_calc
            st.session_state['elev_grid'] = elev_grid
            st.session_state['dx'] = dx
            st.session_state['dy'] = dy
            st.session_state['well_data'] = well_data 
            st.session_state.inversion_done = True 

# ==========================================
# VISUALISASI QC & SEMUA MODUL EXPORT
# ==========================================
if st.session_state.inversion_done:
    st.markdown("---")
    st.header("Step 3: Quality Control & Slicing Analysis")
    
    XI, YI, z_bounds = st.session_state.XI, st.session_state.YI, st.session_state.z_bounds
    Density_3D = st.session_state.Density_3D
    g_obs, g_calc = st.session_state.g_obs, st.session_state.g_calc
    elev_grid = st.session_state.elev_grid
    dx, dy = st.session_state.dx, st.session_state.dy
    z_centers = (z_bounds[:-1] + z_bounds[1:]) / 2
    
    ny_shape, nx_shape, nz_s = Density_3D.shape
    x_min, x_max = float(XI.min()), float(XI.max())
    y_min, y_max = float(YI.min()), float(YI.max())
    
    st.markdown("**Visual & Extent Settings (Sumbu Vertikal Z):**")
    col_v1, col_v2, col_z1, col_z2 = st.columns(4)
    with col_v1: vmin_plot = st.number_input("Warna Min", value=2.0, step=0.1)
    with col_v2: vmax_plot = st.number_input("Warna Max", value=3.0, step=0.1)
    with col_z1: z_plot_top = st.number_input("Atap Plot Z (m)", value=-1000.0, step=500.0)
    with col_z2: z_plot_bottom = st.number_input("Dasar Plot Z (m)", value=float(z_bounds[-1]), step=500.0)
    
    show_well_plot = st.checkbox("Tampilkan Log Sumur di Penampang", value=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        profil_pilihan = st.radio("Orientasi Slicing:", ("Profil XZ (B-T)", "Profil YZ (S-U)", "Arbitrary (Garis Bebas)"))
        
        if profil_pilihan == "Profil XZ (B-T)":
            idx = st.slider("Geser Indeks Y:", 0, ny_shape - 1, ny_shape // 2)
            
            st.markdown("**Batas Horisontal Sumbu X:**")
            c_h1, c_h2 = st.columns(2)
            with c_h1: x_start = st.slider("Min X (m)", min_value=x_min, max_value=x_max, value=x_min, format="%.0f")
            with c_h2: x_end = st.slider("Max X (m)", min_value=x_min, max_value=x_max, value=x_max, format="%.0f")
            
            if x_start > x_end: x_start, x_end = x_end, x_start 
            
            full_x = XI[idx, :]
            mask = (full_x >= x_start) & (full_x <= x_end)
            if not np.any(mask): mask[0] = True 
            
            jarak_x = full_x[mask]
            o_curve, c_curve = g_obs[idx, mask], g_calc[idx, mask]
            slice_den = Density_3D[idx, mask, :]
            slice_elev = elev_grid[idx, mask]
            xlabel, title = "Jarak X (m)", f"Profil XZ di Y={YI[idx, 0]:.0f}"
            
            line_p1, line_p2 = (jarak_x[0], YI[idx, 0]), (jarak_x[-1], YI[idx, 0])
            
        elif profil_pilihan == "Profil YZ (S-U)":
            idx = st.slider("Geser Indeks X:", 0, nx_shape - 1, nx_shape // 2)
            
            st.markdown("**Batas Horisontal Sumbu Y:**")
            c_h1, c_h2 = st.columns(2)
            with c_h1: y_start = st.slider("Min Y (m)", min_value=y_min, max_value=y_max, value=y_min, format="%.0f")
            with c_h2: y_end = st.slider("Max Y (m)", min_value=y_min, max_value=y_max, value=y_max, format="%.0f")
            
            if y_start > y_end: y_start, y_end = y_end, y_start
            
            full_y = YI[:, idx]
            mask = (full_y >= y_start) & (full_y <= y_end)
            if not np.any(mask): mask[0] = True
            
            jarak_x = full_y[mask]
            o_curve, c_curve = g_obs[mask, idx], g_calc[mask, idx]
            slice_den = Density_3D[mask, idx, :]
            slice_elev = elev_grid[mask, idx]
            xlabel, title = "Jarak Y (m)", f"Profil YZ di X={XI[0, idx]:.0f}"
            
            line_p1, line_p2 = (XI[0, idx], jarak_x[0]), (XI[0, idx], jarak_x[-1])
            
        else:
            st.markdown("**Koordinat Garis Sayatan:**")
            c_a1, c_a2 = st.columns(2)
            with c_a1: xa = st.slider("X Awal (A)", min_value=x_min, max_value=x_max, value=x_min, format="%.0f")
            with c_a2: ya = st.slider("Y Awal (A)", min_value=y_min, max_value=y_max, value=y_min, format="%.0f")
            
            c_b1, c_b2 = st.columns(2)
            with c_b1: xb = st.slider("X Akhir (B)", min_value=x_min, max_value=x_max, value=x_max, format="%.0f")
            with c_b2: yb = st.slider("Y Akhir (B)", min_value=y_min, max_value=y_max, value=y_max, format="%.0f")

            num_points = 100
            line_x = np.linspace(xa, xb, num_points)
            line_y = np.linspace(ya, yb, num_points)
            jarak_x = np.sqrt((line_x - xa)**2 + (line_y - ya)**2)
            
            x_1d, y_1d = XI[0, :], YI[:, 0]
            points_2d = np.array([line_y, line_x]).T
            
            interp_obs = RegularGridInterpolator((y_1d, x_1d), g_obs, bounds_error=False, fill_value=np.nan)
            interp_calc = RegularGridInterpolator((y_1d, x_1d), g_calc, bounds_error=False, fill_value=np.nan)
            o_curve = interp_obs(points_2d)
            c_curve = interp_calc(points_2d)
            
            interp_3d = RegularGridInterpolator((y_1d, x_1d, z_centers), Density_3D, bounds_error=False, fill_value=np.nan)
            
            slice_den = np.zeros((num_points, len(z_centers)))
            for i in range(len(z_centers)):
                points_3d = np.array([line_y, line_x, np.full(num_points, z_centers[i])]).T
                slice_den[:, i] = interp_3d(points_3d)

            interp_elev = RegularGridInterpolator((y_1d, x_1d), elev_grid, bounds_error=False, fill_value=np.nan)
            slice_elev = interp_elev(points_2d) 
            
            xlabel, title = "Jarak dari Titik A (m)", "Sayatan Arbitrary (A-B)"
            line_p1, line_p2 = (xa, ya), (xb, yb)

        st.markdown("<br>**📍 Posisi Lintasan Slicing:**", unsafe_allow_html=True)
        fig_inset, ax_inset = plt.subplots(figsize=(4, 4))
        ax_inset.contourf(XI, YI, g_obs, levels=30, cmap='jet', alpha=0.5)
        
        ax_inset.plot([line_p1[0], line_p2[0]], [line_p1[1], line_p2[1]], color='red', linewidth=3, linestyle='--')
        ax_inset.plot(line_p1[0], line_p1[1], 'ro'); ax_inset.plot(line_p2[0], line_p2[1], 'ro')
        ax_inset.text(line_p1[0], line_p1[1], ' A', color='red', weight='bold', va='bottom')
        ax_inset.text(line_p2[0], line_p2[1], ' B', color='red', weight='bold', va='bottom')

        ax_inset.set_xticks([]); ax_inset.set_yticks([])
        ax_inset.set_title("Top-Down View", fontsize=10)
        st.pyplot(fig_inset)

    with col2:
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(2, 1, height_ratios=[1.5, 3], hspace=0.15)
        
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(jarak_x, o_curve, 'ko', markersize=4, label='Observed')
        ax0.plot(jarak_x, c_curve, 'r-', linewidth=2, label='Calculated')
        ax0.set_title(title, fontweight='bold'); ax0.legend(); ax0.grid(True, linestyle='--')
        
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        levels = np.linspace(vmin_plot, vmax_plot, 100)
        
        #c = ax1.imshow(slice_den.T, extent=[jarak_x.min(), jarak_x.max(), z_bounds[-1], z_bounds[0]], aspect='auto', cmap='jet', vmin=vmin_plot, vmax=vmax_plot, interpolation='bicubic')
        c = ax1.pcolormesh(jarak_x, z_centers, slice_den.T, cmap='jet', vmin=vmin_plot, vmax=vmax_plot, shading='nearest')
        ax1.invert_yaxis()
        ax1.set_xlabel(xlabel); ax1.set_ylabel("Depth (m)")
        plt.colorbar(c, ax=ax1, orientation='horizontal', pad=0.15, label='Density (g/cc)')
        
        if st.session_state.get('well_data') is not None and show_well_plot:
            wd = st.session_state.well_data
            for w_name in wd.iloc[:, 0].unique():
                df_w = wd[wd.iloc[:, 0] == w_name]
                wx, wy = df_w.iloc[0, 1], df_w.iloc[0, 2]
                
                plot_well = False
                well_x_pos = 0
                
                if profil_pilihan == "Profil XZ (B-T)" and abs(wy - YI[idx, 0]) <= dy:
                    if jarak_x.min() <= wx <= jarak_x.max():
                        plot_well = True
                        well_x_pos = wx
                elif profil_pilihan == "Profil YZ (S-U)" and abs(wx - XI[0, idx]) <= dx:
                    if jarak_x.min() <= wy <= jarak_x.max():
                        plot_well = True
                        well_x_pos = wy
                elif profil_pilihan == "Arbitrary (Garis Bebas)":
                    p1, p2, p3 = np.array([xa, ya]), np.array([xb, yb]), np.array([wx, wy])
                    d_line = np.linalg.norm(p2-p1)
                    if d_line > 0:
                        jarak_ke_garis = np.abs(np.cross(p2-p1, p3-p1)) / d_line
                        if jarak_ke_garis <= max(dx, dy):
                            plot_well = True
                            well_x_pos = np.dot(p3-p1, p2-p1) / d_line
                
                if plot_well:
                    w_depths = df_w.iloc[:, 3].values
                    w_rhob = df_w.iloc[:, 4].values
                    
                    sorted_idx = np.argsort(w_depths)
                    w_depths = w_depths[sorted_idx]
                    w_rhob = w_rhob[sorted_idx]
                    
                    norm = plt.Normalize(vmin_plot, vmax_plot)
                    cmap = plt.get_cmap('jet')
                    
                    strip_w = (jarak_x.max() - jarak_x.min()) * 0.015
                    
                    for i in range(len(w_depths)-1):
                        color = cmap(norm(w_rhob[i]))
                        ax1.fill_between([well_x_pos - strip_w/2, well_x_pos + strip_w/2], 
                                         w_depths[i], w_depths[i+1], color=color, zorder=10)
                    
                    ax1.plot([well_x_pos - strip_w/2, well_x_pos - strip_w/2], [w_depths.min(), w_depths.max()], 'k-', linewidth=1, zorder=11)
                    ax1.plot([well_x_pos + strip_w/2, well_x_pos + strip_w/2], [w_depths.min(), w_depths.max()], 'k-', linewidth=1, zorder=11)
                    
                    ax1.text(well_x_pos, w_depths.min() - 50, f' {w_name}', color='white', weight='bold', verticalalignment='bottom', 
                             path_effects=[path_effects.withStroke(linewidth=2, foreground='black')], zorder=12)
        
        z_topo_line = -1.0 * slice_elev 
        ax1.plot(jarak_x, z_topo_line, color='black', linewidth=2, label='Permukaan Topografi')
        ax1.fill_between(jarak_x, z_topo_line, -99999, color='white', zorder=5) 
        
        ax1.set_ylim(z_plot_bottom, z_plot_top)
        ax1.set_xlim(jarak_x.min(), jarak_x.max())
        
        st.pyplot(fig)

    # ==========================================
    # MODUL 4: EXPORT TO TXT (FULL 3D CUBE)
    # ==========================================
    st.markdown("---")
    st.header("Step 4: Export Full 3D Cube Model")

    x_3d = np.repeat(XI[:, :, np.newaxis], nz_s, axis=2)
    y_3d = np.repeat(YI[:, :, np.newaxis], nz_s, axis=2)
    z_3d = np.tile(z_centers, (ny_shape, nx_shape, 1))
    
    df_export = pd.DataFrame({
        'X': x_3d.flatten(), 'Y': y_3d.flatten(), 'Depth': z_3d.flatten(), 'Density': Density_3D.flatten()
    })
    txt_data = df_export.to_csv(index=False, sep='\t')
    st.success(f"✅ Matriks 3D Cube dirangkum! Total: **{len(df_export):,} titik**.")
    
    st.download_button("💾 Download Full 3D Cube (.txt)", data=txt_data, file_name="Full_3D_Density_Cube.txt", mime="text/plain", use_container_width=True)