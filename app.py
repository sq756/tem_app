import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import time

from core.physics_engine import KinematicDiffractionModel
from core.perception import preprocess_image, extract_peaks, align_to_reciprocal_space
from core.optimizer import weighted_chamfer_loss, physical_constraints_penalty
from core.viz_engine import plot_diffraction_overlay
from core.scale_reader import detect_scale_bar

# 1. 頁面配置
st.set_page_config(page_title="DeepDiffra AI Analysis Engine", layout="wide")
st.title("🔬 DeepDiffra: Differentiable Physics Analysis for TEM")

# 2. 側邊欄配置
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload TEM HRTEM Image", type=["png", "jpg", "jpeg", "tif"])

st.sidebar.header("Scale Calibration")
scale_pixel_w = st.sidebar.number_input("Scale Bar Length (pixels)", value=100.0)
scale_phys_v = st.sidebar.number_input("Scale Physical Size (nm)", value=10.0)
pixel_size_ang = (scale_phys_v * 10.0) / (scale_pixel_w + 1e-6)
st.sidebar.caption(f"Calculated Pixel Size: **{pixel_size_ang:.4f} Å/px**")

st.sidebar.header("Optimization Parameters")
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.05, format="%.3f")
epochs = st.sidebar.slider("Epochs", 10, 500, 200)
hkl_range = st.sidebar.slider("HKL Grid Range", 1, 5, 3)

st.sidebar.subheader("User Guidance (Pre-alignment)")
init_psi = st.sidebar.slider("Init In-plane Rotation (psi)", 0.0, 180.0, 0.0)
lock_orthogonal = st.sidebar.checkbox("Lock Angles to 90° (Orthogonal)", value=True)

start_fitting = st.sidebar.button("🚀 Start AI Fitting")

# 3. 主界面佈局 (Task 6.1: Tabs 重構)
col1, col2 = st.columns([1, 1.2])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with col1:
        st.subheader("Experimental Data")
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)
        
        with st.spinner("Analyzing image..."):
            if 'scale_detected' not in st.session_state:
                scale_info = detect_scale_bar(tmp_path)
                if scale_info["detected"]:
                    st.session_state.scale_pixel_w = scale_info["pixel_width"]
                    if scale_info["physical_value"]:
                        st.session_state.scale_phys_v = scale_info["physical_value"]
                    st.success(f"Detected Scale Bar: {st.session_state.scale_pixel_w} px")
                st.session_state.scale_detected = True

            mag_spectrum = preprocess_image(tmp_path, patch_size=256)
            peaks_px = extract_peaks(mag_spectrum, num_peaks=15)
            fig_init = plot_diffraction_overlay(mag_spectrum, peaks_px, None)
            st.pyplot(fig_init)

    with col2:
        # Task 6.1: 分頁展示
        tab1, tab2, tab3 = st.tabs(["🎯 Fitting Monitor", "👁️ Perception Pipeline", "⚛️ Physics Engine"])
        
        with tab1:
            plot_placeholder = st.empty()
            metrics_placeholder = st.empty()
        
        with tab2:
            st.info("Visualizing intermediate CV steps...")
            st.image(mag_spectrum / mag_spectrum.max(), caption="FFT Magnitude Spectrum (Normalized)")
            st.write(f"Detected Peaks Coordinates (px): \n {peaks_px}")

        with tab3:
            st.markdown("### White-box Physics Engine")
            st.write("How the 'Red Cross' is generated:")
            st.latex(r"g_{hkl} = h\mathbf{a}^* + k\mathbf{b}^* + l\mathbf{c}^*")
            st.latex(r"F_{hkl} = \sum_{j} f_j \exp[-2\pi i (hx_j + ky_j + lz_j)]")
            st.latex(r"I_{hkl} \propto |F_{hkl}|^2 \cdot \text{sinc}^2(\pi s t)")
            
            # 動態渲染區
            physics_metrics = st.empty()
            matrix_display = st.empty()

        if start_fitting:
            model = KinematicDiffractionModel(
                a=2.5, b=2.5, c=2.5, 
                alpha=90.0, beta=90.0, gamma=90.0,
                theta=0.0, phi=0.0, psi=init_psi,
                s_max=0.05
            )
            
            with torch.no_grad():
                init_scale = 1.0 / (pixel_size_ang + 1e-10)
                model.scale_factor.fill_(init_scale)
            
            optimizer = torch.optim.Adam([
                {'params': [model.cell_lengths], 'lr': 0.05},
                {'params': [model.cell_angles, model.euler_angles], 'lr': lr * 10.0},
                {'params': [model.scale_factor], 'lr': 2.0}
            ])
            
            g_exp_px = torch.tensor(peaks_px.copy(), dtype=torch.float32)
            center_px_tensor = torch.tensor([128.0, 128.0], dtype=torch.float32)
            
            progress_bar = st.progress(0)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(hkl_range=hkl_range, center_px=center_px_tensor)
                pred_px = output['coords']
                weights_calc = output['weights']
                
                loss_p_to_g, loss_g_to_p = weighted_chamfer_loss(g_exp_px, pred_px, weights_calc)
                loss_p = physical_constraints_penalty(model)
                
                loss_lock = torch.tensor(0.0)
                if lock_orthogonal:
                    cur_angles = model.cell_angles * (180.0 / 3.1415926535)
                    loss_lock = torch.sum((cur_angles - 90.0)**2) * 10000.0
                
                total_loss = loss_p_to_g + 5.0 * loss_g_to_p + 1.0 * loss_p + loss_lock
                total_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.cell_lengths.clamp_(min=1.5)
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    a, b, c = model.cell_lengths.detach().numpy()
                    alpha, beta, gamma = np.rad2deg(model.cell_angles.detach().numpy())
                    scale = model.scale_factor.item()
                    psi_deg = np.rad2deg(model.euler_angles[2].item())
                    
                    # 更新 Tab 1 (Monitor)
                    with tab1:
                        metrics_placeholder.markdown(f"""
                        **Epoch:** {epoch+1}/{epochs} | **Total Loss:** `{total_loss.item():.6f}`
                        
                        **Lattice Parameters:**
                        - a: `{a:.4f}` | b: `{b:.4f}` | c: `{c:.4f}`
                        - α: `{alpha:.2f}°` | β: `{beta:.2f}°` | γ: `{gamma:.2f}°`
                        - **Scale Factor:** `{scale:.2f} px/(1/A)`
                        - **Current psi:** `{psi_deg:.2f}°`
                        """)
                        fig_fit = plot_diffraction_overlay(mag_spectrum, peaks_px, pred_px.detach().numpy())
                        plot_placeholder.pyplot(fig_fit, clear_figure=True)
                    
                    # 更新 Tab 3 (Physics)
                    recip_base = model.get_reciprocal_base().detach().numpy()
                    rot_matrix = model.get_rotation_matrix().detach().numpy()
                    
                    with tab3:
                        physics_metrics.markdown(f"""
                        **Current Reciprocal Vectors (1/Å):**
                        - a*: `[{recip_base[0,0]:.4f}, {recip_base[0,1]:.4f}, {recip_base[0,2]:.4f}]`
                        - b*: `[{recip_base[1,0]:.4f}, {recip_base[1,1]:.4f}, {recip_base[1,2]:.4f}]`
                        - c*: `[{recip_base[2,0]:.4f}, {recip_base[2,1]:.4f}, {recip_base[2,2]:.4f}]`
                        """)
                        matrix_display.write("Current Rotation Matrix (Euler):")
                        matrix_display.write(rot_matrix)

                    time.sleep(0.05)
                progress_bar.progress((epoch + 1) / epochs)
            
            st.success(f"Fitting Complete! Final a={a:.2f}, b={b:.2f}, c={c:.2f}")
            
    os.remove(tmp_path)
else:
    st.info("Please upload a TEM image to begin.")
