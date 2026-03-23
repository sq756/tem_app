import torch
import torch.nn as nn
import numpy as np

class KinematicDiffractionModel(nn.Module):
    """
    DeepDiffra 物理引擎基类：运动学衍射模型。
    负责初始化可导的晶体结构参数与实验几何参数。
    
    参数说明:
        a, b, c: 晶格常数 (Angstrom)
        alpha, beta, gamma: 晶格夹角 (Degree)
        theta, phi, psi: 欧拉角 (Degree)
        s_max: 激发误差阈值 (1/Angstrom)
    """
    def __init__(self, 
                 a=5.43, b=5.43, c=5.43, 
                 alpha=90.0, beta=90.0, gamma=90.0,
                 theta=0.0, phi=0.0, psi=0.0,
                 s_max=0.01,
                 device='cpu', dtype=torch.float32):
        super(KinematicDiffractionModel, self).__init__()
        
        self.device = device
        self.dtype = dtype

        # 1. 晶格常数 (Lattice constants in Angstroms)
        self.cell_lengths = nn.Parameter(
            torch.tensor([a, b, c], dtype=dtype, device=device)
        )
        
        # 2. 晶格夹角 (Lattice angles: Degrees -> Radians)
        cell_angles_rad = np.deg2rad([alpha, beta, gamma])
        self.cell_angles = nn.Parameter(
            torch.tensor(cell_angles_rad, dtype=dtype, device=device)
        )
        
        # 3. 欧拉角 (Orientation Euler angles: Degrees -> Radians)
        euler_angles_rad = np.deg2rad([theta, phi, psi])
        self.euler_angles = nn.Parameter(
            torch.tensor(euler_angles_rad, dtype=dtype, device=device)
        )
        
        # 4. 激发误差阈值 (Excitation error threshold)
        self.s_max = nn.Parameter(
            torch.tensor([s_max], dtype=dtype, device=device)
        )

        # Task 4.7: 新增可學習比例尺 (Reciprocal Units -> Pixels)
        self.scale_factor = nn.Parameter(
            torch.tensor([50.0], dtype=dtype, device=device)
        )

    def get_reciprocal_base(self):
        """
        计算倒易空間基向量 (Reciprocal Lattice Base Vectors)。
        """
        a, b, c = self.cell_lengths
        alpha, beta, gamma = self.cell_angles

        ax = a
        ay = torch.zeros_like(a)
        az = torch.zeros_like(a)
        
        bx = b * torch.cos(gamma)
        by = b * torch.sin(gamma)
        bz = torch.zeros_like(b)
        
        cx = c * torch.cos(beta)
        cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
        cz = torch.sqrt(torch.max(c**2 - cx**2 - cy**2, torch.tensor(1e-10, device=self.device)))

        direct_matrix = torch.stack([
            torch.stack([ax, ay, az]),
            torch.stack([bx, by, bz]),
            torch.stack([cx, cy, cz])
        ])

        reciprocal_matrix = torch.inverse(direct_matrix).t()
        
        return reciprocal_matrix

    def generate_3d_grid(self, h_max=5, k_max=5, l_max=5):
        """
        生成 3D 倒易格點陣列 (3D Reciprocal Lattice Points)。
        """
        h_range = torch.arange(-h_max, h_max + 1, device=self.device, dtype=self.dtype)
        k_range = torch.arange(-k_max, k_max + 1, device=self.device, dtype=self.dtype)
        l_range = torch.arange(-l_max, l_max + 1, device=self.device, dtype=self.dtype)
        
        h, k, l = torch.meshgrid(h_range, k_range, l_range, indexing='ij')
        hkl_indices = torch.stack([h.reshape(-1), k.reshape(-1), l.reshape(-1)], dim=1)
        
        # Task 4.6: 移除 (0, 0, 0) 點
        non_zero_mask = torch.any(hkl_indices != 0, dim=1)
        hkl_indices = hkl_indices[non_zero_mask]
        
        reciprocal_base = self.get_reciprocal_base()
        g_3d = torch.matmul(hkl_indices, reciprocal_base)
        
        return g_3d

    def get_rotation_matrix(self):
        """
        根據歐拉角 (theta, phi, psi) 構建 3x3 旋轉矩陣。
        """
        theta, phi, psi = self.euler_angles
        zero = torch.zeros_like(theta)
        one = torch.ones_like(theta)

        Rx = torch.stack([
            torch.stack([one,  zero,  zero]),
            torch.stack([zero, torch.cos(theta), -torch.sin(theta)]),
            torch.stack([zero, torch.sin(theta),  torch.cos(theta)])
        ])
        
        Ry = torch.stack([
            torch.stack([torch.cos(phi), zero, torch.sin(phi)]),
            torch.stack([zero, one, zero]),
            torch.stack([-torch.sin(phi), zero, torch.cos(phi)])
        ])
        
        Rz = torch.stack([
            torch.stack([torch.cos(psi), -torch.sin(psi), zero]),
            torch.stack([torch.sin(psi), torch.cos(psi), zero]),
            torch.stack([zero, zero, one])
        ])
        
        R = torch.matmul(Rz, torch.matmul(Ry, Rx))
        
        return R

    def rotate_to_zone_axis(self, g_3d):
        R = self.get_rotation_matrix()
        g_rotated = torch.matmul(g_3d, R.t())
        return g_rotated

    def apply_ewald_sphere_truncation(self, g_rotated):
        gz = g_rotated[:, 2:3]
        scaled_gz = gz / (self.s_max + 1e-10)
        weights = torch.sinc(scaled_gz) ** 2
        return weights

    def forward(self, hkl_range=5, center_px=None):
        """
        前向傳播完整流程。
        Task 4.7: 輸出映射到像素空間。
        """
        g_3d = self.generate_3d_grid(hkl_range, hkl_range, hkl_range)
        g_rotated = self.rotate_to_zone_axis(g_3d)
        weights = self.apply_ewald_sphere_truncation(g_rotated)
        
        # 取 X, Y 分量 (Reciprocal space)
        coords_recip = g_rotated[:, :2]
        
        # Task 4.7: 像素空間映射 pred_px = (g_recip * scale_factor) + center_px
        # 如果沒有傳入 center_px，默認輸出 (0,0) 為中心的偏移量
        if center_px is None:
            center_px = torch.tensor([0.0, 0.0], device=self.device, dtype=self.dtype)
        
        coords_px = (coords_recip * self.scale_factor) + center_px
        
        return {
            "coords": coords_px,
            "weights": weights,
            "recip_coords": coords_recip
        }
