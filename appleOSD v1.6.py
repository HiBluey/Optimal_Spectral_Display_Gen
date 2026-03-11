import sys
import os

# 1. 环境依赖加载与报错保护
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import colour
except ImportError as e:
    print("="*50)
    print(f"【环境报错】缺少必要的第三方库: {e}")
    print("请打开命令行，运行以下命令安装所需依赖：")
    print("pip install numpy pandas matplotlib colour-science")
    print("="*50)
    input("按回车键退出程序...")
    sys.exit(1)

# 定义可选的标准色空间及目标色度 (x, y) 
COLOR_SPACES = {
    "1": {"name": "Rec.709-sRGB", "R": [0.640, 0.330], "G": [0.300, 0.600], "B": [0.150, 0.060], "W": [0.3127, 0.3290]},
    "2": {"name": "DCI P3",       "R": [0.680, 0.320], "G": [0.265, 0.690], "B": [0.150, 0.060], "W": [0.3140, 0.3510]},
    "3": {"name": "P3D65",        "R": [0.680, 0.320], "G": [0.265, 0.690], "B": [0.150, 0.060], "W": [0.3127, 0.3290]},
    "4": {"name": "Adobe RGB",    "R": [0.640, 0.330], "G": [0.210, 0.710], "B": [0.150, 0.060], "W": [0.3127, 0.3290]},
    "5": {"name": "Rec2020",      "R": [0.708, 0.292], "G": [0.170, 0.797], "B": [0.131, 0.046], "W": [0.3127, 0.3290]}
}

def load_custom_cmf(filepath, name):
    """从本地 CSV 文件加载 CMF 并封装为 colour 对象"""
    if not os.path.exists(filepath):
        print(f"⚠️ [提示] 当前目录下未找到自定义 CMF 文件 '{filepath}'，将跳过 {name} 的计算。")
        return None
    try:
        df = pd.read_csv(filepath)
        # 智能提取前4列数值（假设第1列为波长，第2,3,4列为 x, y, z）
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 4:
            print(f"❌ [错误] '{filepath}' 格式异常，需要至少包含4列数值（波长, x_bar, y_bar, z_bar）。")
            return None
            
        wavelengths = num_df.iloc[:, 0].values
        cmf_values = num_df.iloc[:, 1:4].values
        
        # 将数据组装为字典供 colour 库识别
        data_dict = {w: v for w, v in zip(wavelengths, cmf_values)}
        custom_cmf = colour.MultiSpectralDistributions(data_dict, name=name)
        print(f"✅ [成功] 已加载外部 CMF: {name}")
        return custom_cmf
    except Exception as e:
        print(f"❌ [错误] 读取外部 CMF '{filepath}' 时发生异常: {e}")
        return None

def get_cmf_data(cmf_obj, wavelengths):
    """将CMF对象对齐/插值到指定的波长范围并返回数组"""
    step = np.round(wavelengths[1] - wavelengths[0], 2)
    cmf_shape = colour.SpectralShape(wavelengths[0], wavelengths[-1], step)
    cmf_copy = cmf_obj.copy()
    cmf_copy.align(cmf_shape)
    return cmf_copy.values

def find_true_bounds_high_res(target_xy, cmf_values, wavelengths):
    """内部高精度引擎：用来寻找数学上真实的连续过渡波长"""
    target_x, target_y = target_xy
    best_diff = float('inf')
    best_l1, best_l2, best_type = wavelengths[0], wavelengths[0], 1
    
    n = len(wavelengths)
    cmf_cumsum = np.vstack([np.zeros(3), np.cumsum(cmf_values, axis=0)])
    sum_all = cmf_cumsum[-1]
    
    for i in range(n):
        # Type I 计算
        XYZ_I = cmf_cumsum[i+1:n+1] - cmf_cumsum[i]
        sum_XYZ_I = np.sum(XYZ_I, axis=1)
        
        valid_I = sum_XYZ_I > 1e-9
        diff_I = np.full(n - i, np.inf)
        if np.any(valid_I):
            x_I = XYZ_I[valid_I, 0] / sum_XYZ_I[valid_I]
            y_I = XYZ_I[valid_I, 1] / sum_XYZ_I[valid_I]
            diff_I[valid_I] = (x_I - target_x)**2 + (y_I - target_y)**2
            
        min_idx_I = np.argmin(diff_I)
        if diff_I[min_idx_I] < best_diff:
            best_diff = diff_I[min_idx_I]
            best_l1 = wavelengths[i]
            best_l2 = wavelengths[i + min_idx_I]
            best_type = 1
            
        # Type II 计算
        XYZ_II = sum_all - XYZ_I
        sum_XYZ_II = np.sum(XYZ_II, axis=1)
        
        valid_II = sum_XYZ_II > 1e-9
        diff_II = np.full(n - i, np.inf)
        if np.any(valid_II):
            x_II = XYZ_II[valid_II, 0] / sum_XYZ_II[valid_II]
            y_II = XYZ_II[valid_II, 1] / sum_XYZ_II[valid_II]
            diff_II[valid_II] = (x_II - target_x)**2 + (y_II - target_y)**2
            
        min_idx_II = np.argmin(diff_II)
        if diff_II[min_idx_II] < best_diff:
            best_diff = diff_II[min_idx_II]
            best_l1 = wavelengths[i]
            best_l2 = wavelengths[i + min_idx_II]
            best_type = 2
            
    return best_l1, best_l2, best_type

def calculate_xy(spectrum, cmf_values):
    XYZ = np.dot(spectrum, cmf_values)
    sum_XYZ = np.sum(XYZ)
    if sum_XYZ == 0:
        return 0, 0
    return XYZ[0]/sum_XYZ, XYZ[1]/sum_XYZ

def calculate_white_spectrum(R_spec, G_spec, B_spec, target_W_xy, cmf_values):
    XYZ_R = np.dot(R_spec, cmf_values)
    XYZ_G = np.dot(G_spec, cmf_values)
    XYZ_B = np.dot(B_spec, cmf_values)
    
    xw, yw = target_W_xy
    Xw = xw / yw
    Yw = 1.0
    Zw = (1 - xw - yw) / yw
    Target_XYZ = np.array([Xw, Yw, Zw])
    
    M = np.column_stack([XYZ_R, XYZ_G, XYZ_B])
    coeffs = np.linalg.solve(M, Target_XYZ)
    
    W_spec = coeffs[0]*R_spec + coeffs[1]*G_spec + coeffs[2]*B_spec
    return W_spec / np.max(W_spec)

def main():
    print("=== 最优光谱(Optimal Spectra)生成工具 v1.6 ===")
    print("请选择目标标准色彩空间：")
    for key, val in COLOR_SPACES.items():
        print(f"[{key}] {val['name']}")
    
    choice = input("\n请输入对应序号(1-5): ").strip()
    if choice not in COLOR_SPACES:
        print("输入无效，程序终止。")
        return
        
    space_info = COLOR_SPACES[choice]
    space_name = space_info["name"]
    print(f"\n✅ 您选择了：{space_name}")

    # 刻度设置
    fine_wavelengths = np.arange(390, 760.1, 0.1)
    base_wavelengths = np.arange(390, 761, 1)
    
    # 动态构建 CMFS_DICT
    CMFS_DICT = {}
    
    # 1. 加载库内置的必要标准
    if 'CIE 1931 2 Degree Standard Observer' in colour.MSDS_CMFS:
        CMFS_DICT["CIE1931 2°"] = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    else:
        print("❌ [错误] 缺失核心的 CIE 1931 2° 观察者数据，无法进行推导。")
        return
        
    if 'Judd 1951 2 Degree Standard Observer' in colour.MSDS_CMFS:
        CMFS_DICT["JUDD 2°"] = colour.MSDS_CMFS['Judd 1951 2 Degree Standard Observer']
        
    # 2. 从本地读取用户提供的新观察者数据
    cmf_170_2deg = load_custom_cmf("CMF.xlsx - CIE170-2 2°.csv", "CIE170-2 2°")
    if cmf_170_2deg is not None:
        CMFS_DICT["CIE170-2 2°"] = cmf_170_2deg
        
    cmf_170_10deg = load_custom_cmf("CMF.xlsx - CIE170-2 10°.csv", "CIE170-2 10°")
    if cmf_170_10deg is not None:
        CMFS_DICT["CIE170-2 10°"] = cmf_170_10deg

    cmf_judd_vos = load_custom_cmf("CMF.xlsx - Judd&Vos1978 2°.csv", "Judd&Vos1978 2°")
    if cmf_judd_vos is not None:
        CMFS_DICT["Judd&Vos1978 2°"] = cmf_judd_vos

    print("\n正在解析，请稍候...\n")
    # 高精度用 1931 用于寻找解，低精度 1931 用于混合 White
    cmf_1931_fine = get_cmf_data(CMFS_DICT["CIE1931 2°"], fine_wavelengths)
    cmf_1931_base = get_cmf_data(CMFS_DICT["CIE1931 2°"], base_wavelengths)

    # 求解最优光谱并强制离散投射
    spectra = {}
    for ch in ["R", "G", "B"]:
        l1, l2, b_type = find_true_bounds_high_res(space_info[ch], cmf_1931_fine, fine_wavelengths)
        
        spec = np.zeros(len(base_wavelengths))
        l1_idx = int(np.round(l1) - 390)
        l2_idx = int(np.round(l2) - 390)
        
        if b_type == 1:
            spec[l1_idx : l2_idx + 1] = 1.0
        else:
            spec[:] = 1.0
            spec[l1_idx : l2_idx + 1] = 0.0
            
        spectra[ch] = spec
        
    # 合成 White metamer
    spectra["W"] = calculate_white_spectrum(spectra["R"], spectra["G"], spectra["B"], space_info["W"], cmf_1931_base)

    # 导出 CSV
    df_export = pd.DataFrame({"Wavelength": base_wavelengths})
    for ch in ["W", "R", "G", "B"]:
        df_export[ch] = spectra[ch]
    
    csv_filename = f"Optimal_Spectra_WRGB_{space_name.replace('/','_')}.csv"
    df_export.to_csv(csv_filename, index=False)
    print(f"✅ APPLE Optimal Spectral光谱数据已导出至: {csv_filename}")

    # 计算不同观察者下的xy坐标
    results_text = []
    for cmf_name, cmf_obj in CMFS_DICT.items():
        cmf_vals = get_cmf_data(cmf_obj, base_wavelengths)
        results_text.append(f"\n--- {cmf_name} ---")
        for ch in ["W", "R", "G", "B"]:
            x, y = calculate_xy(spectra[ch], cmf_vals)
            results_text.append(f"{ch}: x={x:.4f}, y={y:.4f}")

    results_str = "\n".join(results_text)
    print("\n各观察者函数计算结果汇总：")
    print(results_str)

    # ================= 绘图重构 =================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 2, width_ratios=[2.5, 1], hspace=0.3)
    
    axes = {
        "W": fig.add_subplot(gs[0, 0]),
        "R": fig.add_subplot(gs[1, 0]),
        "G": fig.add_subplot(gs[2, 0]),
        "B": fig.add_subplot(gs[3, 0])
    }
    
    for ch in ["R", "G", "B"]:
        axes[ch].sharex(axes["W"])

    colors = {"W": "black", "R": "red", "G": "green", "B": "blue"}
    titles = {"W": "White Metamer", "R": "Red Primary", "G": "Green Primary", "B": "Blue Primary"}
    
    for ch in ["W", "R", "G", "B"]:
        ax = axes[ch]
        ax.fill_between(base_wavelengths, 0, spectra[ch], color=colors[ch], alpha=0.1, step='mid')
        ax.plot(base_wavelengths, spectra[ch], color=colors[ch], label=ch, drawstyle='steps-mid', linewidth=2)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("Relative")
        ax.set_title(titles[ch], loc='right', fontsize=10, pad=-15, color=colors[ch], fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if ch != "B":
            plt.setp(ax.get_xticklabels(), visible=False)
            
    axes["B"].set_xlim(390, 760)
    axes["B"].set_xlabel("Wavelength (nm)", fontsize=12)
    
    fig.suptitle(f"APPLE Optimal Spectral Display for {space_name}\n", fontsize=16, fontweight='bold', y=0.96)

    ax_text = fig.add_subplot(gs[:, 1])
    ax_text.axis('off')
    ax_text.text(0.05, 0.95, "Calculated Chromaticities\n(1nm Discretization)", fontsize=14, fontweight='bold', va='top')
    ax_text.text(0.05, 0.88, results_str, fontsize=11, va='top', family='monospace')

    plt.tight_layout()
    img_filename = f"Optimal_Spectra_Chart_{space_name.replace('/','_')}.png"
    plt.savefig(img_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {img_filename}")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序运行时发生意外错误: {e}")
    finally:
        print("\n" + "="*50)
        input("运行结束。按回车键(Enter)退出窗口...")
