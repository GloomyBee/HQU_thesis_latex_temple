import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# Set default tensor type to float64 for consistency with Julia's Float64
torch.set_default_dtype(torch.float64)

plt.rc('text', usetex=False)  # 使用 Mathtext 渲染
plt.rc('mathtext', fontset='cm')  # 使用 Computer Modern 字体
plt.rc('font', family='Arial', size=16)  # 全局字体为 Arial，字号 14
plt.rc('axes', titlesize=16, labelsize=16)  # 标题 16，轴标签 14
plt.rc('legend', fontsize=16)  # 图例 12
plt.rc('xtick', labelsize=16)  # x 轴刻度 12
plt.rc('ytick', labelsize=16)  # y 轴刻度 12

# 1. Define nodes and parameters
NP = 11  # Number of nodes
nodes = torch.linspace(0.0, 1.0, NP)  # Uniformly distributed nodes
s = 0.2  # Support domain size
n_gauss = 3  # Number of Gauss points
gauss_points = torch.tensor([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])  # Gauss quadrature points
gauss_weights = torch.tensor([5 / 9, 8 / 9, 5 / 9])  # Gauss quadrature weights

# 2. Cubic spline kernel function and its derivative
def cubic_spline(r):
    r = torch.abs(r)
    condition1 = (r <= 0.5)
    condition2 = (r > 0.5) & (r <= 1.0)
    condition3 = (r > 1.0)
    result = torch.zeros_like(r)
    result[condition1] = 2 / 3 - 4 * r[condition1] ** 2 + 4 * r[condition1] ** 3
    result[condition2] = 4 / 3 - 4 * r[condition2] + 4 * r[condition2] ** 2 - 4 / 3 * r[condition2] ** 3
    result[condition3] = 0.0
    return result

def cubic_spline_deriv(r):
    r_abs = torch.abs(r)
    sign_r = torch.sign(r)
    condition1 = (r_abs <= 0.5)
    condition2 = (r_abs > 0.5) & (r_abs <= 1.0)
    condition3 = (r_abs > 1.0)
    result = torch.zeros_like(r)
    result[condition1] = -8 * r_abs[condition1] + 12 * r_abs[condition1] ** 2
    result[condition2] = -4 + 8 * r_abs[condition2] - 4 * r_abs[condition2] ** 2
    result[condition3] = 0.0
    return result * sign_r

# 3. MLS shape function and its derivative
def shape_function(x, nodes, s):
    n = len(nodes)
    A = torch.zeros((2, 2))
    A_deriv = torch.zeros((2, 2))
    B = torch.zeros((2, n))
    phi = torch.zeros(n)
    phi_deriv = torch.zeros(n)
    Ψ = torch.zeros(n)
    Ψ_deriv = torch.zeros(n)

    # Compute A, A_deriv, and B
    for i in range(n):
        r = (x - nodes[i]) / s
        phi[i] = cubic_spline(r)
        phi_deriv[i] = cubic_spline_deriv(r) / s
        p = torch.tensor([1.0, (nodes[i] - x) / s])
        B[:, i] = phi[i] * p
        A += phi[i] * torch.outer(p, p)
        A_deriv += phi_deriv[i] * torch.outer(p, p)

    # Prevent singularity in A
    if torch.abs(torch.det(A)) < 1e-10:
        A += 1e-10 * torch.eye(2)

    inv_A = torch.inverse(A)

    # Compute shape functions Ψ and their derivatives Ψ_deriv
    for i in range(n):
        p = torch.tensor([1.0, (nodes[i] - x) / s])
        p_deriv = torch.tensor([0.0, -1.0 / s])
        B_deriv = phi_deriv[i] * p
        B_i = B[:, i]
        Ψ[i] = torch.tensor([1.0, 0.0]) @ inv_A @ B_i
        Ψ_deriv[i] = (p_deriv @ inv_A @ B_i) + \
                     (torch.tensor([1.0, 0.0]) @ inv_A @ (B_deriv - A_deriv @ inv_A @ B_i))

    return Ψ, Ψ_deriv

# 4. Assemble stiffness matrix and load vector
def assemble_system(nodes, s):
    NP = len(nodes)
    K = torch.zeros((NP, NP))  # Stiffness matrix
    f = torch.zeros(NP)  # Load vector

    elements = [(nodes[i], nodes[i + 1]) for i in range(NP - 1)]

    for elem in elements:
        x1, x2 = elem
        h = x2 - x1
        for i in range(n_gauss):
            xi = gauss_points[i]
            w = gauss_weights[i]
            x = (x2 + x1) / 2 + (x2 - x1) / 2 * xi
            Ψ, Ψ_deriv = shape_function(x, nodes, s)

            for I in range(NP):
                for J in range(NP):
                    K[I, J] += Ψ_deriv[I] * Ψ_deriv[J] * w * h / 2
                f[I] += -Ψ[I] * w * h / 2

    return K, f

# 5. Apply Dirichlet boundary conditions using Lagrange multipliers
def apply_boundary_conditions(K, f, nodes, s):
    NP = len(nodes)
    K_aug = torch.zeros((NP + 2, NP + 2))  # Augmented matrix for 2 Dirichlet conditions
    f_aug = torch.zeros(NP + 2)
    K_aug[:NP, :NP] = K
    f_aug[:NP] = f

    # u(0) = 0
    Ψ_0, _ = shape_function(torch.tensor(0.0), nodes, s)
    for I in range(NP):
        K_aug[I, NP] = Ψ_0[I]
        K_aug[NP, I] = Ψ_0[I]
    f_aug[NP] = 0.0

    # u(1) = 1
    Ψ_1, _ = shape_function(torch.tensor(1.0), nodes, s)
    for I in range(NP):
        K_aug[I, NP + 1] = Ψ_1[I]
        K_aug[NP + 1, I] = Ψ_1[I]
    f_aug[NP + 1] = 1.0

    return K_aug, f_aug

# 6. Solve the Poisson equation
def solve_poisson():
    start_time = time.time()  # 记录开始时间

    K, f = assemble_system(nodes, s)
    K_aug, f_aug = apply_boundary_conditions(K, f, nodes, s)

    # Solve the linear system
    sol = torch.linalg.solve(K_aug, f_aug)
    d = sol[:NP]  # Displacement solution
    λ = sol[NP:NP + 2]  # Lagrange multipliers

    end_time = time.time()  # 记录结束时间
    computation_time = end_time - start_time  # 计算时间（秒）

    # Print nodal solutions
    print("节点位移解：")
    for i in range(NP):
        print(f"x = {nodes[i].item():.3f}, u = {d[i].item():.6f}")
    print(f"拉格朗日乘子 λ = {λ.tolist()}")

    # Compute predicted and exact solutions on a fine grid
    x_fine = torch.linspace(0, 1, 101)  # Use finer grid for error computation
    u_pred_fine = []
    u_exact_fine = []

    for x in x_fine:
        Ψ, Ψ_deriv = shape_function(x, nodes, s)
        u_comp = (Ψ @ d).item()
        u_exact_val = 0.5 * x * (1 + x)
        u_pred_fine.append(u_comp)
        u_exact_fine.append(u_exact_val)

    # Compute numerical gradients using central difference
    grad_pred_fine = []
    grad_exact_fine = []

    for idx in range(len(x_fine)):
        if idx == 0:  # Forward difference for the first point
            grad_pred = (u_pred_fine[1] - u_pred_fine[0]) / (x_fine[1] - x_fine[0])
            grad_exact = (u_exact_fine[1] - u_exact_fine[0]) / (x_fine[1] - x_fine[0])
        elif idx == len(x_fine) - 1:  # Backward difference for the last point
            grad_pred = (u_pred_fine[-1] - u_pred_fine[-2]) / (x_fine[-1] - x_fine[-2])
            grad_exact = (u_exact_fine[-1] - u_exact_fine[-2]) / (x_fine[-1] - x_fine[-2])
        else:  # Central difference for interior points
            grad_pred = (u_pred_fine[idx + 1] - u_pred_fine[idx - 1]) / (x_fine[idx + 1] - x_fine[idx - 1])
            grad_exact = (u_exact_fine[idx + 1] - u_exact_fine[idx - 1]) / (x_fine[idx + 1] - x_fine[idx - 1])
        grad_pred_fine.append(grad_pred)
        grad_exact_fine.append(grad_exact)

    # Compute L2 error
    l2_error = torch.sqrt(torch.mean(torch.tensor([(u_pred_fine[i] - u_exact_fine[i]) ** 2 for i in range(len(x_fine))])))
    # Compute gradient L2 error
    grad_l2_error = torch.sqrt(torch.mean(torch.tensor([(grad_pred_fine[i] - grad_exact_fine[i]) ** 2 for i in range(len(x_fine))])))
    # Compute H1 error
    h1_error = torch.sqrt(l2_error ** 2 + grad_l2_error ** 2)
    print(f"\nL2 Error: {l2_error.item():.6f}")
    print(f"Gradient L2 Error: {grad_l2_error.item():.6f}")
    print(f"H1 Error: {h1_error.item():.6f}")

    # Print exact solution comparison
    print("\n解析解对比：")
    for i in range(NP):
        x = nodes[i].item()
        u_exact = 0.5 * x * (1 + x)
        print(f"x = {x:.3f}, u_exact = {u_exact:.6f}, u_computed = {d[i].item():.6f}, "
              f"误差 = {abs(u_exact - d[i].item()):.6f}")

    return d, computation_time, h1_error  # 返回解、计算时间和 H1 误差

# 保存图像的函数
def save_plot(plt_obj, filename):
    # 获取项目路径（当前脚本所在目录）
    project_path = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(project_path, "figures")

    # 创建 figures 目录（如果不存在）
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # 添加时间戳以确保文件名唯一
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = os.path.join(figures_dir, f"{filename}.png")

    # 保存图像
    plt_obj.savefig(full_filename, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {full_filename}")

# 7. Plot EFG vs Exact Solution
def plot_solution(nodes, d, computation_time, h1_error):
    x_fine = torch.linspace(0, 1, 101)
    u_fine = []
    u_exact_fine = []

    for x in x_fine:
        Ψ, _ = shape_function(x, nodes, s)
        u_comp = (Ψ @ d).item()
        u_exact_val = 0.5 * x * (1 + x)
        u_fine.append(u_comp)
        u_exact_fine.append(u_exact_val)

    plt.figure(figsize=(10, 6))
    plt.plot(x_fine.numpy(), u_fine, label="EFG")
    plt.plot(x_fine.numpy(), u_exact_fine, label="Exact", linestyle="--")
    plt.scatter(nodes.numpy(), d.numpy(), color="red", label="Computed Nodes")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    # 修改标题，去除 H1 误差字段，仅保留计算时间
    plt.title(f"EFG Solution vs Exact Solution\nComputation Time: {computation_time:.4f} seconds")
    plt.legend()
    plt.grid(True)

    # Save image
    save_plot(plt, "4-1")
    plt.show()

# 8. Plot Relative Error
def plot_relative_error(nodes, d, h1_error):
    x_fine = torch.linspace(0, 1, 101)
    error_rel = []
    u_pred_fine = []
    u_exact_fine = []

    # Compute predicted and exact solutions for L2 error
    for x in x_fine:
        Ψ, _ = shape_function(x, nodes, s)
        u_comp = (Ψ @ d).item()
        u_exact_val = 0.5 * x * (1 + x)
        rel_error = abs(u_exact_val - u_comp) / abs(u_exact_val) if u_exact_val != 0 else 0.0
        error_rel.append(rel_error)
        u_pred_fine.append(u_comp)
        u_exact_fine.append(u_exact_val)

    # Compute L2 error for the title
    l2_error = torch.sqrt(torch.mean(torch.tensor([(u_pred_fine[i] - u_exact_fine[i]) ** 2 for i in range(len(x_fine))])))

    plt.figure(figsize=(10, 6))
    plt.plot(x_fine.numpy(), error_rel, label="Absolute Error", color="blue")
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    # 修改标题，增加 L2 误差字段，保留 H1 误差
    plt.title(f"Absolute Error\n(L2 Error: {l2_error.item():.6f})") #, H1 Error: {h1_error.item():.6f}
    plt.legend()
    plt.grid(True)

    # Save image
    save_plot(plt, "4-2")
    plt.show()

# 9. Plot H1 Error Distribution
def plot_h1_error_distribution(nodes, d, h1_error):
    x_fine = torch.linspace(0, 1, 101)
    u_pred_fine = []
    u_exact_fine = []

    # First compute all u_pred and u_exact values
    for x in x_fine:
        Ψ, Ψ_deriv = shape_function(x, nodes, s)
        u_comp = (Ψ @ d).item()
        u_exact_val = 0.5 * x * (1 + x)
        u_pred_fine.append(u_comp)
        u_exact_fine.append(u_exact_val)

    # Compute numerical gradients
    grad_pred_fine = []
    grad_exact_fine = []

    for idx in range(len(x_fine)):
        if idx == 0:  # Forward difference for the first point
            grad_pred = (u_pred_fine[1] - u_pred_fine[0]) / (x_fine[1] - x_fine[0])
            grad_exact = (u_exact_fine[1] - u_exact_fine[0]) / (x_fine[1] - x_fine[0])
        elif idx == len(x_fine) - 1:  # Backward difference for the last point
            grad_pred = (u_pred_fine[-1] - u_pred_fine[-2]) / (x_fine[-1] - x_fine[-2])
            grad_exact = (u_exact_fine[-1] - u_exact_fine[-2]) / (x_fine[-1] - x_fine[-2])
        else:  # Central difference for interior points
            grad_pred = (u_pred_fine[idx + 1] - u_pred_fine[idx - 1]) / (x_fine[idx + 1] - x_fine[idx - 1])
            grad_exact = (u_exact_fine[idx + 1] - u_exact_fine[idx - 1]) / (x_fine[idx + 1] - x_fine[idx - 1])
        grad_pred_fine.append(grad_pred)
        grad_exact_fine.append(grad_exact)

    # Compute error components
    absolute_error = [abs(u_pred_fine[i] - u_exact_fine[i]) for i in range(len(x_fine))]
    grad_errors = [abs(grad_pred_fine[i] - grad_exact_fine[i]) for i in range(len(x_fine))]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_fine.numpy(), absolute_error, label="$Absolute\ Error\ |u_{pred} - u_{exact}|$", color="blue")
    plt.plot(x_fine.numpy(), grad_errors, label="$Gradient\ Error\ |du_{pred}/dx - du_{exact}/dx|$", color="red", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("Error")
    # 标题保持不变
    plt.title(f"Point-wise Error Components Distribution\n(Overall H1 Error: {h1_error.item():.6f})")
    plt.legend()
    plt.grid(True)

    # Save image
    save_plot(plt, "4-3")
    plt.show()

# Run the solver and generate plots
d, comp_time, h1_error = solve_poisson()  # 获取解、计算时间和 H1 误差
plot_solution(nodes, d, comp_time, h1_error)  # 传递 H1 误差
plot_relative_error(nodes, d, h1_error)  # 传递 H1 误差
plot_h1_error_distribution(nodes, d, h1_error)  # 添加 H1 误差绘制