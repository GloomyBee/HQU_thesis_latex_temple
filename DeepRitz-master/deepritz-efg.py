import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
from datetime import datetime

plt.rc('text', usetex=False)  # 使用 Mathtext 渲染
plt.rc('mathtext', fontset='cm')  # 使用 Computer Modern 字体
plt.rc('font', family='Arial', size=16)  # 全局字体为 Arial，字号 14
plt.rc('axes', titlesize=16, labelsize=16)  # 标题 16，轴标签 14
plt.rc('legend', fontsize=16)  # 图例 12
plt.rc('xtick', labelsize=16)  # x 轴刻度 12
plt.rc('ytick', labelsize=16)  # y 轴刻度 12

# EFG 模块
def cubic_spline(r, device):
    r = torch.abs(r)
    condition1 = (r <= 0.5)
    condition2 = (r > 0.5) & (r <= 1.0)
    condition3 = (r > 1.0)
    result = torch.zeros_like(r, device=device)
    result[condition1] = 2 / 3 - 4 * r[condition1] ** 2 + 4 * r[condition1] ** 3
    result[condition2] = 4 / 3 - 4 * r[condition2] + 4 * r[condition2] ** 2 - 4 / 3 * r[condition2] ** 3
    result[condition3] = 0.0
    return result

def shape_function(x, nodes, s, device):
    batch_size = x.shape[0]  # batchsize采样点数量，x采样点张量
    num_nodes = len(nodes)  # 节点张量

    x = x.view(-1, 1)  # 改变张量的形状，但不改变数据本身。-1 表示自动推断该维度的大小
    nodes = nodes.view(1, -1)  # 将一维张量变为二维，第一维为 1，第二维自动推断
    r = (x - nodes) / s

    phi = cubic_spline(r, device)  # 核函数，输入r（batch_size,num_nodes),输出phi

    p = torch.ones((batch_size, num_nodes, 2), device=device)  # 基函数张量，表示节点中每个采样点的基函数值
    p[:, :, 1] = (nodes - x) / s

    phi_expanded = phi.unsqueeze(-1)
    A = torch.einsum('bni,bnj->bij', p * phi_expanded, p)
    A += 1e-10 * torch.eye(2, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    B = p * phi_expanded
    inv_A = torch.linalg.inv(A)

    p0 = torch.tensor([1.0, 0.0], device=device).view(1, 2, 1)
    Psi = (p0.transpose(1, 2) @ (inv_A @ B.transpose(1, 2))).squeeze(1)  # (batch_size, num_nodes)
    return Psi

# Network structure
class EnhancedRitzNet(torch.nn.Module):
    def __init__(self, params, nodes, s):
        super(EnhancedRitzNet, self).__init__()
        self.params = params
        self.nodes = nodes.clone().detach().to(self.params.get("device", "cpu"))
        self.s = s.clone().detach().to(self.params.get("device", "cpu"))
        self.linearIn = nn.Linear(self.params["d"] + len(nodes), self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))
        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def compute_efg_shape(self, x):
        Psi = shape_function(x, self.nodes, self.s, x.device)
        return Psi  # Shape: (batch_size, num_nodes)

    def forward(self, x):
        efg_features = self.compute_efg_shape(x)
        x_enhanced = torch.cat((x, efg_features), dim=1)  # Combine original input with EFG features
        x = torch.tanh(self.linearIn(x_enhanced))  # Using tanh; consider ReLU if gradient vanishing occurs
        for layer in self.linear:
            x = torch.tanh(layer(x))
        return self.linearOut(x)

# 一维区间内部采样
def sampleFromInterval(start, end, num_points):
    return np.linspace(start, end, num_points).reshape(-1, 1)

# 边界点采样
def sampleFromBoundary(start, end):
    return np.array([[start], [end]])

# 源项 f(x) = -1
def ffun(data):
    return -torch.ones([data.shape[0], 1], dtype=torch.float)

# 解析解 u(x) = 0.5 * x^2 + 0.5 * x
def exact(x):
    return 0.5 * x ** 2 + 0.5 * x

# 误差计算
def errorFun(output, target):
    error = output - target
    error = math.sqrt(torch.mean(error ** 2))
    ref = math.sqrt(torch.mean(target ** 2))
    return error / ref

# 测试函数
def test(model, device, params):
    numQuad = params["numQuad"]
    data = torch.from_numpy(sampleFromInterval(0, 1, numQuad)).float().to(device)
    data.requires_grad = True  # 启用梯度计算以获取导数

    # 预测解
    output = model(data)
    target = exact(data).to(device)

    # 计算 L2 误差
    l2_error = errorFun(output, target)

    # 计算导数
    grad_output = torch.autograd.grad(output, data, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    grad_target = torch.autograd.grad(target, data, grad_outputs=torch.ones_like(target), create_graph=True)[0]

    # 计算 H1 误差
    l2_grad_error = torch.sqrt(torch.mean((grad_output - grad_target) ** 2))
    h1_error = torch.sqrt(l2_error ** 2 + l2_grad_error ** 2)

    return l2_error, h1_error  # 直接返回浮点数，无需 .item()

# 训练函数
def train(model, device, params, optimizer, scheduler):
    model.train()

    # 初始化采样点
    data_body = torch.from_numpy(sampleFromInterval(0, 1, params["bodyBatch"])).float().to(device)
    data_body.requires_grad = True
    data_boundary = torch.from_numpy(sampleFromBoundary(0, 1)).float().to(device)

    # 记录误差和步数
    steps = []
    l2_errors = []
    h1_errors = []

    # 创建文件保存损失值
    with open("loss_history.txt", "w") as f:
        f.write("Step Loss\n")  # 写入表头

        for step in range(params["trainStep"]):
            # 内部点输出
            output_body = model(data_body)
            dfdx = torch.autograd.grad(output_body, data_body, grad_outputs=torch.ones_like(output_body),
                                       retain_graph=True, create_graph=True, only_inputs=True)[0]

            # 体积分损失（未乘以区间长度，优化方向不受影响）
            fTerm = ffun(data_body).to(device)
            loss_body = torch.mean(0.5 * dfdx ** 2 - fTerm * output_body)

            # 边界条件损失
            output_boundary = model(data_boundary)
            loss_boundary = torch.mean((output_boundary[0] - 0) ** 2 + (output_boundary[1] - 1) ** 2) * params["penalty"]

            # 总损失
            loss = loss_body + loss_boundary

            # 每 writeStep 步记录一次误差
            if step % params["writeStep"] == 0:
                model.eval()  # 切换到评估模式
                l2_error, h1_error = test(model, device, params)  # 计算 L2 和 H1 误差
                model.train()  # 切换回训练模式
                steps.append(step)
                l2_errors.append(l2_error)  # 直接添加浮点数
                h1_errors.append(h1_error.item())  # 将 h1_error 从张量转换为浮点数
                print(f"Step {step}: Loss = {loss.item()}, L2 Error = {l2_error:.6f}, H1 Error = {h1_error:.6f}")
                f.write(f"{step} {loss.item()}\n")  # 保存到文件

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # 保存误差数据
    with open("error_history.txt", "w") as f:
        f.write("Step L2_Error H1_Error\n")
        for step, l2_err, h1_err in zip(steps, l2_errors, h1_errors):
            f.write(f"{step} {l2_err} {h1_err}\n")

    return steps, l2_errors, h1_errors  # 返回误差数据用于绘图

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 保存图像的函数
def save_plot(plt_obj, filename, params):
    project_path = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(project_path, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    # 使用 params["k"] 控制第二位，提取原文件名中的第三位
    suffix = filename.split('-')[-1]  # 获取第三位数字（1, 2, 3, 4, 5）
    full_filename = os.path.join(figures_dir, f"{filename.split('-')[0]}-{params['k']}-{suffix}.png")
    plt_obj.savefig(full_filename, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {full_filename}")

# 可视化预测解与解析解
def pltResult(model, device, nSample, params,  train_time, test_time ):
    x = np.linspace(0, 1, nSample).reshape(-1, 1)
    data = torch.from_numpy(x).float().to(device)
    data.requires_grad = True

    # 预测解
    pred = model(data).detach().cpu().numpy()
    exact_sol = exact(x)

    # 打印边界点预测值以验证边界条件
    boundary_points = torch.tensor([[0.0], [1.0]], device=device).float()
    boundary_pred = model(boundary_points).detach().cpu().numpy()
    print(f"Boundary predictions: u(0) = {boundary_pred[0][0]:.6f}, u(1) = {boundary_pred[1][0]:.6f}")

    # 保存数据到文件
    with open("solution_data.txt", "w") as f:
        f.write("x Predicted Exact\n")
        for i in range(len(x)):
            f.write(f"{x[i][0]} {pred[i][0]} {exact_sol[i][0]}\n")

    # 计算导数
    grad_pred = torch.autograd.grad(model(data), data, grad_outputs=torch.ones_like(model(data)), create_graph=True)[0].detach().cpu().numpy()
    # 使用一维 x 计算解析解的导数
    grad_exact = np.gradient(exact_sol[:, 0], x[:, 0])  # 转换为一维数组

    # 绘图
    plt.figure(figsize=(10, 6))

    # 子图 1：解的比较
    plt.plot(x, pred, label="Predicted Solution")
    plt.plot(x, exact_sol, label="Exact Solution", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title(
        f"Deep Ritz Solution vs Exact Solution\n(width={params['width']}, depth={params['depth']}, penalty={params['penalty']},trainStep={params['trainStep']})\n"
        f"Train Time: {train_time:.2f}s, Test Time: {test_time:.2f}s")
    plt.grid(True)

    # 保存图像
    save_plot(plt, "5-1-1", params)
    plt.tight_layout()
    plt.show()

# 绘制损失曲线（前 50 步和之后）
def plot_loss_curve(params):
    steps, losses = [], []
    with open("loss_history.txt", "r") as f:
        lines = f.readlines()[1:]  # 跳过表头
        for line in lines:
            step, loss = map(float, line.strip().split())
            steps.append(step)
            losses.append(loss)

    # 筛选前 50 步的数据
    initial_steps = [s for s in steps if s <= 100]
    initial_losses = [losses[i] for i in range(len(steps)) if steps[i] <= 100]

    # 筛选第 50 步之后的数据
    converge_steps = [s for s in steps if s > 100]
    converge_losses = [losses[i] for i in range(len(steps)) if steps[i] > 100]

    # 绘制前 50 步的损失下降趋势
    plt.figure(figsize=(10, 6))
    plt.plot(initial_steps, initial_losses, label="Initial Loss Decline", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Initial Loss Decline (First 100 Steps)")
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    # 保存图像
    save_plot(plt, "5-1-3", params)
    plt.show()

    # 绘制之后损失的下降趋势
    plt.figure(figsize=(10, 6))
    plt.plot(converge_steps, converge_losses, label="Convergence Loss", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Convergence Loss (After Step 100)")
    plt.legend()
    plt.grid(True)

    # 保存图像
    save_plot(plt, "5-1-4", params)
    plt.show()

# 绘制 L2 误差
def plot_absolute_error_distribution(params):
    x, pred, exact = [], [], []
    with open("solution_data.txt", "r") as f:
        lines = f.readlines()[1:]  # 跳过表头
        for line in lines:
            xi, predi, exacti = map(float, line.strip().split())
            x.append(xi)
            pred.append(predi)
            exact.append(exacti)

    l2_errors = [abs(pred[i] - exact[i]) for i in range(len(x))]

    # 计算 L2 误差并打印
    l2_error = np.sqrt(np.mean(np.array(l2_errors) ** 2))
    print(f"L2 Error: {l2_error}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x, l2_errors, label="Absolute Error")
    #plt.ylim(0, 0.007)
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error Distribution\n(L2 Error: {l2_error:.6f})")
    plt.legend()
    plt.grid(True)

    # 保存图像
    save_plot(plt, "5-1-2", params)
    plt.show()

# 绘制 H1 误差
def plot_h1_error_distribution(params, h1_error):
    x, pred, exact = [], [], []
    with open("solution_data.txt", "r") as f:
        lines = f.readlines()[1:]  # 跳过表头
        for line in lines:
            xi, predi, exacti = map(float, line.strip().split())
            x.append(xi)
            pred.append(predi)
            exact.append(exacti)

    # 计算数值导数
    grad_pred = np.gradient(pred, x[1] - x[0])  # 使用相邻点的差值作为步长
    grad_exact = np.gradient(exact, x[1] - x[0])  # 使用相邻点的差值作为步长

    # 计算 H1 误差组成部分
    absolute_error = [abs(pred[i] - exact[i]) for i in range(len(x))]
    grad_errors = [abs(grad_pred[i] - grad_exact[i]) for i in range(len(grad_pred))]

    # 计算全局 L2 误差和导数 L2 误差
    l2_error = np.sqrt(np.mean(np.array(absolute_error) ** 2))
    grad_l2_error = np.sqrt(np.mean(np.array(grad_errors) ** 2))
    computed_h1_error = np.sqrt(l2_error ** 2 + grad_l2_error ** 2)
    print(f"Computed L2 Error: {l2_error:.6f}, Gradient L2 Error: {grad_l2_error:.6f}, Computed H1 Error: {computed_h1_error:.6f}")

    # 绘图
    plt.figure(figsize=(10, 6))

    plt.plot(x, absolute_error, label="$Absolute\ Error\ |u_{pred} - u_{exact}|$", color="blue")
    plt.plot(x, grad_errors, label="$Gradient\ Error\ |du_{pred}/dx - du_{exact}/dx|$", color="red", linestyle="--")

    plt.xlabel("x")
    plt.ylabel("Error")
    plt.title(f"H1 Error Components Distribution\n(H1 Error: {h1_error:.6f})")
    plt.legend()
    plt.grid(True)

    # 保存图像
    save_plot(plt, "5-1-5", params)  # 保存为 5-1-5
    plt.show()

# 绘制误差-步数曲线（新增）
def plot_error_convergence(steps, l2_errors, h1_errors, params):
    # 普通误差-步数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(steps, l2_errors, label="L2 Error", color="blue")
    plt.plot(steps, h1_errors, label="H1 Error", color="red", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Error")
    plt.title("Error Convergence with Training Steps")
    plt.legend()
    plt.grid(True)
    save_plot(plt, "5-1-6", params)  # 保存为 5-1-6
    plt.show()

    # 对数尺度误差-步数曲线
    plt.figure(figsize=(10, 6))
    plt.semilogy(steps, l2_errors, label="Log L2 Error", color="blue")
    plt.semilogy(steps, h1_errors, label="Log H1 Error", color="red", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Log Error")
    plt.title("Log-Error Convergence with Training Steps")
    plt.legend()
    plt.grid(True)
    save_plot(plt, "5-1-7", params)  # 保存为 5-1-7
    plt.show()

# 主函数
def main():
    # Parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 强制初始化CUDA上下文
    if torch.cuda.is_available():
        dummy_tensor = torch.zeros(1).cuda()
        dummy_tensor += 1
    params = dict()
    params["k"] = 2  # 图片名第二位，通过修改此值控制文件名（例如改为 2 生成 5-2-1 等）
    params["d"] = 1  # 1D
    params["dd"] = 1  # Scalar field
    params["bodyBatch"] = 100  # Batch size
    params["bdryBatch"] = 2  # Batch size for the boundary integral
    params["lr"] = 0.001  # Learning rate
    params["preLr"] = 0.01  # Learning rate (Pre-training)
    params["width"] = 50  # Width of layers (sufficient for 1D, adjust for complex problems)
    params["depth"] = 1  # Depth of the network: depth+2
    params["numQuad"] = 100  # Number of quadrature points for testing
    params["trainStep"] = 10000
    params["penalty"] = 1000  # Increased to ensure boundary conditions
    params["preStep"] = 0
    params["diff"] = 0.01
    params["writeStep"] = 10
    params["sampleStep"] = 10
    params["step_size"] = 300  # Adjusted to allow learning rate decay within 1000 steps
    params["gamma"] = 0.3
    params["decay"] = 0.00001
    params["device"] = device

    # EFG 参数
    NP = 11  # Number of nodes
    nodes = torch.linspace(0.0, 1.0, NP).to(device)  # Node spacing 0.1
    s = torch.tensor(0.2, dtype=torch.float32).to(device)  # Support domain covers 2 nodes on each side

    # 验证 EFG 参数
    test_x = torch.tensor([[0.05]], device=device)
    psi = shape_function(test_x, nodes, s, device)
    print(f"EFG shape function at x=0.05: {psi.cpu().numpy()}")

    # 初始化模型
    #startTime = time.time()
    model = EnhancedRitzNet(params, nodes, s).to(device)
    #init_time = time.time() - startTime
    #print("Generating network costs %s seconds." % init_time)

    preOptimizer = torch.optim.Adam(model.parameters(), lr=params["preLr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["decay"])
    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])

    # 训练模型
    startTime = time.time()
    steps, l2_errors, h1_errors = train(model, device, params, optimizer, scheduler)  # 获取误差数据
    train_time = time.time() - startTime
    print("Training costs %s seconds." % train_time)


    # 测试模型
    model.eval()
    start_test_time = time.time()
    l2_error, h1_error = test(model, device, params)
    test_time = time.time() - start_test_time
    print(f"Testing costs {test_time:.4f} seconds.")
    print(f"The L2 error (of the last model) is {l2_error}.")
    print(f"The H1 error (of the last model) is {h1_error}.")
    print(f"The number of parameters is {count_parameters(model)}.")

    torch.save(model.state_dict(), "last_model.pt")

    # 保存训练时间到文件
    #total_time = init_time + train_time
    with open("training_time.txt", "w") as f:
        #f.write(f"Initialization Time: {init_time} seconds\n")
        f.write(f"Training Time: {train_time} seconds\n")
        f.write(f"Test Time: {test_time} seconds\n")


    # 可视化结果
    pltResult(model, device, 100, params,  train_time, test_time)
    plot_loss_curve(params)
    plot_absolute_error_distribution(params)  # 单独调用 L2 误差
    plot_h1_error_distribution(params, h1_error)  # 单独调用 H1 误差，传递 h1_error
    plot_error_convergence(steps, l2_errors, h1_errors, params)  # 绘制误差-步数曲线

if __name__ == "__main__":
    main()