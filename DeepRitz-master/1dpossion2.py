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
plt.rc('font', family='Arial', size=16)  # 全局字体
plt.rc('axes', titlesize=16, labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

# Network structure
class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):
        x = torch.tanh(self.linearIn(x))  # Match dimension
        for layer in self.linear:
            x_temp = torch.tanh(layer(x))
            x = x_temp

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
    data = torch.from_numpy(sampleFromInterval(0, 1, params["numQuad"])).float().to(device)
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

    return l2_error, h1_error  # 返回 L2 和 H1 误差

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

            # 体积分损失
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
                # 直接添加浮点数（l2_error 和 h1_error 已经是 float 类型）
                l2_errors.append(l2_error)
                h1_errors.append(h1_error.item())  # h1_error 可能是张量，使用 .item() 转换为浮点数
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
    suffix = filename.split('-')[-1]  # 获取第三位数字（1, 2, 3, 4）
    full_filename = os.path.join(figures_dir, f"{filename.split('-')[0]}-{params['k']}-{suffix}.png")
    plt_obj.savefig(full_filename, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {full_filename}")

# 可视化预测解与解析解
def pltResult(model, device, nSample, params, train_time, test_time):
    x = np.linspace(0, 1, nSample).reshape(-1, 1)
    pred = model(torch.from_numpy(x).float().to(device)).detach().cpu().numpy()
    exact_sol = exact(x)

    # 保存数据到文件
    with open("solution_data.txt", "w") as f:
        f.write("x Predicted Exact\n")
        for i in range(len(x)):
            f.write(f"{x[i][0]} {pred[i][0]} {exact_sol[i][0]}\n")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x, pred, label="Predicted Solution")
    plt.plot(x, exact_sol, label="Exact Solution", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title(
        f"Deep Ritz Solution vs Exact Solution\n(width={params['width']}, depth={params['depth']}, penalty={params['penalty']}, trainStep={params['trainStep']})\n"
        f"Train Time: {train_time:.2f}s, Test Time: {test_time:.4f}s")
    plt.grid(True)

    # 保存图像
    save_plot(plt, "3-1-1", params)
    plt.show()

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
    save_plot(plt, "3-1-5", params)  # 保存为 3-1-5
    plt.show()

# 绘制损失曲线（前 100 步和之后）
def plot_loss_curve(params):
    steps, losses = [], []
    with open("loss_history.txt", "r") as f:
        lines = f.readlines()[1:]  # 跳过表头
        for line in lines:
            step, loss = map(float, line.strip().split())
            steps.append(step)
            losses.append(loss)

    # 筛选前 100 步的数据（Step 0 到 Step 100）
    initial_steps = []
    initial_losses = []
    for i, step in enumerate(steps):
        if step <= 100:
            initial_steps.append(step)
            initial_losses.append(losses[i])

    # 筛选第 100 步之后的数据（Step 100 到最后）
    converge_steps = []
    converge_losses = []
    for i, step in enumerate(steps):
        if step >= 100:
            converge_steps.append(step)
            converge_losses.append(losses[i])

    # 绘制前 100 步的损失下降趋势
    plt.figure(figsize=(10, 6))
    plt.plot(initial_steps, initial_losses, label="Initial Loss Decline", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Initial Loss Decline (First 100 Steps)")
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    # 保存图像
    save_plot(plt, "3-1-2", params)
    plt.show()

    # 绘制之后损失的下降趋势（从第 100 步开始）
    plt.figure(figsize=(10, 6))
    plt.plot(converge_steps, converge_losses, label="Convergence Loss", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Convergence Loss (After Step 100)")
    plt.legend()
    plt.grid(True)

    # 保存图像
    save_plot(plt, "3-1-3", params)
    plt.show()

# 绘制绝对误差分布
def plot_error_distribution(params):
    x, pred, exact = [], [], []
    with open("solution_data.txt", "r") as f:
        lines = f.readlines()[1:]  # 跳过表头
        for line in lines:
            xi, predi, exacti = map(float, line.strip().split())
            x.append(xi)
            pred.append(predi)
            exact.append(exacti)

    errors = [abs(pred[i] - exact[i]) for i in range(len(x))]

    # 计算 L2 误差（离散形式）
    l2_error = np.sqrt(np.mean(np.array(errors) ** 2))
    print(f"L2 Error: {l2_error}")

    plt.figure(figsize=(10, 6))
    plt.plot(x, errors, label="Absolute Error")
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error Distribution\n(L2 Error: {l2_error:.6f})")
    plt.legend()
    plt.grid(True)

    # 保存图像
    save_plot(plt, "3-1-4", params)
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
    save_plot(plt, "3-1-6", params)  # 保存为 3-1-6
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
    save_plot(plt, "3-1-7", params)  # 保存为 3-1-7
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
    params["k"] = 3  # 图片名第二位，通过修改此值控制文件名（例如改为 2 生成 3-2-1 等）
    params["d"] = 1  # 1D
    params["dd"] = 1  # Scalar field
    params["bodyBatch"] = 100  # Batch size
    params["bdryBatch"] = 2  # Batch size for the boundary integral
    params["lr"] = 0.0007  # Learning rate
    params["preLr"] = 0.01  # Learning rate (Pre-training)
    params["width"] = 50  # Width of layers
    params["depth"] = 1  # Depth of the network: depth+2
    params["numQuad"] = 100  # Number of quadrature points for testing
    params["trainStep"] = 20000
    params["penalty"] = 1000
    params["preStep"] = 0
    params["diff"] = 0.01
    params["writeStep"] = 10
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["gamma"] = 0.3
    params["decay"] = 0.00001

    # 初始化模型
    model = RitzNet(params).to(device)
    preOptimizer = torch.optim.Adam(model.parameters(), lr=params["preLr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["decay"])
    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])

    # 训练模型
    start_train_time = time.time()
    steps, l2_errors, h1_errors = train(model, device, params, optimizer, scheduler)  # 获取误差数据
    train_time = time.time() - start_train_time
    print(f"Training costs {train_time:.2f} seconds.")

    # 测试模型
    model.eval()
    start_test_time = time.time()
    l2_error, h1_error = test(model, device, params)  # 获取 L2 和 H1 误差
    test_time = time.time() - start_test_time
    print(f"Testing costs {test_time:.4f} seconds.")
    print(f"The L2 error (of the last model) is {l2_error}.")
    print(f"The H1 error (of the last model) is {h1_error}.")

    # 保存时间到文件
    with open("training_time.txt", "w") as f:
        f.write(f"Training Time: {train_time:.2f} seconds\n")
        f.write(f"Test Time: {test_time:.4f} seconds\n")

    # 可视化结果
    pltResult(model, device, 100, params, train_time, test_time)
    plot_loss_curve(params)
    plot_error_distribution(params)
    plot_h1_error_distribution(params, h1_error)
    plot_error_convergence(steps, l2_errors, h1_errors, params)  # 绘制误差-步数曲线

    print(f"The L2 error (of the last model) is {l2_error}.")
    print(f"The number of parameters is {count_parameters(model)}.")

    torch.save(model.state_dict(), "last_model.pt")

if __name__ == "__main__":
    main()