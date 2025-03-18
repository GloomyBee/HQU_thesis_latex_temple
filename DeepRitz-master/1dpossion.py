import numpy as np
import math, torch, generateData, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import writeSolution


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

"""
def preTrain(model, device, params, preOptimizer, preScheduler, fun):
    model.train()
    file = open("lossData.txt", "w")

    for step in range(params["preStep"]):
        # The volume integral
        data = torch.from_numpy(generateData.sampleFromDisk(params["radius"], params["bodyBatch"])).float().to(device)

        output = model(data)

        target = fun(params["radius"], data)

        loss = output - target
        loss = torch.mean(loss * loss) * math.pi * params["radius"] ** 2

        if step % params["writeStep"] == params["writeStep"] - 1:
            with torch.no_grad():
                ref = exact(params["radius"], data)
                error = errorFun(output, ref, params)
                # print("Loss at Step %s is %s."%(step+1,loss.item()))
                print("Error at Step %s is %s." % (step + 1, error))
            file.write(str(step + 1) + " " + str(error) + "\n")

        model.zero_grad()
        loss.backward()

        # Update the weights.
        preOptimizer.step()
        # preScheduler.step()
"""

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
    return 0.5 * x**2 + 0.5 * x

# 误差计算
def errorFun(output, target):
    error = output - target
    error = math.sqrt(torch.mean(error**2))
    ref = math.sqrt(torch.mean(target**2))
    return error/ref

# 测试函数
def test(model, device, params):
    numQuad = params["numQuad"]
    data = torch.from_numpy(sampleFromInterval(0, 1, params["numQuad"])).float().to(device)
    output = model(data)
    target = exact(data).to(device)
    return errorFun(output, target)

# 训练函数
def train(model, device, params, optimizer, scheduler):
    model.train()

    # 初始化采样点
    data_body = torch.from_numpy(sampleFromInterval(0, 1, params["bodyBatch"])).float().to(device)
    data_body.requires_grad = True
    data_boundary = torch.from_numpy(sampleFromBoundary(0, 1)).float().to(device)

    for step in range(params["trainStep"]):
        # 内部点输出
        output_body = model(data_body)
        dfdx = torch.autograd.grad(output_body, data_body, grad_outputs=torch.ones_like(output_body),
                                   retain_graph=True, create_graph=True, only_inputs=True)[0]

        # 体积分损失
        fTerm = ffun(data_body).to(device)
        loss_body = torch.mean(0.5 * dfdx**2 - fTerm * output_body)

        # 边界条件损失
        output_boundary = model(data_boundary)
        loss_boundary = torch.mean((output_boundary[0] - 0)**2 + (output_boundary[1] - 1)**2) * params["penalty"]

        # 总损失
        loss = loss_body + loss_boundary

        # 打印损失
        if step % params["writeStep"] == 0:
            print(f"Step {step}: Loss = {loss.item()}")

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
"""
def rough(r, data):
    # A rough guess
    output = r ** 2 - r * torch.sum(data * data, dim=1) ** 0.5
    return output.unsqueeze(1)
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# 可视化
def pltResult(model, device, nSample, params):
    x = np.linspace(0, 1, nSample).reshape(-1, 1)
    pred = model(torch.from_numpy(x).float().to(device)).detach().cpu().numpy()
    exact_sol = exact(x)

    plt.plot(x, pred, label="Predicted Solution")
    plt.plot(x, exact_sol, label="Exact Solution", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.title("#Deep Ritz Solution vs Exact Solution") #1D Rod Problem Solution
    plt.show()

# 主函数
def main():
    # Parameters
    # torch.manual_seed(21)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 强制初始化CUDA上下文
    if torch.cuda.is_available():
        dummy_tensor = torch.zeros(1).cuda()
        dummy_tensor += 1
    params = dict()
    """params["radius"] = 1 # 半径"""
    params["d"] = 1  # 1D
    params["dd"] = 1  # Scalar field
    params["bodyBatch"] = 100  # Batch size
    params["bdryBatch"] = 2  # Batch size for the boundary integral
    params["lr"] = 0.001  # Learning rate
    params["preLr"] = 0.01  # Learning rate (Pre-training)
    params["width"] = 50  # Width of layers
    params["depth"] = 1  # Depth of the network: depth+2
    params["numQuad"] = 100  # Number of quadrature points for testing
    params["trainStep"] = 10000
    params["penalty"] = 500
    params["preStep"] = 0
    params["diff"] = 0.01
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["gamma"] = 0.3
    params["decay"] = 0.00001
# 初始化模型
    startTime = time.time()
    model = RitzNet(params).to(device)
    print("Generating network costs %s seconds." % (time.time() - startTime))

    preOptimizer = torch.optim.Adam(model.parameters(), lr=params["preLr"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["decay"])
    scheduler = StepLR(optimizer, step_size=params["step_size"], gamma=params["gamma"])
# 训练模型
    startTime = time.time()
    # preTrain(model, device, params, preOptimizer, None, rough)
    train(model, device, params, optimizer, scheduler)
    print("Training costs %s seconds." % (time.time() - startTime))

# 测试模型
    model.eval()
    testError = test(model, device, params)
    print("The test error (of the last model) is %s." % testError)
    print("The number of parameters is %s," % count_parameters(model))

    torch.save(model.state_dict(), "last_model.pt")

    pltResult(model, device, 100, params)


if __name__ == "__main__":
    main()