import numpy as np
import plotly.graph_objs as go

# 定义目标函数
def f(x, y):
    return np.sin(x) * np.cos(y)

# 定义模拟退火算法
def simulated_annealing(f, x0, y0, T0, alpha, n_iter):
    x = x0
    y = y0
    T = T0
    history = [(x, y, f(x, y))]
    for i in range(n_iter):
        # 随机生成新的状态
        x_new = x + np.random.normal(scale=0.1)
        y_new = y + np.random.normal(scale=0.1)
        # 计算新状态的目标函数值
        f_new = f(x_new, y_new)
        # 计算能量差
        delta_E = f_new - f(x, y)
        # 判断是否接受新状态
        if delta_E > 0 or np.exp(delta_E / T) > np.random.rand():
            x = x_new
            y = y_new
            history.append((x, y, f_new))
        else:
            history.append((x, y, f(x, y)))
        # 降低温度
        T *= alpha
    return history

# 运行模拟退火算法
history = simulated_annealing(f, x0=1, y0=1, T0=20, alpha=0.99, n_iter=2000)

# 绘制目标函数曲面
x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
surface = go.Surface(x=X, y=Y, z=Z)

# 绘制模拟退火算法的路径
path = go.Scatter3d(
    x=[p[0] for p in history],
    y=[p[1] for p in history],
    z=[p[2] for p in history],
    mode='lines',
    line=dict(
        color='red',
        width=2
    )
)

# 把最后一个点用大红点标出来
last_point = go.Scatter3d(
    x=[history[-1][0]],
    y=[history[-1][1]],
    z=[history[-1][2]],
    mode='markers',
    marker=dict(
        color='green',
        size=10,
    )
)

# 把第一个点用小旗子标出来
first_point = go.Scatter3d(
    x=[history[0][0]],
    y=[history[0][1]],
    z=[history[0][2]],
    mode='markers',
    marker=dict(
        color='blue',
        size=10,
    )
)

fig = go.Figure(data=[surface, path, last_point, first_point])
fig.show()
