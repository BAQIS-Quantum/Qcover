import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
node_d, p = [2], [3, 4, 5, 6, 7]
Fourier_tm = [39.82834434509277, 70.68283319473267, 221.981107711792, 219.1239206790924, 961.8482122421265]
COBYLA_tm = [12.406248569488525, 46.92031240463257, 112.66444778442383, 414.25853872299194, 3002.007876396179]

Fourier_ep = [-17.52356047443737, -19.95655456409755, -20.121290948456906, -83.90646488094582, -17.878072229328218]
COBYLA_ep = [-75.40026093126744, -32.05658765533073, -28.55836130045919, -29.414369385675837, -46.222880091593204]

plt.figure(1)
plt.plot(range(3, 8), Fourier_tm, "ob-", label="Fourier")
plt.plot(range(3, 8), COBYLA_tm, "^r-", label="COBYLA")
plt.ylabel('Times cost')
plt.xlabel('P value')
plt.title("node = 23, degree = 2")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(range(3, 8), Fourier_ep, "ob-", label="Fourier")
plt.plot(range(3, 8), COBYLA_ep, "^r-", label="COBYLA")
plt.ylabel('Expectation value')
plt.xlabel('P value')
plt.title("node = 23, degree = 2")
plt.legend()
plt.show()



# plt.figure(1)
# 手动设置标度
# x = [1, 4, 8, 12, 16, 25, 40, 60]            #虚假的x值，用来等间距分割
# x_index = [str(e) for e in num_nodes]  # ['4', '6', '8', '10', '12', '14', '100', 1000, 10000]   x 轴显示的刻度
# plt.plot(x[:len(qiskit_tm)], qiskit_tm, "ob-", label="Qiskit", marker='d')
# plt.plot(x, qcover_tm, "^r-", label="QCover", marker='d')
# plt.xticks(x, x_index)  # range(1, len(num_nodes) + 1)

# plt.plot(num_nodes[:len(qiskit_tm)], qiskit_tm, "ob-", label="Qiskit", marker='d')
# plt.plot(num_nodes, qcover_tm, "*r-", label="QCover", marker='d')
#设置指数标度
# plt.xscale("symlog")
# plt.yscale("symlog")
# plt.xticks(x, x_index)  # range(1, len(num_nodes) + 1)
# plt.ylabel('single query times/(log10 s)')
# plt.xlabel('node numbers')

# plt.legend()
# plt.savefig('/home/wfzhuang/data/Qcover/result_log/backends_compare/res_serial_%s.png' % str(i))
# plt.savefig('/public/home/humengjun/QCover/result_log/maxcut_time_large.png')
# plt.savefig('/home/puyanan/QCover/result_log/backends_compare/tm_serial_%s.png' % str(i))
# plt.savefig('E:/Working_projects/QAOA/QCover/result_log/maxcut_time_large.png')  # maxcut_serial
# plt.show()
# plt.cla()


# 不等距坐标轴
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# plt.rcParams['font.sans-serif']=['SimHei']         # 处理中文无法正常显示的问题 成功
# plt.rcParams['axes.unicode_minus'] = False #负号显示
# plt.xlabel("x轴")  # 设置x轴名称
# plt.ylabel("y轴")  # 设置y轴名称
# plt.title("标题")  # 设置标题
#
# x=[1,2,3,4,5,6]                    #虚假的x值，用来等间距分割
# x_index=['1','10','100','1000','10000','100000']  # x 轴显示的刻度
# y=[0.1,0.15,0.2,0.3,0.35,0.5]       #y值
# plt.plot(x,y,marker='d')
# _ = plt.xticks(x,x_index)           # 显示坐标字
# plt.show()
#
# # 纵坐标刻度为10^n
# import matplotlib.pyplot as plt
# y=[pow(10,i) for i in range(0,10)]
# x=range(0,len(y))
# plt.plot(x, y, 'r')
# plt.yscale('log')#设置纵坐标的缩放，写成m³格式
# plt.show()