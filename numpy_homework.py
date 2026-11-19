# NP-E1
import numpy as np

A = np.array([[0.20, -0.50, 1.10, 0.40],
              [0.80, 0.30, -0.70, 0.60],
              [-0.40, 0.90, 0.10, -0.20]])

print("A.ndim  =", A.ndim)
print("A.shape =", A.shape)
print("A.size  =", A.size)

row1 = A[1]
print("A[1] =", row1, " shape =", row1.shape)

last2 = A[:, -2:]
print("A[:, -2:] =", last2, " shape =", last2.shape)

B = A.copy()
B[B < 0] = 0.0
print("B =", B)
print("A =", A)

# NP-E2
import numpy as np

a = np.array([0.80, -0.60, 0.70])
b = np.array([-0.20, 0.90, 0.40])

#逐元素乘
prod = a * b
print("a * b =", prod, " shape =", prod.shape)

#点乘
dot = (a @ b)
print("a @ b =", dot, " shape =", np.shape(dot))

lena = np.sqrt((a ** 2).sum())
lenb = np.sqrt((b ** 2).sum())
print("a欧=", lena)
print("b欧=", lenb)

#核对
lenaref = np.linalg.norm(a)
lenbref = np.linalg.norm(b)
print("a欧=", lenaref)
print("b欧", lenbref)

#比较
print("两种算法一致:", np.allclose(lena, lenaref) and np.allclose(lenb, lenbref))
#接近

# NP-E3
import numpy as np

a = np.array([1.00, -1.00])
b = np.array([1.00, 0.00])

cos_theta =(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("cos(theta) 原始值 =", cos_theta)

#clip限制
cos_theta = np.clip(cos_theta, -1.0, 1.0)
print("clip 之后 =", cos_theta)

#求角度转度
theta_rad = np.arccos(cos_theta)
theta_du = np.degrees(theta_rad)
#theta * 180 / np.pi
print("角度(弧度) =", theta_rad)
print("角度(度) =", theta_du)

#验证
print("是否等于 45 度:", np.isclose(theta_du, 45.0))             #单体判断

#函数处理零向量
def angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("a、b 必须是同维的一维向量")
    if not np.any(a) or not np.any(b):                         #有一个非零即true
        raise ValueError("零向量没有方向，无法定义夹角")

    cos_theta = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))             #np的float变成python的float
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

#测试
try:
    angle_deg(np.zeros(2), b)
except ValueError as e:
    print("零向量测试 → ValueError:", e)

# NP-E4
import numpy as np

# 1. 给定两个点
p = np.array([-0.50, 1.20, 2.00])
q = np.array([0.80, -0.40, 1.10])

diff = p - q
print("p - q =", diff, " shape =", diff.shape)

#欧氏距
oul = np.sqrt((diff ** 2).sum())
print("欧氏距离ss=", oul)
print("欧氏距离（norm） =", np.linalg.norm(diff))                           #ord=2

#曼哈顿距离
mhd = np.abs(diff).sum()
print("曼哈顿距离ss=", mhd)
print("曼哈顿距离（norm=1） =", np.linalg.norm(diff, ord=1))

# 5. 验证关系
print("欧氏 <= 曼哈顿:", oul<= mhd)









#ord=1	L1 范数	|x₁| + |x₂| + ... + |xₙ|	曼哈顿距离
#ord=2（默认）	L2 范数	sqrt(x₁² + x₂² + ... + xₙ²)	欧氏距离
#ord=np.inf	无穷范数	max(|x₁|, |x₂|, ..., |xₙ|)	最大分量

# NP-E5
import numpy as np

a = np.array([-1.20, 0.40, 0.80])

u = a / np.linalg.norm(a)
print("单位向量u=",u)

print("|u|=", np.linalg.norm(u))
print("np.allclose(|u|, 1.0):", np.allclose(np.linalg.norm(u), 1.0))


#abs(np.linalg.norm(u) - 1.0) < 1e-6

# NP-E6
import numpy as np

A = np.array([[0.20, -0.50, 1.10],
              [0.80,  0.40, -0.30]])
A1 = A.sum(axis=0)
print("A.sum(axis=0) =", A1, " shape =", A1.shape)

A2 = A.mean(axis=1)
print("A.mean(axis=1) =", A2, " shape =", A2.shape)

max= A.max()
max_pos = np.unravel_index(np.argmax(A), A.shape)           #argmax()以为坐标   unravel_index还原坐标
print("A.max() =", max, " 位置 =", max_pos)




'''centered = A - A.mean(axis=1, keepdims=True)
print("A - 行均值(keepdims) =", centered)
print("中心化后每行和 ≈ 0:", np.allclose(centered.sum(axis=1), 0.0))'''

# NP-E7
import numpy as np

X = np.array([[0.80, -0.60, 0.70],
              [-0.40, 0.50, 0.20]])
scale = np.array([1.50, 0.50, -1.00])
bias = np.array([-0.10, 0.20, 0.30])

Y = X * scale + bias
print("Y = X * scale + bias =")
print(Y)
print("Y.shape =", Y.shape)

print("X.shape     =", X.shape)
print("scale.shape =", scale.shape)
print("bias.shape  =", bias.shape)

# NP-E8
import numpy as np

#a，shape (C, H, W)
a = np.array([[[0.20, -0.40], [0.70, 0.10]],
              [[0.30, 0.80], [-0.20, 0.50]],
              [[-0.10, 0.60], [0.40, -0.30]]])
b = a + 0.10

print("a.shape =", a.shape)                     #(C, H, W)

#组成batch
batch = np.stack([a, b], axis=0)         #(B, C, H, W)
print("batch shape =", batch.shape)

#提取蓝色通道（C=2）
blue = batch[:, 2]
print("blue.shape =", blue.shape, )       #(B, H, W)

#转成 (B,H,W,C)  bchw->bhwc
batch_bhwc = np.transpose(batch, (0, 2, 3, 1))
print("batch_bhwc.shape=", batch_bhwc.shape)

'''
cat = np.concatenate([a, b], axis=0)
print("np.concatenate.shape =", cat.shape")'''#不升维

# NP-M1
import numpy as np


def jisuan_jiajiao(X, q):
    chaxun = np.asarray(X, dtype=float)
    juzhen = np.asarray(q, dtype=float)

    #长度
    changdu_chaxun = np.linalg.norm(chaxun)

    #不能是零
    if changdu_chaxun == 0:
        raise ValueError("不要0向量")

    #批量点成
    dianji = juzhen @ chaxun

    #求矩阵长度
    changdu_juzhen = np.linalg.norm(juzhen, axis=1)

    #分母
    fenmu = changdu_juzhen * changdu_chaxun

    fenmu_safe = np.where(fenmu == 0, 1.0, fenmu)

    #余弦值
    yuxian = dianji / fenmu_safe

    #截断[-1, 1]
    yuxian = np.clip(yuxian, -1.0, 1.0)

    #反余弦转角度
    jiaodu = np.degrees(np.arccos(yuxian))


    # 10. 零向量行覆盖为
    jiaodu = np.where(changdu_juzhen == 0, np.nan, jiaodu)     #条件 true false       nan无意义

    return jiaodu


chaxun = np.array([0.60, -0.80, 0.00])
juzhen = np.array([[0.30, -0.40, 0.00],
                   [-0.60, 0.80, 0.00],
                   [0.80, 0.60, 0.00],
                   [0.00, 0.00, 0.00]])

result = jisuan_jiajiao(chaxun, juzhen)
print("jiaodu =", result)
print("shape =", result.shape)

# NP-M2
import numpy as np


def qian_k_yuxian(query, gallery, k):
    query = np.asarray(query, dtype=float)
    gallery = np.asarray(gallery, dtype=float)


    changdu_query = np.linalg.norm(query)
    if changdu_query == 0:
        raise ValueError("query 为0")

    #所有点乘和行长度
    dianji = gallery @ query
    changdu_gallery = np.linalg.norm(gallery, axis=1)

    #排除零向量，记下原始行号
    youxiao_mask = changdu_gallery > 0
    youxiao_idx = np.where(youxiao_mask)[0]                         #返下标

    #对有效行算余弦
    yuxian = dianji[youxiao_mask] / (changdu_gallery[youxiao_mask] * changdu_query)
    yuxian = np.clip(yuxian, -1.0, 1.0)

    #k合法性检查
    n_youxiao = yuxian.size
    if not isinstance(k, (int, np.integer)) or k <= 0 or k > n_youxiao:
        raise ValueError(f"k 必须是1-{n_youxiao}的正整数")

    paixu = np.argsort(-yuxian, kind="stable")    #稳定排

    #取前k 翻译回原始行号
    top_xiangdui = paixu[:k]
    top_yuanshi = youxiao_idx[top_xiangdui]
    top_fenshu = yuxian[top_xiangdui]

    return top_yuanshi, top_fenshu


query = np.array([0.60, -0.80, 0.00])
gallery = np.array([[0.60, -0.80, 0.00],
                    [0.30, -0.40, 0.00],
                    [0.80, 0.60, 0.00],
                    [-0.60, 0.80, 0.00],
                    [0.00, 0.00, 0.00]])
ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])

top_idx, top_score = qian_k_yuxian(query, gallery, k=3)
print("top_idx   =", top_idx)
print("top_score =", top_score)
print("top_ids   =", ids[top_idx])

# NP-M3
import numpy as np

A = np.array([[0.20, -0.40],
              [0.80,  0.30],
              [-0.50, 0.70]])   # (3, 2)
B = np.array([[0.10,  0.20],
              [-0.60, -0.20]])  # (2, 2)
#312 122
#广播构造差数组
diff = A[:, None, :] - B[None, :, :]
print("diff.shape =", diff.shape)

#最后一维求范数，得距离矩阵
D = np.linalg.norm(diff, axis=2)
print("D =")
print(D)
print("D.shape =", D.shape)

#A的每一行找最近B索引
nearest = np.argmin(D, axis=1)
print("nearest =", nearest)

# NP-M4
import numpy as np

X  = np.array([[0.80, -0.60, 0.70],
               [-0.40, -0.50, 0.90]])
W1 = np.array([[-0.10, 0.20, -0.80],
               [0.30, -0.50, 0.10]])
b1 = np.array([-0.15, 0.05])
W2 = np.array([[0.20, -0.60],
               [-0.80, 0.52]])
b2 = np.array([0.10, -0.20])

#第一层线性运算
Z = X @ W1.T + b1
print("Z =")
print(Z)
print("Z.shape =", Z.shape)

#ReLU负 数变 0
H = np.maximum(Z, 0.0)
print("\nH =")
print(H)
print("H.shape =", H.shape)

#第二层线性运算
Y = H @ W2.T + b2
print("\nY =")
print(Y)
print("Y.shape =", Y.shape)
