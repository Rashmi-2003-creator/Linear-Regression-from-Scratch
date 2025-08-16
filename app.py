import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'notebook\data.csv')

print(data.head())

def gradient_descent(m_now, b_now, points, L):
    m_grad = 0
    b_grad  = 0
    n = len(points)
    
    for i in range(n):
        x1 = points.iloc[i].x
        y1 = points.iloc[i].y
        
        m_grad += -(2/n) * x1 * (y1 - (m_now * x1 + b_now))
        b_grad += -(2/n) * (y1 - (m_now * x1 + b_now))
        
    m = m_now - m_grad * L
    b = b_now - b_grad * L
    
    return m, b

m = 0
b = 0
L = 0.01
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
    
print(m, b)

plt.scatter(data.x, data.y, color = "black")
plt.plot(list(range(0, 12)), [m * x + b for  x in range(0, 12)])
plt.show()