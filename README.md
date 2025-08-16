# Linear Regression with Gradient Descent

## Overview
This project implements a simple linear regression model using gradient descent. The model reads data from a CSV file, iteratively updates the slope (`m`) and intercept (`b`), and then visualizes the results using Matplotlib.

## Features
- Reads dataset using Pandas
- Implements gradient descent to optimize `m` and `b`
- Trains the model over multiple epochs
- Visualizes the fitted line against data points

## Prerequisites
Ensure you have Python installed along with the necessary dependencies:

```sh
pip install pandas matplotlib
```

## Usage
1. Place your dataset in the `notebook` directory and name it `data.csv`.
2. Run the script:

```sh
python app.py
```

## Code Breakdown
### Importing Libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
```

### Loading the Data
```python
data = pd.read_csv(r'notebook\data.csv')
print(data.head())
```

### Gradient Descent Function
```python
def gradient_descent(m_now, b_now, points, L):
    m_grad = 0
    b_grad = 0
    n = len(points)
    
    for i in range(n):
        x1 = points.iloc[i].x
        y1 = points.iloc[i].y
        
        m_grad += -(2/n) * x1 * (y1 - (m_now * x1 + b_now))
        b_grad += -(2/n) * (y1 - (m_now * x1 + b_now))
        
    m = m_now - m_grad * L
    b = b_now - b_grad * L
    
    return m, b
```

### Training the Model
```python
m = 0
b = 0
L = 0.00001
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
```

### Visualizing the Results
```python
plt.scatter(data.x, data.y, color="black")
plt.plot(list(range(0, 10)), [m * x + b for x in range(0, 10)])
plt.show()
```

## Output
The script prints the final values of `m` and `b` and displays a scatter plot with the regression line.

## License
This project is open-source and available under the [MIT License](LICENSE).

