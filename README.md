# distance_loss_torch
The Official Repository for Distance Loss Function with pytorch.

## Distance Loss Function
Distance loss function based on weighted mean square error. It can apply to `monotonic classificaiton`. Below is a formula of `Distance Mean Square(DiMS)` loss function.

$$
L_{DiMS} = \frac{1}{nl}\sum\limits_{i=1}^{n} \sum\limits_{j=1}^{l} (\lvert A(T_{i}) - j \rvert + 1)^{\alpha} (T_{ij} - Y_{ij})^{2}
$$

where

$$
A(L) = \underset{x \in S}{argmax} L_{x} = \lbrace x \in S|L(s) \leq L(x), \forall s \in S \rbrace
$$

$$
S = \lbrace s|1 \leq s \leq l, s \in N\rbrace
$$

$\alpha$ is a hyper-parameter for tuning. It can determine experimentally.
## How to Use
You can use distance loss function by using pypi package. Please type below on your terminal.

```bash
pip install distance_loss_torch
```

```python
from distanceLoss import DiMSLoss

model = Model()
loss_fn = DiMSLoss(alpha = 2)
'''
Your Model
'''
y_pred = model(x_train)
loss = loss_fn(y_pred, y_train)
loss.backward()
```
## Implements of distance loss function
- [Distance-Loss-Experiments](https://github.com/9tailwolf/Distance-Loss-Experiments) : The official repository for optain experimental performance result on distance loss functions.
- [Distance-Loss-Experiments_SST-5](https://github.com/9tailwolf/Distance-Loss-Experiment_SST-5) : The official repository for achieve score on SST-5 semantic analysis. 
