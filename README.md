# Trotter-Error-Interference

[`Error interference in quantum simulation`](https://arxiv.org/pdf/xxxx.pdf).
Boyang Chen, Jue Xu, Xiao Yuan and Qi Zhao, 2024
[![https://arxiv.org/abs/xxxx](https://img.shields.io/badge/paper%20%28v1%29-arXiv%3A24xx.xxxxx-B31B1B)](https://arxiv.org/abs/xxxx)


## Tri-grouping Hamiltonian 
<!-- ![Figure](./figs/step.png) -->
More details in [1D Heisenberg model](./code/nearest_neighbor.ipynb) and [1D Fermi-Hubbard model](./code/hubbard.ipynb).

## Second-order Product Formula
<!-- ![Figure](./figs/random.png) -->
More details in [1D TF Ising model](./code/second_order.ipynb) .

## Main Reference
- M. C. Tran, S. K. Chu, Y. Su, A. M. Childs, A. Gorshkov, [Destructive Error Interference in Product-Formula Lattice Simulation](http://arxiv.org/abs/1912.11047), Physical Review Letter 124, 22, 220502 (2020).
- A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, and S. Zhu, [Theory of Trotter Error with Commutator Scaling](https://arxiv.org/abs/1912.08854), Physical Review X 11, 011020 (2021).
- Layden, [First-Order Trotter Error from a Second-Order Perspective](http://arxiv.org/abs/2107.08032), Physical Review Letter 128, 21, 210501 (2022).

## Usage 
```
# Create python environment
conda create --name myenv python=3.10 

# Install requirements
pip install -r ./code/requirements.txt 
```
