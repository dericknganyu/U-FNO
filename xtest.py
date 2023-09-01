import torch
import numpy as np

np.random.seed(0)

# ntrain = 2
# s = 3
# x_train = torch.randn(ntrain, s, s)
# # print(x_train.reshape(ntrain,s,s,1))
# # print("______________\n \n")
# grids = []
# grids.append(np.linspace(0, 1, s))
# grids.append(np.linspace(0, 1, s))
# grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
# # print (grid)
# # print("______________\n \n")
# grid = grid.reshape(1,s,s,2)
# # print (grid)
# # print("______________\n \n")
# grid = torch.tensor(grid, dtype=torch.float)
# # print (grid.repeat(ntrain,1,1,1))
# # print("______________\n \n")
# x = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
# print(x)
# print(x.shape)
# print("______________\n \n")


# def get_grid(shape):
#     ntrain, s, s = shape[0], shape[1], shape[2]
#     gridx = torch.tensor(np.linspace(0, 1, s), dtype=torch.float)
#     gridx = gridx.reshape(1, s, 1, 1).repeat([ntrain, 1, s, 1])
#     gridy = torch.tensor(np.linspace(0, 1, s), dtype=torch.float)
#     gridy = gridy.reshape(1, 1, s, 1).repeat([ntrain, s, 1, 1])
#     return torch.cat((gridx, gridy), dim=-1)

# x_train = x_train.reshape(ntrain,s,s,1)
# shape = x_train.shape
# grid = get_grid(shape)
# x_train = torch.cat((x_train, grid), dim=-1)
# torch.permute(x_train , (0, 1, 3, 2))
# print(x_train)
# print(x_train.shape)
# print("______________\n \n")







