import time
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from Assignment1_Model_Definition import CnnRegressor
from Assignment1_Model_Loss import model_loss
from Assignment1_Split_Data import x_train_np, y_train_np

# %%
batch_size = 32
model = CnnRegressor(batch_size, X.shape[1], 1)

model.cuda()

epochs = 50

optimizer = SGD(model.parameters(), lr=1e-5)

inputs = torch.from_numpy(x_train_np).cuda().float()
outputs = torch.from_numpy(y_train_np.reshape(y_train_np.shape[0], 1)).cuda().float()

tensor = TensorDataset(inputs, outputs)

loader = DataLoader(tensor, batch_size, shuffle=True, drop_last=True)

tic = time.perf_counter()  # starts a timer for the training time

for epoch in range(epochs):
    avg_loss, avg_r2_score, avg_mse = model_loss(model, loader, train=True, optimizer=optimizer)
    print("Epoch " + str(epoch + 1) + ":\n\tLoss = " + str(avg_loss) + "\n\tR^2 Score = " + str(
        avg_r2_score) + ":\n\tMSE = " + str(avg_mse))

toc = time.perf_counter()  # ends the timer

print(f"Model was trained in {toc - tic:0.4f} seconds")

inputs = torch.from_numpy(x_test_np).cuda().float()
outputs = torch.from_numpy(y_test_np.reshape(y_test_np.shape[0], 1)).cuda().float()

avg_loss, avg_r2_score, avg_mse = model_loss(model, loader)

print("The model's L1 loss is:" + str(avg_loss))
print("The model's R^2 score is:" + str(avg_r2_score))
print("The model's MSE is:" + str(avg_mse))