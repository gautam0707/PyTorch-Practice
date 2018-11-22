import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]], requires_grad=True)
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # one in and one out

    def forward(self, x):
        y_prediction = self.linear(x)
        return y_prediction

model = Model()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
y = torch.Tensor([[4.0]])
y_pred = model(y)
print("predict (after training)",  4, model(y).data[0][0])
