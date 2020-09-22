import torch

class CustomNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(CustomNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.relu    = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self.linear1(x)
        h_relu = self.relu(h)
        y_pred = self.linear2(h_relu)
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = CustomNet(D_in, H, D_out)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Backprop to compute gradients of w1 and w2 with respect to loss
    loss.backward()

    # Update weights
    optimizer.step()

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()



