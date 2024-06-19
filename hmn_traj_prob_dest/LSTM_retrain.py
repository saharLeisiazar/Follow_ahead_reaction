import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 2  # 2D points
hidden_size = 64
output_size = 3  # left, straight, right
num_layers = 1
num_epochs = 200

learning_rate = 0.01
seq_length = 6 
input_length = 5
batch_size = 16


def generate_target(seq, input_length):
    #### human's current orientation
    p1 = seq[input_length-1]
    p2 = seq[input_length-2]

    p0 = seq[-1] ## future point

    curr_theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    fut_theta = np.arctan2(p0[1]-p1[1], p0[0]-p1[0])
    alpha = fut_theta - curr_theta

    f_left = max(np.tanh(alpha), 0) **0.5
    f_right = max(np.tanh(-alpha), 0) **0.5
    f_straight = 1 - f_left - f_right

    return [f_left, f_straight, f_right]

def generate_2d_data(seq_length, input_length):
    # data loaidng and preperation
    raw_data = np.load('../datasets/data_3d_h36m.npz', allow_pickle=True)['positions_3d'].item()
    samples = []
    target = []

    dist = 0.5
    angle_list = np.deg2rad(np.arange(5, 13, 1))
    th_list = [40, 34, 30, 27, 24, 22, 20, 19]
    for angle , th in zip(angle_list, th_list):
        # Add U shape trajectories
        seq = [[0,0]]
        theta = 0
        for i in range(1,th+5):
            if i <5 or i>th:
                turn = 0
            else:
                turn = -angle 

            x = seq[i-1][0] + dist * np.cos(turn + theta)
            y = seq[i-1][1] + dist * np.sin(turn + theta)
            seq.append([x,y])
            theta += turn

        for i in range(len(seq)-seq_length-1):
            sample = seq[i:i+seq_length]
            samples.append(sample[:input_length])
            target.append(generate_target(sample, input_length))


        ################
        # right U shape
        seq = [[0,0]]
        theta = 0
        for i in range(1,th+5):
            if i <5 or i>th:
                turn = 0
            else:
                turn = angle

            x = seq[i-1][0] + dist * np.cos(turn + theta)
            y = seq[i-1][1] + dist * np.sin(turn + theta)
            seq.append([x,y])
            theta += turn

        for i in range(len(seq)-seq_length-1):
            sample = seq[i:i+seq_length]
            samples.append(sample[:input_length])
            target.append(generate_target(sample, input_length))


    samples = np.array(samples)
    target = np.array(target)
    return samples, target


# Define the LSTM model for 2D points
class LSTMModel2D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel2D, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.last = nn.Softmax(dim=1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.last(out)
        return out


# Prepare data
data, target = generate_2d_data(seq_length, input_length)
data = data.astype(np.float32)
target = target.astype(np.float32)
# Transform data to be relative to the last point
data -= np.repeat(data[:, input_length-1], input_length, axis=0).reshape(data.shape[0], input_length, 2)

# shuffle data
assert data.shape[0] == target.shape[0]
idx = np.random.permutation(data.shape[0])
data = data[idx]
target = target[idx]


data = torch.from_numpy(data).cuda()
target = torch.from_numpy(target).cuda()

train_size = int(0.90 * data.shape[0])
x_train = data[:train_size]
y_train = target[:train_size]
x_test = data[train_size:]
y_test = target[train_size:]

# Initialize model, loss function, and optimizer
# model = LSTMModel2D(input_size, hidden_size, output_size, num_layers)
model = torch.load('model.pth')
model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5000 , gamma=0.5)

# Training loop
train_loss = []
test_loss = []

for epoch in range(num_epochs):
    model.train()
    idx = np.random.permutation(x_train.shape[0])[:batch_size]
    model_input = x_train[idx]
    target = y_train[idx]

    outputs = model(model_input)
    optimizer.zero_grad()
    loss = criterion(outputs, target)
    train_loss.append(loss.item())     
    loss.backward()
    optimizer.step()

    scheduler.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    # Evaluation
    model.eval() 
    with torch.no_grad():
        test_outputs = model(x_test)
        loss = criterion(test_outputs, y_test)
        test_loss.append(loss.item())
        # print(f'Test Loss: {loss.item():.4f}')
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], test_Loss: {loss.item():.4f}')




fig = plt.figure()
plt.plot(train_loss, label='Train Loss', color='blue')
plt.plot(test_loss, label='Test Loss', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()
fig.savefig('loss.png')
# Plotting x_test, y_test, and y_pred


dist_to_goal = 0.5
for i in range(x_test.shape[0]):
    model.eval()
    with torch.no_grad():
        x = x_test[i]
        y = y_test[i]
        y_pred = model(torch.unsqueeze(x, 0))
        x = x.squeeze().cpu()
        y_pred = y_pred.squeeze().cpu()
        p1 = x[-1].cpu()
        p2 = x[-2].cpu()
        theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
        
    fig = plt.figure()    
    plt.plot(x[:, 0], x[:, 1], label='History', color='blue')
    y_pred = torch.flip(y_pred, [0]) # left, st, right   -> right, st, left

    for j in range(y.shape[0]):
        x_goal = dist_to_goal*np.cos(theta+ (0.5*(j-1))) + p1[0]
        y_goal = dist_to_goal*np.sin(theta+ (0.5*(j-1))) + p1[1]

        plt.scatter(x_goal, y_goal, color='red', s=int(100*y_pred[j]))
        plt.text(x_goal, y_goal, str(round(y_pred[j].item(),2)), fontsize=12, color='black')
            
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title('Ground Truth vs Predicted')
    plt.legend()
    fig.savefig(str(i) + '.png')

torch.save(model, 'model.pth')
print()