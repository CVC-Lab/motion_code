import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Initialize the model
input_dim = 1
num_inducing_points = 10
num_latents = 2
num_outputs = 2  # Number of labels/GPs

device = "cuda:0"

from model.AM import MultiPhaseAMModel
from data.AM_dataset import RowDataset, MultiPhaseDataset, row_collate_fn
from torch.utils.data import DataLoader
data_path = '/mnt/data/yiwang/code/RandomMaterial/clustered_grain/massive_05_testify_bilevel_sampling_2phases_with_clustered_grains'
material_dataset = MultiPhaseDataset(data_path, [512,512], n_phases=2)
row_dataset = RowDataset(material_dataset, rows_per_image=400)
row_loader = DataLoader(row_dataset, batch_size=400, shuffle=False, collate_fn=row_collate_fn)
loaders = iter(row_loader)
train_batch = next(loaders)
test_batch = next(loaders)

Y_train, labels_train = train_batch
X_train = torch.linspace(0, 1, Y_train.shape[1]).reshape(1, -1).repeat(Y_train.shape[0], 1)
Y_train.squeeze_()

Y_test, labels_test = test_batch
Y_test.squeeze_()
X_test = torch.linspace(0, 1, Y_test.shape[1]).reshape(1, -1).repeat(Y_test.shape[0], 1)

# Generate synthetic data with different lengths for each GP
def unpack_data_1d(X, Y, labels):
    # Initialize lists to store x_list and y_list for each class
    x_list_by_class = []
    y_list_by_class = []

    # Iterate through each unique class label
    for label in [0,1]:
        # Filter X_train and Y_train for the current class
        x_list = [x for x, l in zip(X, labels) if l == label]
        y_list = [y for y, l in zip(Y, labels) if l == label]
        # Append the filtered x_list and y_list to the respective lists
        x_list_by_class.append(torch.tensor(np.stack(x_list), dtype=torch.float32).unsqueeze(-1).to(device))
        y_list_by_class.append(torch.tensor(np.stack(y_list), dtype=torch.float32).unsqueeze(-1).to(device))

    return x_list_by_class, y_list_by_class



def classify(model, X_test, Y_test, L_test):
    model.eval()
    with torch.no_grad():
        #X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        #Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        label_predict = model.predict(X_test, Y_test)
        correct = (label_predict.detach().cpu().numpy() == L_test).sum()

        accuracy = 100 * correct / len(L_test)
        print(f"ACC: {accuracy}%")
    model.train()
    return accuracy

def plot_check(label = 0):
    
    # We predict on test time horizon [1, 1.2] with 20 time steps
    test_time_horizon = np.linspace(0, 1.0, 50, endpoint=False)
    # label is the label for the stochastic process we predict
    with torch.no_grad():
        means, covars, S_ms = model.forecast(torch.tensor(test_time_horizon, dtype=torch.float32).to(device).unsqueeze(0))
        Sm = S_ms[label].detach().cpu().numpy()
        prior_mean = model.freq_model.gaussian_processes[label].variational_mean.squeeze().detach().cpu().numpy()
        mean = means[label].squeeze().detach().cpu().numpy()
        covar = covars[label].squeeze().detach().cpu().numpy()

    std = np.sqrt(np.diag(covar)).reshape(-1)
    #print(std)


    # We plot all the time series on the horizon [0, 1]
    fig = plt.figure()
    X = X_train[labels_train==label, :]
    Y = Y_train[labels_train==label, :]
    Y, wave = model.transmute(Y)
    
    plt.plot(test_time_horizon, Y[0], c='blue', lw=0.5, zorder=1, label='Observation')
    for i in range(1, X.shape[0]):
        plt.plot(test_time_horizon, Y[i], c='blue', lw=0.5, zorder=1)

    

    # Then we plot our prediction with uncertainty on test time horizon = [1, 1.2]
    plt.plot(test_time_horizon, mean, c='red', lw=2, zorder=1, label='Mean prediction')
    plt.fill_between(test_time_horizon, mean+2*std, mean-2*std, color='red', alpha=0.1, zorder=1)
    plt.scatter(Sm, prior_mean, c='green', s=45, label='most informative timestamps')

    # Plot setting and plot show
    handle_list, _ = plt.gca().get_legend_handles_labels()
    handle_list.append(mpatches.Patch(color='red', label='Uncertainty region'))
    plt.legend(handles=handle_list, fontsize='10', loc ="lower left")
    #M = 1.1*np.max(np.abs(Y_test))
    #plt.ylim(-M, M)
    plt.title('Time series forecasting by motion code with filtering class {}'.format(label))
    return fig

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs/debug/AM')

model = MultiPhaseAMModel(input_dim, num_inducing_points, num_latents, num_outputs, sigma_y=0.1).to(device)

x_list_by_class, y_list_by_class = unpack_data_1d(X_train, Y_train, labels_train)

# Training loop
num_epochs = 1000
model.train()
# Optimizer
kernel_vars = [p for name, p in model.named_parameters() if 'kernel' in name]
encoding_vars = [p for name, p in model.named_parameters() if 'kernel' not in name]

from model.optimizers import SGLD, SGHMC
optimizer = torch.optim.Adam([
    {'params': kernel_vars, 'lr': 1e-3},
    {'params': encoding_vars, 'lr': 5e-3}
    ],
    lr=1e-2)
best_ACC = 0
num_iters = 1
for _ in range(num_iters):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Compute loss
        loss = model.compute_loss(x_list_by_class, y_list_by_class, epoch=epoch)

        # Backward pass
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        optimizer.step()

        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.10f}")
            ACC = classify(model, X_test, Y_test, labels_test)
            best_ACC = max(ACC, best_ACC)
            if best_ACC == ACC:
                torch.save(model.state_dict(), "saved_models/am_test.pth")
            writer.add_scalar('training_loss', loss, epoch)
            writer.add_scalar('test_acc', ACC, epoch)
            writer.add_scalar('process_0_lengthscale', model.freq_model.gaussian_processes[0].kernel.lengthscale , epoch)
            writer.add_scalar('process_0_variance', model.freq_model.gaussian_processes[0].kernel.variance , epoch)
            writer.add_figure('check process 0', plot_check(), global_step=epoch)
            writer.add_figure('check process 1', plot_check(label=1), global_step=epoch)


print(best_ACC)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
