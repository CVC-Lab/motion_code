from model.moskgp import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Initialize the model
input_dim = 1
num_inducing_points = 10
num_latents = 2
num_outputs = 2  # Number of labels/GPs

device = "cuda:0"
model = MultiOutputSparseGPLayer(input_dim, num_inducing_points, num_latents, num_outputs, sigma_y=0.1).to(device)

from data_processing import load_data, process_data_for_motion_codes

# In this example, we use ItalyPowerDemand dataset. First we load dataset include Y-values of the series and their labels
name = 'ItalyPowerDemand'
Y_train, labels_train = load_data(name=name, split='train')

print(Y_train.shape, labels_train.shape)
# Then we process the data for motion code model and generate X-variable, which is needed for training.
X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train, labels_train)

# Now we load the test set
Y_test, labels_test = load_data(name=name, split='test')
X_test, Y_test, labels_test = process_data_for_motion_codes(Y_test, labels_test)

print(X_train.shape, Y_train.shape, labels_train.shape)
print(X_test.shape, Y_test.shape, labels_test.shape)

# Optimizer
kernel_vars = [p for name, p in model.named_parameters() if 'kernel' in name]
encoding_vars = [p for name, p in model.named_parameters() if 'kernel' not in name]

from model.optimizers import SGLD, SGHMC
optimizer = torch.optim.Adam([
    {'params': kernel_vars, 'lr': 1e-3},
    {'params': encoding_vars, 'lr': 5e-3}
    ],
    lr=1e-2)
# optimizer = SGHMC(
#     [
#     {'params': kernel_vars, 'lr': 1e-3},
#     {'params': encoding_vars, 'lr': 1e-2}
#     ],
#     num_burn_in_steps=0,
#     mdecay=0.9,
#     lr=5e-3)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
def unpack_data_1d(X,Y,labels):
    unique_labels = sorted(set(labels), key=float)

    # Initialize lists to store x_list and y_list for each class
    x_list_by_class = []
    y_list_by_class = []

    # Iterate through each unique class label
    for label in unique_labels:
        # Filter X_train and Y_train for the current class
        x_list = [x for x, l in zip(X, labels) if l == label]
        y_list = [y for y, l in zip(Y, labels) if l == label]
        
        # Append the filtered x_list and y_list to the respective lists
        x_list_by_class.append(torch.tensor(np.stack(x_list), dtype=torch.float32).unsqueeze(-1).to(device))
        y_list_by_class.append(torch.tensor(np.stack(y_list), dtype=torch.float32).unsqueeze(-1).to(device))

    return x_list_by_class, y_list_by_class

# Generate synthetic data with different lengths for each GP
x_list_by_class, y_list_by_class = unpack_data_1d(X_train, Y_train, labels_train)

# Training loop
num_epochs = 1000
model.train()


def classify(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)
        label_predict = model.predict(X_test, Y_test)
        correct = (label_predict.detach().cpu().numpy() == labels_test).sum()

        accuracy = 100 * correct / len(labels_test)
        print(f"ACC: {accuracy}%")
    model.train()
    return accuracy

def plot_check(label = 0):
    
    # We predict on test time horizon [1, 1.2] with 20 time steps
    test_time_horizon = np.linspace(0, 1.1, 100)
    # label is the label for the stochastic process we predict
    with torch.no_grad():
        means, covars, S_ms = model.forecast(torch.tensor(test_time_horizon, dtype=torch.float32).to(device).unsqueeze(0))
        Sm = S_ms[label].detach().cpu().numpy()
        prior_mean = model.gaussian_processes[label].variational_mean.squeeze().detach().cpu().numpy()
        mean = means[label].squeeze().detach().cpu().numpy()
        covar = covars[label].squeeze().detach().cpu().numpy()

    std = np.sqrt(np.diag(covar)).reshape(-1)
    #print(std)


    # We plot all the time series on the horizon [0, 1]
    fig = plt.figure()
    X = X_train[labels_train==label, :]
    Y = Y_train[labels_train==label, :]
    plt.plot(X[0], Y[0], c='blue', lw=0.5, zorder=1, label='Observation')
    for i in range(1, X.shape[0]):
        plt.plot(X[i], Y[i], c='blue', lw=0.5, zorder=1)

    

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
writer = SummaryWriter('./logs/debug/MOSKGP')

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
            ACC = classify(model, X_test, Y_test)
            best_ACC = max(ACC, best_ACC)
            writer.add_scalar('training_loss', loss, epoch)
            writer.add_scalar('test_acc', ACC, epoch)
            writer.add_scalar('process_0_lengthscale', model.gaussian_processes[0].kernel.lengthscale , epoch)
            writer.add_scalar('process_0_variance', model.gaussian_processes[0].kernel.variance , epoch)
            writer.add_figure('check process 0', plot_check(), global_step=epoch)
            writer.add_figure('check process 1', plot_check(label=1), global_step=epoch)


print(best_ACC)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
