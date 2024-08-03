from motion_code import MotionCode
from data_processing import load_data, process_data_for_motion_codes

# In this example, we use ItalyPowerDemand dataset. First we load dataset include Y-values of the series and their labels
name = 'ItalyPowerDemand'
Y_train, labels_train = load_data(name=name, split='train')

# Then we process the data for motion code model and generate X-variable, which is needed for training.
X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train, labels_train)

# Now we load the test set
Y_test, labels_test = load_data(name=name, split='test')
X_test, Y_test, labels_test = process_data_for_motion_codes(Y_test, labels_test)

print(X_train.shape, Y_train.shape, labels_train.shape)
print(X_test.shape, Y_test.shape, labels_test.shape)