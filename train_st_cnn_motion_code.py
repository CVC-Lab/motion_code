
from single_image_ppmi_dataloader import PPMIDataset
from .data_processing import process_ppmi_data_for_motion_codes
from .motion_code import MotionCode
import joblib
import os

ds = joblib.load('artifacts/timeseries_ds.pkl')
X_train, Y_train, labels_train = process_ppmi_data_for_motion_codes(
    ds['train_X'], ds['train_labels'])
X_test, Y_test, labels_test = process_ppmi_data_for_motion_codes(
    ds['test_X'], ds['test_labels'])

model = MotionCode(m=20, Q=1, latent_dim=3, sigma_y=0.1)
# Then we train model on the given X_train, Y_train, label_train set and saved it to a file named test_model.
model_path = 'saved_models/' + 'test_model_ppmi_NHY_st_cnn'
os.makedirs(model_path, exist_ok=True)
model.fit(X_train, Y_train, labels_train, model_path)

model.load(model_path)
acc = model.classify_predict_on_batches(X_test_list=X_test, Y_test_list=Y_test, true_labels=labels_test)
print('Accurary:', acc)