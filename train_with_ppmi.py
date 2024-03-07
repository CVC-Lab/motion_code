import joblib
import re
import os
import numpy as np
import pandas as pd
from datetime import datetime

from single_image_ppmi_dataloader import PPMIDataset
from .data_processing import process_ppmi_data_for_motion_codes
from .motion_code import MotionCode
import pdb

time_series = joblib.load('./notebooks/artifacts/time_series.pkl')

def train_test_split(x, split=0.2):
    sp_idx = int(len(x)*split)
    test_x, train_x = x[:sp_idx], x[sp_idx:]
    return train_x, test_x

def pad_zeros(x):
    maxlen = max([len(item) for item in x])
    new_x = np.zeros((len(x), maxlen))
    for i, item in enumerate(x):
        new_x[i, :len(item)] = item
    return new_x


time_series = pad_zeros(time_series)
# filter out rows for which labels do not exist in your dataset
df = pd.read_csv("./dataset/PPMI_Curated_Data_Cut_Public_20230612_rev.csv")
root_dir = '/mnt/data/ashwin/DTI/PPMI/'
dataset = PPMIDataset(root_dir=root_dir)

# we have filenames that can be used to match rows in the dataframe
# create 2 columns row_idx, PATNO, visit_time
data_extracted = []
for i, path in enumerate(dataset.scans):
    patno_match = re.search(r'/PPMI/(\d+)/', path)
    date_match = re.search(r'/(\d{4})-(\d{2})-(\d{2})_', path)
    if patno_match and date_match:
        patno = int(patno_match.group(1))
        date_str = f"{date_match.group(1)}-{date_match.group(2)}"
        # Convert to datetime to reformat
        date_dt = datetime.strptime(date_str, '%Y-%m')
        # Format date to match 'Feb-23' style
        formatted_date = date_dt.strftime('%b%Y')
        data_extracted.append({'PATNO': patno, 
                               'visit_date': formatted_date.upper(), 
                            'row_id': i})
        
# Create DataFrame from extracted data
df_paths = pd.DataFrame(data_extracted)
df_joined = pd.merge(df, df_paths, on=['PATNO', 'visit_date'], how='inner')
select_cols = [
'row_id', 
 'NHY']
df_select = df_joined[select_cols]
df_select = df_select[df_select.NHY != '.'] # this get rids of 13 examples
time_series_select = time_series[df_select['row_id']]
labels_select = df_select['NHY'].to_numpy(dtype=np.int64)

# join with above df
# extract columns with all metric labels
Y_train, Y_test = train_test_split(time_series_select)
labels_train, labels_test = train_test_split(labels_select)

X_train, Y_train, labels_train = process_ppmi_data_for_motion_codes(Y_train, labels_train)
X_test, Y_test, labels_test = process_ppmi_data_for_motion_codes(Y_test, labels_test)

model = MotionCode(m=20, Q=1, latent_dim=3, sigma_y=0.1)
# Then we train model on the given X_train, Y_train, label_train set and saved it to a file named test_model.
model_path = 'saved_models/' + 'test_model_ppmi_NHY'
os.makedirs(model_path, exist_ok=True)
model.fit(X_train, Y_train, labels_train, model_path)

model.load(model_path)
acc = model.classify_predict_on_batches(X_test_list=X_test, Y_test_list=Y_test, true_labels=labels_test)
print('Accurary:', acc)
