import re
import glob
import numpy as np
from sequential_lstm import *

# get training X and y
X_and_y_dic = np.load("X_train_and_y_train.npz")
X_train = X_and_y_dic["X"]
y_train = X_and_y_dic["y"]

# get testing X and y
X_and_y_dic = np.load("X_test_and_y_test.npz")
X_test = X_and_y_dic["X"]
y_test = X_and_y_dic["y"]

model = Model()
model.define_model()
model.fit_model(X_train, y_train)

predictions = model.predict(X_test)

# if y_test is just the label numbers then leave from_one_hot as default False
accuracy = model.accuracy(y_test, from_one_hot=True)

print(f"Accuracy = {accuracy} or {accuracy*100} %")

# # gets the names if wanted
# import yaml
# with open('frames_data.yaml', 'r') as file:
#     yaml_dic = yaml.safe_load(file)
    
# categories = yaml_dic['names']
# y_test_names = [categories[label_num] for label_num in y_test_label_nums]
# predictions_names = [categories[label_num] for label_num in predictions]



model_list = glob.glob("./model_*")
if len(model_list) != 0:
    model_nums = [re.findall(r'\d+', file_name)[0] for file_name in model_list]
    new_model_num = max(map(int, model_nums)) + 1
else:
    new_model_num = 0
model.save_model(f"model_{new_model_num}.keras")


#model.load_model("model_0.keras")

