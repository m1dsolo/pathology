import os
from sklearn.model_selection import train_test_split
from yang.dl import read_label_file, set_seed, save_label_file

set_seed(1053532442)

label_path = '/home/yangxuan/dataset/camelyon16/'
file_names, labels = read_label_file(os.path.join(label_path, 'label.csv'))

file_names_train, file_names_test, labels_train, labels_test = train_test_split(file_names, labels, test_size=0.2)

print(len(file_names_train), len(file_names_test))

save_label_file(os.path.join(label_path, 'label_train.csv'), file_names_train, labels_train)
save_label_file(os.path.join(label_path, 'label_test.csv'), file_names_test, labels_test)
