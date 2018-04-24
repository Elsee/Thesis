import numpy as np;
import gzip;
from io import StringIO;
import os;

all_users_uuid = os.listdir('../original labels/ExtraSensory.per_uuid_original_labels');

for (i,item) in enumerate(all_users_uuid):
    uuid = all_users_uuid[i].split('.');
    all_users_uuid[i] = uuid[0];
    
watch_acc_path = '../watch_accelerometer/ExtraSensory.raw_measurements.watch_acc/watch_acc';
    
smartwatch_users_uuid = os.listdir(watch_acc_path);

for (i,item) in enumerate(smartwatch_users_uuid):
    uuid = smartwatch_users_uuid[i].split('.');
    smartwatch_users_uuid[i] = uuid[0];
    
userNum = 1
    
for uuid in smartwatch_users_uuid:
#    completeActivityArray = np.empty[1,3]

    user_data_file = '../original labels/ExtraSensory.per_uuid_original_labels/' + uuid + '.original_labels.csv.gz';
    
    user_readings = os.listdir(watch_acc_path + '/' + uuid);
    user_timestamps = np.zeros(len(user_readings), dtype=np.int64)
    
    for (i,item) in enumerate(user_readings):
        timestamp = user_readings[i].split('.');
        user_timestamps[i] = timestamp[0];
    
    # Read the entire csv file of the user:
    with gzip.open(user_data_file,'rt') as fid:
        csv_str = fid.read();
        pass;
        
    headline = csv_str[:csv_str.index('\n')];
    columns = headline.split(',');
    
    first_label_ind = 1;
    
    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci;
            break;
        pass;
    
    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind];
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1];
    
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('original_label:');
        label_names[li] = label.replace('original_label:','');
        pass;
    
        
    full_table = np.loadtxt(StringIO(csv_str),delimiter=',',skiprows=1);
    
    walk = np.where(full_table[:,5] > 0);
    run = np.where(full_table[:,6] > 0);
    walkUpstairs = np.where(full_table[:,58] > 0);
    walkDownstairs = np.where(full_table[:,59] > 0);
    
    walkCounter = 0;
    runCounter = 0;
    walkUpstairsCounter = 0;
    walkDownstairsCounter = 0;
    
    flag = 0
    
    for (i,idx) in enumerate(walkDownstairs[0]):
        
        cur_timestamp = int(full_table[idx][0])
        
        if cur_timestamp in user_timestamps:
            temp_walk_array = np.loadtxt('../watch_accelerometer/ExtraSensory.raw_measurements.watch_acc/watch_acc/'+ uuid +'/' + str(cur_timestamp) + '.m_watch_acc.dat')
            if (temp_walk_array.shape[1] == 4):
                temp_walk_array = temp_walk_array[:,np.array([False, True, True, True])]
            
            if (i == 0):
                completeActivityArray = np.copy(temp_walk_array)
                flag = 1
            
            else:
                completeActivityArray = np.concatenate((completeActivityArray, temp_walk_array), axis=0)
                flag = 1

    if(flag):
        np.savetxt('./concatenatedData/walkingDownstairs/total_walkingDownstairs' + '#' + str(userNum) + '.csv', completeActivityArray, delimiter=",")
            
                        
    userNum = userNum + 1