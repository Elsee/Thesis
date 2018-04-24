import numpy as np;
import gzip;
from io import StringIO;
import os;

all_users_uuid = os.listdir('./ExtraSensory.per_uuid_original_labels');

for (i,item) in enumerate(all_users_uuid):
    uuid = all_users_uuid[i].split('.');
    all_users_uuid[i] = uuid[0];
    
watch_acc_path = '../watch_accelerometer/ExtraSensory.raw_measurements.watch_acc/watch_acc';
    
smartwatch_users_uuid = os.listdir(watch_acc_path);

for (i,item) in enumerate(smartwatch_users_uuid):
    uuid = smartwatch_users_uuid[i].split('.');
    smartwatch_users_uuid[i] = uuid[0];
    
for uuid in smartwatch_users_uuid:

    user_data_file = './ExtraSensory.per_uuid_original_labels/' + uuid + '.original_labels.csv.gz';
    
    user_timestamps = os.listdir(watch_acc_path + '/' + uuid);
    
    for (i,item) in enumerate(user_timestamps):
        timestamp = user_timestamps[i].split('.');
        user_timestamps[i] = timestamp[0];
    
    # Read the entire csv file of the user:
    with gzip.open(user_data_file,'rt') as fid:
        csv_str = fid.read();
        pass;
        
    headline = csv_str[:csv_str.index('\n').encode()];
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
    
    from_watch_ind = np.where(full_table[:,117] == 6);
    from_watch = np.zeros((len(from_watch_ind[0]), full_table.shape[1]));
    
    for (i,idx) in enumerate(from_watch_ind[0]):
        from_watch[i,:] = full_table[idx, :];
    
    walk = np.where(full_table[:,5] > 0);
    run = np.where(full_table[:,6] > 0);
    walkUpstairs = np.where(full_table[:,58] > 0);
    walkDownstairs = np.where(full_table[:,59] > 0);
    
    walkCounter = 0;
    runCounter = 0;
    walkUpstairsCounter = 0;
    walkDownstairsCounter = 0;
    
    for (i,idx) in enumerate(walk[0]):
        if str(int(full_table[idx][0])) in user_timestamps:
            walkCounter = walkCounter+1;
            
    for (i,idx) in enumerate(run[0]):
        if str(int(full_table[idx][0])) in user_timestamps:
            runCounter = runCounter+1;
            
    for (i,idx) in enumerate(walkUpstairs[0]):
        if str(int(full_table[idx][0])) in user_timestamps:
            walkUpstairsCounter = walkUpstairsCounter+1;
    
    for (i,idx) in enumerate(walkDownstairs[0]):
        if str(int(full_table[idx][0])) in user_timestamps:
            walkDownstairsCounter = walkDownstairsCounter+1;
    
    with open("results.txt", "a") as myfile:
                myfile.write("User: " + str(uuid) + "\nWalking examples: " + str(walkCounter) + "\nRunning examples: " + str(runCounter) + 
                               "\nWalking up-stairs examples: " + str(walkUpstairsCounter) +
                               "\nWalking down-stairs examples: " + str(walkDownstairsCounter) + "\n\n\n")

