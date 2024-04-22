import json
import os
import glob
import pandas as pd
from os.path import exists
import subprocess
from datetime import date, datetime
import hashlib
import numpy as np
from sklearn.impute import SimpleImputer


folderpath = r'ml2.0/trained_data'

# Checks all files and imputes missing values
def impute_missing_values(filepath):
    df = pd.read_csv(filepath)
    if df[['EAR', 'MAR', 'PUC', 'MOE']].isna().values.any():
        columns_to_impute = ['EAR', 'MAR', 'PUC', 'MOE']
        sub_df = df[columns_to_impute]
        imputer = SimpleImputer(strategy='mean')
        imputed_subset_df = pd.DataFrame(imputer.fit_transform(sub_df), columns=sub_df.columns)
        df.update(imputed_subset_df)
        
        df.to_csv(filepath, index=False, header=True)
# Checks if file has a size of 5001 rows, and returns true if that is the case
def create_new_file(filepath):
    if exists(filepath):
        df = pd.read_csv(filepath)
        num_rows = df.shape[0]
        if num_rows > 5000:
            return True
        else:
            return False
    else:
        return True
# returns and hashes the linux device id for usage as participant number
def get_device_id():
    return_id = subprocess.Popen('cat /var/lib/dbus/machine-id',shell=True,stdout=subprocess.PIPE)
    output, _ = return_id.communicate()
    sha256_hash = hashlib.sha256(output).hexdigest()
    shortened_hash = int(sha256_hash[:10], 16)
    return shortened_hash
    
# Creates a filename after which time of day it is
def get_filename(id):
        currentdate = date.today()
        current_time = datetime.now()
        if current_time.hour > 6 and current_time.hour < 18:
            time_of_day = 'Day'
        else:
            time_of_day = 'Night'
        return f"ml2.0/trained_data/data-{id}-{time_of_day}"
    
# returns the last file that was created, used for keeping all files same rowsize
def getLastCSVFile():
    files = os.listdir(folderpath)
    csv_files = [file for file in files if file.endswith('.csv')]
    csv_files.sort()
    
    if csv_files:
        if len(csv_files) < 2:
            return csv_files[-1]
        return csv_files[-2]
    else:
        return None 

# returns the amount of csv files in the trained data folder for filename purposes
def amount_of_csv_files():
    csv_count = 0
    
    for file_name in os.listdir(folderpath):
        if file_name.endswith(r'.csv'):
            csv_count += 1
    return csv_count  

# Checks is file needs to be expaned or create a new file and saves it
def savefile(filepath, data, device_id):
    if os.path.exists(folderpath) == False:
        os.makedirs(folderpath, exist_ok=True)
        
    if exists(filepath):
        if create_new_file(f"ml2.0/trained_data/{getLastCSVFile()}"):
            filepath = f"{get_filename(device_id)}({amount_of_csv_files()}).csv"
            data.to_csv(filepath, mode='a', index=False, header=True) 
            return filepath
        else:
            if filepath != f"ml2.0/trained_data/{getLastCSVFile()}":
                filepath = f"ml2.0/trained_data/{getLastCSVFile()}"  
            data.to_csv(filepath, mode='a', index=False, header=False)    
            return filepath
    else:
        # appends new data and does add header values (if file doesn't exists)
        data.to_csv(filepath, mode='a', index=False, header=True)
        return filepath
    
# creates the data    
def createdata(Y,ear,mar,puc,moe,frame):
    # Changes 0 and 1 to become the more default values of 1 and 10
    # if Y == 1:
    #     Y = 10
        
    device_id = get_device_id()
    filepath = f"{get_filename(device_id)}.csv"
            
    # creates data 
    data = {
        'Y': [Y],
        'Participant': [device_id],
        'EAR': [ear],
        'MAR': [mar],
        'PUC': [puc],
        'MOE': [moe],
        'Frame': [frame]   
    }
    
    df = pd.DataFrame(data)
    filepath_to_impute = savefile(filepath, df, device_id)

    # Imputes NaN values  
    # if frame > 10:
    #     impute_missing_values(filepath_to_impute)



    ######## 
    # JSON #
    ########
     
    ## creates Json file from csv file (if json is needed)
    # filepath = f"{get_filename(part)}.json"
    
    # if not os.path.exists(filepath):
    #     open(f"{get_filename(part)}.json", 'w').close()

    # df = pd.read_csv(f"{get_filename(part)}.csv")
    # json_data = df.to_json(f"{get_filename(part)}.json", orient='records')


