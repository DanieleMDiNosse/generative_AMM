from datetime import datetime
import os
import pandas as pd

def print_(obj):
    # Determine the filename based on the process ID and the formatted date
    filename = f'logs/log_{os.getpid()}.txt'
    
    # Convert the object to a string based on its type
    if isinstance(obj, pd.DataFrame):
        obj_str = obj.to_string()  # Use DataFrame's to_string method for better formatting
    else:
        obj_str = str(obj)  # Convert other objects to string using str()
    
    # Open the file in append mode, or 'w' for write mode (which overwrites the file)
    with open(filename, 'a') as file:
        # Print to terminal
        print(obj)
        # Write the same text to the file
        file.write(obj_str + '\n')  # Adding '\n' to move to the next line for each call