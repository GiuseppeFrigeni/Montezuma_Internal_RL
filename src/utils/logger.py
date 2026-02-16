import csv
import os
import time

class CSVLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # If file exists, assume header is already written.
        self.file_exists = os.path.isfile(log_file)
        self.headers = None

    def log(self, metrics, step=None):
        if self.headers is None:
            self.headers = ['timestamp', 'step'] + list(metrics.keys())
            if not self.file_exists:
                with open(self.log_file, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.headers)
        
        with open(self.log_file, 'a') as f:
            writer = csv.writer(f)
            row = [time.time(), step] + list(metrics.values())
            writer.writerow(row)
