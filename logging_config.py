# logging_config.py

import os
from openpyxl import Workbook

class ExcelLogger:
    def __init__(self, filename='log.xlsx'):
        self.filename = filename
        self.latest_sample = None
        self.latest_path = None
        self.data_by_sample = {}

        if os.path.exists(self.filename):
            os.remove(self.filename)

    def log(self, data: dict):
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        sample_name = data.get('sample_image', self.latest_sample)
        path = data.get('path', self.latest_path)

        if sample_name is None or path is None:
            raise ValueError("Must provide at least 'sample_image' and 'path' initially")

        self.latest_sample = sample_name
        self.latest_path = path

        # Create new entry if not exists
        if sample_name not in self.data_by_sample:
            self.data_by_sample[sample_name] = {'sample name': sample_name, 'path': path}

        # Update the existing row
        for k, v in data.items():
            if k == 'sample_image':
                self.data_by_sample[sample_name]['sample name'] = v
            elif k == 'path':
                self.data_by_sample[sample_name]['path'] = v
            else:
                self.data_by_sample[sample_name][k] = v

    def save(self):
        all_keys = ['sample name', 'path']
        for entry in self.data_by_sample.values():
            for k in entry.keys():
                if k not in all_keys:
                    all_keys.append(k)

        wb = Workbook()
        ws = wb.active
        ws.title = 'Logs'

        # Write header
        ws.append(all_keys)

        # Write rows sorted by sample name (optional)
        for sample_name in sorted(self.data_by_sample):
            row = self.data_by_sample[sample_name]
            row_data = [row.get(k, '') for k in all_keys]
            ws.append(row_data)

        wb.save(self.filename)
