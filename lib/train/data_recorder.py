import os
import pandas as pd
from datetime import datetime
from openpyxl import load_workbook, Workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.dimensions import ColumnDimension
from openpyxl.styles import Alignment

# Primary key columns
PRIMARY_KEYS = ['seq_name', 'template_ids', 'search_id']

# Persistent DataFrame and Excel filename
_session_df = pd.DataFrame(columns=PRIMARY_KEYS)
_excel_file = None
_last_row_index = None

# Columns to merge across 2 rows
MERGE_COLUMNS = [
    'seq_name', 'search_id',
    'seq_id', 'seq_path', 'class_name', 'vid_id',
    'search_frame_names', 'search_frame_path'
]

# Columns to span 2 rows but remain unmerged
UNMERGED_TWO_ROW_COLUMNS = [
    'template_ids', 'template_frame_path'
]
def _init_log_file():
    global _excel_file
    if _excel_file is None:
        _excel_file = "log_data.xlsx"
    return _excel_file

def _adjust_column_widths(ws):
    for col in ws.columns:
        max_length = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = max_length + 2
        ws.column_dimensions[col_letter].width = adjusted_width

def _write_custom_excel(df, filename):
    wb = Workbook()
    ws = wb.active

    columns = list(df.columns)
    col_letters = {}


    col_index = 1
    for col in columns:
        col_letter = get_column_letter(col_index)
        col_letters[col] = col_letter

        if col in MERGE_COLUMNS:
            ws.merge_cells(start_row=1, start_column=col_index, end_row=2, end_column=col_index)
            cell = ws.cell(row=1, column=col_index, value=col)
            cell.alignment = Alignment(horizontal='center', vertical='center')
        elif col in UNMERGED_TWO_ROW_COLUMNS:
            ws.cell(row=1, column=col_index, value=col + "_1")
            ws.cell(row=2, column=col_index, value=col + "_2")
        else:
            ws.merge_cells(start_row=1, start_column=col_index, end_row=2, end_column=col_index)
            cell = ws.cell(row=1, column=col_index, value=col)
            cell.alignment = Alignment(horizontal='center', vertical='center')

        col_index += 1

    # Write data starting from row 3
    for row_idx, row in df.iterrows():
        for col_idx, col in enumerate(columns, start=1):
            value = row[col]
            ws.cell(row=row_idx + 3, column=col_idx, value=str(value))

    _adjust_column_widths(ws)
    wb.save(filename)
def log_data(data_info):
    global _session_df, _last_row_index
    excel_file = _init_log_file()

    if all(k in data_info for k in PRIMARY_KEYS):
        # Primary key info is provided — insert/update as normal
        key_values = {
            'seq_name': str(data_info['seq_name']),
            'template_ids': str(data_info['template_ids']),
            'search_id': str(data_info['search_id'])
        }

        # Find match
        match = (_session_df[PRIMARY_KEYS] == pd.Series(key_values)).all(axis=1) if not _session_df.empty else pd.Series([False])
        if match.any():
            idx = match.idxmax()
            _last_row_index = idx
            for k, v in data_info.items():
                if k not in _session_df.columns:
                    _session_df[k] = None
                _session_df.at[idx, k] = v
        else:
            new_row = {**key_values, **data_info}
            for k in new_row:
                if k not in _session_df.columns:
                    _session_df[k] = None
            _session_df = pd.concat([_session_df, pd.DataFrame([new_row])], ignore_index=True)
            _last_row_index = len(_session_df) - 1
    else:
        # No primary key info — update last known row
        if _last_row_index is None:
            raise ValueError("No previous primary key set to update.")
        for k, v in data_info.items():
            if k not in _session_df.columns:
                _session_df[k] = None
            _session_df.at[_last_row_index, k] = v

    _write_custom_excel(_session_df, excel_file)

# def log_data(data_info):
#     global _session_df
#     excel_file = _init_log_file()
#
#     # Ensure primary key values are strings
#     key_values = {
#         'seq_name': str(data_info['seq_name']),
#         'template_ids': str(data_info['template_ids']),
#         'search_id': str(data_info['search_id'])
#     }
#
#     # Search for existing row
#     match = (_session_df[PRIMARY_KEYS] == pd.Series(key_values)).all(axis=1) if not _session_df.empty else pd.Series([False])
#
#     if match.any():
#         idx = match.idxmax()
#         for k, v in data_info.items():
#             if k not in _session_df.columns:
#                 _session_df[k] = None
#             _session_df.at[idx, k] = v
#     else:
#         new_row = {**key_values, **data_info}
#         for k in new_row:
#             if k not in _session_df.columns:
#                 _session_df[k] = None
#         _session_df = pd.concat([_session_df, pd.DataFrame([new_row])], ignore_index=True)
#
#     # Save after every log call
#     _write_custom_excel(_session_df, excel_file)


def save_log():
    if _session_df.empty:
        print("No data to save.")
        return
    excel_file = _init_log_file()
    _write_custom_excel(_session_df, excel_file)
    print(f"Log saved to: {excel_file}")

