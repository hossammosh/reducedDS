import os
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# Global sample index
#sample_index = 0

FILENAME = 'log_data.xlsx'

# Initialize a new Excel workbook with header
def init_excel():
    wb = Workbook()
    ws = wb.active
    ws.title = "DataInfo"

    # Corrected header row without duplicates
    headers = [
        "Index", "Sample Index", "Seq Name",
        "Template Frame ID", "Template Frame Path",
        "Search Frame ID",
        "Seq ID", "Seq Path", "Class Name", "Vid ID", "Search Names", "Search Path"
    ]
    ws.append(headers)
    _format_cells(ws)
    return wb, ws

# Apply alignment and sizing
def _format_cells(ws):
    align = Alignment(horizontal='center', vertical='center', wrap_text=True)

    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = align

    for col in ws.columns:
        max_length = max(len(str(cell.value)) if cell.value else 0 for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max_length + 4

    for row in ws.iter_rows():
        ws.row_dimensions[row[0].row].height = 25

# Logging function
def log_data(data_info: dict):
    global sample_index

    # Create a fresh workbook each time (overwrite previous file)
    wb, ws = init_excel()

    start_row = 2
    end_row = start_row + 1

    # Merge columns vertically for rows 2 and 3 (adjusted columns)
    merge_columns = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]
    for col in merge_columns:
        ws.merge_cells(start_row=start_row, start_column=col, end_row=end_row, end_column=col)

    # Helper to safely convert lists to string (first element fallback)
    def safe_str_list(value, idx=0):
        if isinstance(value, list):
            if len(value) > idx:
                elem = value[idx]
                if isinstance(elem, list):
                    return ", ".join(map(str, elem))
                else:
                    return str(elem)
            else:
                return ""
        elif value is None:
            return ""
        else:
            return str(value)

    # Write row 1 values (merged and unmerged)
    ws.cell(row=start_row, column=1, value=1)
    ws.cell(row=start_row, column=2, value=sample_index)
    ws.cell(row=start_row, column=3, value=data_info.get("seq_name", ""))
    ws.cell(row=start_row, column=4, value=safe_str_list(data_info.get("template_ids"), 0))
    ws.cell(row=start_row, column=5, value=safe_str_list(data_info.get("template_path"), 0))
    ws.cell(row=start_row, column=6, value=safe_str_list(data_info.get("search_id")))
    ws.cell(row=start_row, column=7, value=data_info.get("seq_id", ""))
    ws.cell(row=start_row, column=8, value=data_info.get("seq_path", ""))
    ws.cell(row=start_row, column=9, value=data_info.get("class_name", ""))
    ws.cell(row=start_row, column=10, value=data_info.get("vid_id", ""))
    ws.cell(row=start_row, column=11, value=", ".join(map(str, data_info.get("search_names", []))))
    ws.cell(row=start_row, column=12, value=", ".join(map(str, data_info.get("search_path", []))))

    # Write row 2 values (template frame IDs and paths)
    ws.cell(row=end_row, column=4, value=safe_str_list(data_info.get("template_ids"), 1))
    ws.cell(row=end_row, column=5, value=safe_str_list(data_info.get("template_path"), 1))

    _format_cells(ws)

    # Save to file
    wb.save(FILENAME)
    sample_index += 1


# import os
# from openpyxl import Workbook
# from openpyxl.styles import Alignment
# from openpyxl.utils import get_column_letter
#
# # Global sample index
# sample_index = 0
#
# # Output Excel file name
# FILENAME = 'log_data.xlsx'
#
# # Initialize a new Excel workbook with header
# def init_excel():
#     wb = Workbook()
#     ws = wb.active
#     ws.title = "DataInfo"
#
#     # Header row
#     headers = [
#         "Index", "Sample Index", "Seq Name",
#         "Template Frame IDs", "Template Frame IDs",
#         "Template Frame Path", "Template Frame Path",
#         "Search Frame ID",
#         "Seq ID", "Seq Path", "Class Name", "Vid ID", "Search Names", "Search Path"
#     ]
#     ws.append(headers)
#     _format_cells(ws)
#     return wb, ws
#
# # Apply alignment and sizing
# def _format_cells(ws):
#     align = Alignment(horizontal='center', vertical='center', wrap_text=True)
#
#     for row in ws.iter_rows():
#         for cell in row:
#             cell.alignment = align
#
#     for col in ws.columns:
#         max_length = max(len(str(cell.value)) if cell.value else 0 for cell in col)
#         col_letter = get_column_letter(col[0].column)
#         ws.column_dimensions[col_letter].width = max_length + 4
#
#     for row in ws.iter_rows():
#         ws.row_dimensions[row[0].row].height = 25
#
# # Logging function
# def log_data(data_info: dict):
#     global sample_index
#
#     # Create a fresh workbook each time (overwrite previous file)
#     wb, ws = init_excel()
#
#     start_row = 2
#     end_row = start_row + 1
#
#     # Merge columns vertically for rows 2 and 3
#     merge_columns = [1, 2, 3, 8, 9, 10, 11, 12, 13, 14]
#     for col in merge_columns:
#         ws.merge_cells(start_row=start_row, start_column=col, end_row=end_row, end_column=col)
#
#     # Helper to safely convert lists to string (first element fallback)
#     def safe_str_list(value, idx=0):
#         if isinstance(value, list):
#             if len(value) > idx:
#                 elem = value[idx]
#                 if isinstance(elem, list):
#                     return ", ".join(map(str, elem))
#                 else:
#                     return str(elem)
#             else:
#                 return ""
#         elif value is None:
#             return ""
#         else:
#             return str(value)
#
#     # Write row 1 values (merged and unmerged)
#     ws.cell(row=start_row, column=1, value=1)
#     ws.cell(row=start_row, column=2, value=sample_index)
#     ws.cell(row=start_row, column=3, value=data_info.get("seq_name", ""))
#     ws.cell(row=start_row, column=4, value=safe_str_list(data_info.get("template_ids"), 0))
#     ws.cell(row=start_row, column=6, value=safe_str_list(data_info.get("template_path"), 0))
#     ws.cell(row=start_row, column=8, value=safe_str_list(data_info.get("search_id")))
#     ws.cell(row=start_row, column=9, value=data_info.get("seq_id", ""))
#     ws.cell(row=start_row, column=10, value=data_info.get("seq_path", ""))
#     ws.cell(row=start_row, column=11, value=data_info.get("class_name", ""))
#     ws.cell(row=start_row, column=12, value=data_info.get("vid_id", ""))
#     ws.cell(row=start_row, column=13, value=", ".join(map(str, data_info.get("search_names", []))))
#     ws.cell(row=start_row, column=14, value=", ".join(map(str, data_info.get("search_path", []))))
#
#     # Write row 2 values (template frame IDs and paths)
#     ws.cell(row=end_row, column=4, value=safe_str_list(data_info.get("template_ids"), 1))
#     ws.cell(row=end_row, column=6, value=safe_str_list(data_info.get("template_path"), 1))
#
#     _format_cells(ws)
#
#     # Save to file
#     wb.save(FILENAME)
#     sample_index += 1
#
