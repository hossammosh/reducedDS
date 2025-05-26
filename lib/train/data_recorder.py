import os
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# Output Excel file name
FILENAME = 'log_data.xlsx'

# Global variable to track if file has been initialized
_file_initialized = False


# Initialize a new Excel workbook with header (called once at the beginning)
def init_excel():
    global _file_initialized

    # Check if file already exists, if so, don't recreate it
    if os.path.exists(FILENAME):
        _file_initialized = True
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "DataInfo"

    # Updated header row with specific headers 'stats/Loss/total' and 'stats/IoU'
    headers = [
        "Index", "Sample Index", "stats/Loss_total", "stats_IoU", "Seq Name",
        "Template Frame ID", "Template Frame Path",
        "Search Frame ID",
        "Seq ID", "Seq Path", "Class Name", "Vid ID", "Search Names", "Search Path"
    ]
    ws.append(headers)
    _format_header_cells(ws)

    # Save the initial file
    wb.save(FILENAME)
    _file_initialized = True
    #print(f"Excel file '{FILENAME}' created with headers.")


# Apply alignment and sizing to header row only
def _format_header_cells(ws):
    align = Alignment(horizontal='center', vertical='center', wrap_text=True)

    # Format only the header row
    for cell in ws[1]:
        cell.alignment = align

    # Set column widths based on header content
    for col in ws.columns:
        max_length = 0
        for cell in col:
            try:
                if cell.value is not None:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max_length + 4

    # Set row height for header
    ws.row_dimensions[1].height = 25


# Apply formatting to newly added rows
def _format_new_rows(ws, start_row, end_row):
    align = Alignment(horizontal='center', vertical='center', wrap_text=True)

    # Format the new rows
    for row_num in range(start_row, end_row + 1):
        for cell in ws[row_num]:
            cell.alignment = align
        ws.row_dimensions[row_num].height = 25


# Get the next available row number
def _get_next_row(ws):
    return ws.max_row + 1


# Logging function - Now appends data instead of overwriting
def log_data(sample_index: int, data_info: dict, stats: dict):
    global _file_initialized

    # Initialize file if not already done
    if not _file_initialized:
        init_excel()

    # Load existing workbook
    wb = load_workbook(FILENAME)
    ws = wb.active

    # Get the next available row
    next_row = _get_next_row(ws)
    start_row = next_row
    end_row = next_row + 1

    # Calculate the actual index (row number - 1 for header)
    data_index = next_row - 1

    # Merge columns vertically for the two new rows
    merge_columns = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
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

    # Extract specific stats, handle missing keys
    loss_total = stats.get("Loss/total", None)
    iou = stats.get("IoU", None)

    # Write row 1 values (merged and unmerged)
    ws.cell(row=start_row, column=1, value=data_index)
    ws.cell(row=start_row, column=2, value=sample_index)
    ws.cell(row=start_row, column=3, value=loss_total)
    ws.cell(row=start_row, column=4, value=iou)
    ws.cell(row=start_row, column=5, value=data_info.get("seq_name", ""))
    ws.cell(row=start_row, column=6, value=safe_str_list(data_info.get("template_ids"), 0))
    ws.cell(row=start_row, column=7, value=safe_str_list(data_info.get("template_path"), 0))
    ws.cell(row=start_row, column=8, value=safe_str_list(data_info.get("search_id")))
    ws.cell(row=start_row, column=9, value=data_info.get("seq_id", ""))
    ws.cell(row=start_row, column=10, value=data_info.get("seq_path", ""))
    ws.cell(row=start_row, column=11, value=data_info.get("class_name", ""))
    ws.cell(row=start_row, column=12, value=data_info.get("vid_id", ""))
    ws.cell(row=start_row, column=13, value=", ".join(map(str, data_info.get("search_names", []))))
    ws.cell(row=start_row, column=14, value=", ".join(map(str, data_info.get("search_path", []))))

    # Write row 2 values (template frame IDs and paths)
    ws.cell(row=end_row, column=6, value=safe_str_list(data_info.get("template_ids"), 1))
    ws.cell(row=end_row, column=7, value=safe_str_list(data_info.get("template_path"), 1))

    # Format the newly added rows
    _format_new_rows(ws, start_row, end_row)

    # Save to file
    wb.save(FILENAME)
    #print(f"Data appended to row {start_row}-{end_row} in '{FILENAME}'")


# Optional: Function to manually initialize the Excel file at the start of training
def initialize_training_log():
    """Call this function at the beginning of your training to ensure the Excel file is created."""
    init_excel()


# Optional: Function to reset the log file (delete and recreate)
def reset_log():
    """Delete the existing log file and reinitialize."""
    global _file_initialized
    if os.path.exists(FILENAME):
        os.remove(FILENAME)
    _file_initialized = False
    init_excel()

# import os
# from openpyxl import Workbook
# from openpyxl.styles import Alignment
# from openpyxl.utils import get_column_letter
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
#     # Updated header row with specific headers 'stats/Loss/total' and 'stats/IoU'
#     headers = [
#         "Index", "Sample Index", "stats/Loss_total", "stats_IoU", "Seq Name",
#         "Template Frame ID", "Template Frame Path",
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
#         max_length = 0
#         for cell in col:
#             try:
#                 if cell.value is not None:
#                     max_length = max(max_length, len(str(cell.value)))
#             except: # Handle potential errors if cell value is not string convertible
#                 pass
#         col_letter = get_column_letter(col[0].column)
#         ws.column_dimensions[col_letter].width = max_length + 4
#
#     for row in ws.iter_rows():
#         ws.row_dimensions[row[0].row].height = 25
#
# # Logging function - Updated for specific stats columns and headers
# def log_data(sample_index: int, data_info: dict, stats: dict):
#     # Create a fresh workbook each time (overwrite previous file)
#     # Consider loading existing workbook if appending is desired, but current logic overwrites.
#     wb, ws = init_excel()
#
#     start_row = 2
#     end_row = start_row + 1
#
#     # Merge columns vertically for rows 2 and 3 (adjusted columns due to insertion)
#     # Columns: Index(1), Sample Index(2), Loss(3), IoU(4), Seq Name(5), Tpl ID(6), Tpl Path(7), Sch ID(8), Seq ID(9), ...
#     merge_columns = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
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
#     # Extract specific stats, handle missing keys
#     # Use the exact keys provided in the stats dictionary
#     loss_total = stats.get("Loss/total", None)
#     iou = stats.get("IoU", None)
#
#     # Write row 1 values (merged and unmerged) - Adjusted column indices
#     ws.cell(row=start_row, column=1, value=1) # Index (e.g., row number in this log file)
#     ws.cell(row=start_row, column=2, value=sample_index) # Use passed sample_index
#     ws.cell(row=start_row, column=3, value=loss_total) # Insert Loss/total value directly
#     ws.cell(row=start_row, column=4, value=iou) # Insert IoU value directly
#     ws.cell(row=start_row, column=5, value=data_info.get("seq_name", "")) # Shifted to col 5
#     ws.cell(row=start_row, column=6, value=safe_str_list(data_info.get("template_ids"), 0)) # Shifted to col 6
#     ws.cell(row=start_row, column=7, value=safe_str_list(data_info.get("template_path"), 0)) # Shifted to col 7
#     ws.cell(row=start_row, column=8, value=safe_str_list(data_info.get("search_id"))) # Shifted to col 8
#     ws.cell(row=start_row, column=9, value=data_info.get("seq_id", "")) # Shifted to col 9
#     ws.cell(row=start_row, column=10, value=data_info.get("seq_path", "")) # Shifted to col 10
#     ws.cell(row=start_row, column=11, value=data_info.get("class_name", "")) # Shifted to col 11
#     ws.cell(row=start_row, column=12, value=data_info.get("vid_id", "")) # Shifted to col 12
#     ws.cell(row=start_row, column=13, value=", ".join(map(str, data_info.get("search_names", [])))) # Shifted to col 13
#     ws.cell(row=start_row, column=14, value=", ".join(map(str, data_info.get("search_path", [])))) # Shifted to col 14
#
#     # Write row 2 values (template frame IDs and paths) - Adjusted column indices
#     ws.cell(row=end_row, column=6, value=safe_str_list(data_info.get("template_ids"), 1)) # Shifted to col 6
#     ws.cell(row=end_row, column=7, value=safe_str_list(data_info.get("template_path"), 1)) # Shifted to col 7
#
#     _format_cells(ws)
#
#     # Save to file
#     wb.save(FILENAME)
#
#
#
