import os
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# Global variables
_current_epoch = 1  # Default to epoch 1 if not specified
_initialized_epochs = set()  # Track which epochs have been initialized


# Function to get the filename for the current epoch
def get_filename(epoch=None):
    """Get the filename for the specified epoch or current epoch if not specified."""
    epoch_num = epoch if epoch is not None else _current_epoch
    return f'samples_log_epoch_{epoch_num}.xlsx'


# Set the current epoch number
def set_epoch(epoch_num):
    """Set the current epoch number for file naming."""
    global _current_epoch
    _current_epoch = epoch_num
    # Note: We don't reset initialization status here anymore


# Check if an epoch has been initialized
def is_epoch_initialized(epoch=None):
    """Check if the specified epoch or current epoch has been initialized."""
    epoch_to_check = epoch if epoch is not None else _current_epoch
    return epoch_to_check in _initialized_epochs


# Initialize a new Excel workbook with header (called once at the beginning of each epoch)
def init_excel(epoch=None):
    global _initialized_epochs

    # Use specified epoch or current epoch
    epoch_to_use = epoch if epoch is not None else _current_epoch

    # Get the filename for the specified epoch
    filename = get_filename(epoch_to_use)

    # Check if file already exists, if so, mark as initialized and return
    if os.path.exists(filename):
        _initialized_epochs.add(epoch_to_use)
        return

    # Create new workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "DataInfo"

    # Updated header row with specific headers 'stats/Loss/total' and 'stats/IoU'
    headers = [
        "Index", "Sample Index", "stats/Loss_total", "stats_IoU", "Seq Name",
        "Template Frame ID", "Template Frame Path", "Search Frame ID",
        "Seq ID", "Seq Path", "Class Name", "Vid ID", "Search Names", "Search Path"
    ]
    ws.append(headers)
    _format_header_cells(ws)

    # Save the initial file
    wb.save(filename)

    # Mark this epoch as initialized
    _initialized_epochs.add(epoch_to_use)
    # print(f"Excel file '{filename}' created with headers.")


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
def log_data(sample_index: int, data_info: dict, stats: dict, epoch=None):
    # Use specified epoch or current epoch
    epoch_to_use = epoch if epoch is not None else _current_epoch

    # Get the filename for the specified epoch
    filename = get_filename(epoch_to_use)

    # Initialize file if this epoch hasn't been initialized yet
    if not is_epoch_initialized(epoch_to_use):
        init_excel(epoch_to_use)

    # Load existing workbook
    wb = load_workbook(filename)
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
    wb.save(filename)
    # print(f"Data appended to row {start_row}-{end_row} in '{filename}'")


# Optional: Function to manually initialize the Excel file at the start of training
def initialize_training_log(epoch=1):
    """Call this function at the beginning of your training to ensure the Excel file is created."""
    set_epoch(epoch)
    init_excel()


# Optional: Function to reset the log file for a specific epoch (delete and recreate)
def reset_log(epoch=None):
    """Delete the existing log file and reinitialize."""
    global _initialized_epochs

    # Use specified epoch or current epoch
    epoch_to_use = epoch if epoch is not None else _current_epoch
    filename = get_filename(epoch_to_use)

    if os.path.exists(filename):
        os.remove(filename)

    # Remove from initialized epochs set
    if epoch_to_use in _initialized_epochs:
        _initialized_epochs.remove(epoch_to_use)

    init_excel(epoch_to_use)
