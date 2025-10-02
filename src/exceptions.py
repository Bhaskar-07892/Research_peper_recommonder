import sys
from src.utils.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in Python script: [{file_name}] at line [{line_number}] — Error message: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys = sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
