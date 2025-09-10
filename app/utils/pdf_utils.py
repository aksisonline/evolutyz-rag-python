"""
pdf_utils.py
Utility functions for PDF processing and page extraction.
"""

from typing import List
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
import io

class PDFUtils:
    """Utility class for PDF file operations."""

    @staticmethod
    def extract_pages(pdf_path: str) -> List[bytes]:
        """
        Extracts each page of a PDF as a separate bytes object.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            List[bytes]: List of PDF page bytes.
        """
        reader = PdfReader(pdf_path)
        pages = []
        for page_num in range(len(reader.pages)):
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])
            page_bytes = io.BytesIO()
            writer.write(page_bytes)
            pages.append(page_bytes.getvalue())
        return pages

    @staticmethod
    def extract_text_from_pages(pdf_path: str) -> List[str]:
        """
        Extracts text content from each page of a PDF.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            List[str]: List of text content from each page.
        """
        reader = PdfReader(pdf_path)
        pages_text = []
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            pages_text.append(text)
        return pages_text
