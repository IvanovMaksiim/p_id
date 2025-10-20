from pdf2image import convert_from_path
import os
"""
Конвертация PDF в PNG-изображения (по одной странице на файл).
Используется poppler для рендеринга страниц.
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, '..', 'data', 'pdf_pid', 'Итог1.pdf')

output_folder = os.path.join(BASE_DIR, '..', 'data', 'pages')
os.makedirs(output_folder, exist_ok=True)

poppler_path = r'C:\Program Files\poppler-25.07.0\poppler-25.07.0\Library\bin'

pages = convert_from_path(pdf_path,
                          dpi=300,
                          fmt="png",
                          output_folder=output_folder,
                          poppler_path=poppler_path)

print(f"Сохранено {len(pages)} страниц в папку 'pages'")