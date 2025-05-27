from pdf2image import convert_from_path
import os

# Define paths
pdf_dir = r"C:\Users\khura\Downloads\compare 1 (2)\compare 1"  # Folder with your sample PDFs
image_output_dir = r"C:\Users\khura\OneDrive\Documents\Agentic AI\1. Projects\3. AI Page Comparison\Ref_Files\Sample Images"
poppler_path = r"C:\Users\khura\OneDrive\Documents\Agentic AI\1. Projects\3. AI Page Comparison\Ref_Files\poppler-24.08.0\Library\bin"  # Adjust to your Poppler bin folder

# Create output directory if it doesn't exist
os.makedirs(image_output_dir, exist_ok=True)

# Verify PDF directory exists
if not os.path.exists(pdf_dir):
    print(f"Error: PDF directory '{pdf_dir}' does not exist.")
    exit(1)

# Process PDF files
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        try:
            # Convert PDF to images (300 DPI), specify poppler_path
            pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
            for i, page in enumerate(pages):
                image_filename = os.path.join(image_output_dir, f"{os.path.splitext(pdf_file)[0]}_page_{i+1}.png")
                page.save(image_filename, "PNG")
                print(f"Saved {image_filename}")
        except Exception as e:
            print(f"Error converting {pdf_file}: {e}")