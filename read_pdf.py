
import pypdf

def extract_text_from_pdf(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

if __name__ == "__main__":
    pdf_path = r"c:\Users\debuf\Desktop\6002\6002_dele\CityU_6002_DeLELSTM\Attention_delelstm.pptx.pdf"
    content = extract_text_from_pdf(pdf_path)
    print(content)
