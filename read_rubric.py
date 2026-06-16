import pdfplumber
import glob

files = glob.glob(r'C:\Users\Yasar\Downloads\COMP430*.pdf')
output_lines = []

for f in files:
    with pdfplumber.open(f) as pdf:
        for i, page in enumerate(pdf.pages):
            output_lines.append(f'=== PAGE {i+1} ===')
            txt = page.extract_text()
            if txt:
                output_lines.append(txt)
            tables = page.extract_tables()
            for t in tables:
                output_lines.append('--- TABLE ---')
                for row in t:
                    output_lines.append(str(row))

with open('rubric_text.txt', 'w', encoding='utf-8') as fout:
    fout.write('\n'.join(output_lines))

print('Saved rubric_text.txt with', len(output_lines), 'lines')
