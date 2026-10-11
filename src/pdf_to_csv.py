import re
import csv
from PyPDF2 import PdfReader

PDF_FILE = "url-list.pdf"
OUTPUT_CSV = "video_metadata.csv"

reader = PdfReader(PDF_FILE)

all_text = ""

# Extract all text
for page in reader.pages:
    all_text += page.extract_text() + "\n"

# Extract URLs
url_pattern = r"http[s]?://[^\s]+"
urls = re.findall(url_pattern, all_text)

# Extract time ranges like 00:18 - 00:24
time_pattern = r"\d{2}:\d{2}\s*-\s*\d{2}:\d{2}"
time_ranges = re.findall(time_pattern, all_text)

print("Total URLs found:", len(urls))
print("Total time ranges found:", len(time_ranges))

# Convert mm:ss to 00:mm:ss format
def convert_time(t):
    return "00:" + t.strip()

# Write CSV
with open(OUTPUT_CSV, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "url", "start", "end"])

    for idx, (url, time_range) in enumerate(zip(urls, time_ranges)):
        start, end = time_range.split("-")
        start = convert_time(start)
        end = convert_time(end)

        filename = f"video{idx+1}.mp4"
        writer.writerow([filename, url, start, end])

print("CSV file created successfully.")