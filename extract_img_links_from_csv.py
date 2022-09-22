import csv
import json

CSV_FILE = "./1k.csv"
OUTPUT_FILE = "./img_links.txt"

with open(CSV_FILE, 'r') as f:
    reader = csv.reader(f)
    _headers = next(reader)

    for idx, csv_data in enumerate(reader):
        try:
            img_link = json.loads(
                csv_data[-1])["raw_discord_data"]["image_uri"]
            with open(OUTPUT_FILE, 'a') as f:
                f.write(img_link + "\n")

        except Exception as e:
            print(f'error in line {idx + 1} :/')
            print(e)
