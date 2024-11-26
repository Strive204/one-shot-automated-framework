import openai
import os
import csv
import re
import pdfplumber

api_key = "sk-50O77fa1edac53c9f4a30040d52704ed83c7070561dzADcn"
api_base = "https://api.gptsapi.net/v1"

openai.api_key = api_key
openai.api_base = api_base

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def extract_results_and_discussion(text):
    pattern = r"(Results and discussion)(.*?)(?:\n[A-Z][^\n]*|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(2).strip()
    else:
        print("No 'Results and discussion' section found.")
        return ""

def extract_experiment_section(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Please extract the key points from the 'Results and discussion' section:\n{text}"}
            ],
            max_tokens=1500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error extracting experiment section: {e}")
        return ""

pdf_path = "1.pdf"

document_text = extract_text_from_pdf(pdf_path)

results_discussion_text = extract_results_and_discussion(document_text)

if results_discussion_text:
    experiment_summary = extract_experiment_section(results_discussion_text)
    print("Extracted experiment summary:")
    print(experiment_summary)
else:
    print("Failed to extract 'Results and discussion' section content.")
