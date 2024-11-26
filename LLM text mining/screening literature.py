import requests
import json
import openai
import csv
import time


def get_wos_articles(api_key, keyword, year_range, article_type):
    base_url = "https://api.clarivate.com/apis/wos-starter/v1/documents"
    headers = {
        "X-ApiKey": api_key,
        "Accept": "application/json"
    }

    query = f"TI=({keyword}) AND PY=({year_range}) AND DT=({article_type})"
    articles = []
    page = 1
    max_records = 50

    while True:
        params = {
            "q": query,
            "limit": max_records,
            "page": page,
            "db": "WOS",
            "sortField": "LD+D"
        }

        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            records = data.get('hits', [])
            if not records:
                break

            for record in records:
                title = record.get('title', 'No Title')
                abstract = record.get('abstract', 'No Abstract')
                articles.append({"title": title, "abstract": abstract})

            page += 1
            time.sleep(1)  # Limit requests to 1 per second
        elif response.status_code == 403:
            print("Error 403: Forbidden. You may not have permission to access this resource.")
            break
        elif response.status_code == 429:
            print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break

    return articles


def summarize_abstracts(gpt_api_key, abstracts):
    openai.api_key = gpt_api_key
    summaries = []

    for abstract in abstracts:
        while True:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini4",
                messages=[{"role": "user", "content": f"Summarize the following abstract: {abstract}"}],
                max_tokens=100
            )
            if response.status_code == 429:
                print("Rate limit exceeded for GPT API. Waiting for 60 seconds before retrying...")
                time.sleep(60)
            else:
                break

        summary = response.choices[0].text.strip()
        summaries.append(summary)

    return summaries


def save_to_csv(articles, summaries, filename="articles.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Abstract", "Summary"])
        for article, summary in zip(articles, summaries):
            writer.writerow([article['title'], article['abstract'], summary])


def main():
    wos_api_key = "56cb132811898b8b57dfbe4dafd7719531bdc6af"
    gpt_api_key = "sk-50O77fa1edac53c9f4a30040d52704ed83c7070561dzADcn"

    keyword = "Carbon capture"
    year_range = "2000-2024"
    article_type = "Article"

    articles = get_wos_articles(wos_api_key, keyword, year_range, article_type)

    if articles:
        for i, article in enumerate(articles):
            print(f"\nArticle {i + 1}:\nTitle: {article['title']}\nAbstract: {article['abstract']}")

        abstracts = [article['abstract'] for article in articles]
        summaries = summarize_abstracts(gpt_api_key, abstracts)

        for i, summary in enumerate(summaries):
            print(f"\nSummary of Article {i + 1}: {summary}")

        save_to_csv(articles, summaries)
        print("\nArticles have been saved to articles.csv")
    else:
        print("No articles found.")


if __name__ == "__main__":
    main()
