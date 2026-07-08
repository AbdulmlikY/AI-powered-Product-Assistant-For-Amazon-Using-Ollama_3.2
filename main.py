import ollama
import pandas as pd
import json

try:
    df = pd.read_csv('amazon_product_clean.csv')
except FileNotFoundError:
    raise FileNotFoundError("Path is not valid")

model = 'llama3.2'



def search_by_price(max_price):
    filtered_df = df[df['product_price'] <= max_price]
    return filtered_df

def search_top_rated():
    filtered_df = df.sort_values('product_star_rating', ascending=False).head(5)
    return filtered_df

def search_most_reviewed():
    filtered_df = df.sort_values('product_num_ratings', ascending=False).head(5)
    return filtered_df

def search_best_sellers():
    filtered_df = df[df['is_best_seller'] == True]
    return filtered_df


def classify_intent(user_input):
    prompt = f"""
Classify the user's request into one of these intents:

- search_by_price (needs a number: max_price)
- search_top_rated (no number needed)
- search_most_reviewed (no number needed)
- search_best_sellers (no number needed)

Respond with ONLY valid JSON.
Do not include explanations, markdown, or code fences.


Examples:
{{"intent":"search_by_price","max_price":80}}

{{"intent":"search_top_rated"}}

{{"intent":"search_most_reviewed"}}

{{"intent":"search_best_sellers"}}

User request:
{user_input}
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    return response.message.content


print("Welcome to the AI Chatbot!")
print('Type "exit" to quit.\n')

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'exit':
        print('Have a nice day!')
        break
    
    try:
        intent = json.loads(classify_intent(user_input))
    except json.JSONDecodeError:
        print("Sorry, I couldn't understand your request.")
        continue
    if "intent" not in intent:
        print("Sorry, I couldn't understand your request.")
        continue


    if intent["intent"] == "search_by_price":
        results = search_by_price(intent["max_price"])

    elif intent["intent"] == "search_top_rated":
        results = search_top_rated()

    elif intent["intent"] == "search_most_reviewed":
        results = search_most_reviewed()

    elif intent["intent"] == "search_best_sellers":
        results = search_best_sellers()

    else:
        print("Sorry, I can only search by price, top rated, most reviewed, or best sellers.")
        continue 

    
    if results.empty:
        print("No matching products found.")
        continue
    summary = results[["product_title","product_price","product_star_rating","product_num_ratings"]].head(5).to_string(index=False)

    prompt = f"""
    The user asked:

    {user_input}

    Here are the search results:

    {summary}

    Answer the user's question using ONLY these search results.
    Do not make up products that are not listed.
    Be friendly and concise.
    """
    response = ollama.chat(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ]
    )

    print(response.message.content)

