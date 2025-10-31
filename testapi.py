import requests
url="http://127.0.0.1:8000/ask"
while True:
    query = input(" Enter your question (or type 'exit' to quit): ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        print("\nGoodbye!")
        break

    payload = {"query": query}

 #data you want to send to the API
response=requests.post(url,json=payload)
 # Convert server response to Python dictionary
result = response.json()
# 💡 Print neatly formatted output
print("\n===============================")
#Try to get the "answer" key from the dictionary.

#If it doesn’t exist (for example, if there was an error), print 'No answer found'.
print(f"Answer: {result.get('answer', 'No answer found')}")
print(f"Confidence: {result.get('confidence', 'N/A')}")
    
    # Print each source if available
sources = result.get("sources", [])
if sources:
        print("\nSources:")
        """ Loops through each dictionary inside the sources list.

For each source, it prints:

The file name ('source')

The similarity score ('score')"""
        for src in sources:
            print(f" {src.get('source', 'unknown')} (score: {src.get('score', 'N/A')})")
print("===============================\n")
"""requests.get("https://google.com")

/ask is the endpoint you created in FastAPI

You’re sending an HTTP POST request to your API using the requests library.

url → where to send it (http://127.0.0.1:8000/ask)
fetches Google’s webpage.
Send a request to my FastAPI app running locally on port 8000, at the /ask endpoint.”"""