from google import genai

client = genai.Client(api_key="AIzaSyDj2q7BW2X_LUNr8bqG-iJAiRWzkmNkgIQ")

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Explain how AI works in a few words"
)
print(response.text)