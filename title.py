from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

def chat_title(user_message: str, assistant_response: str) -> str:
    query = f"User: {user_message}\nAssistant: {assistant_response}"
    prompt = f""""Create a brief, professional title (3-8 words) for this conversation. "
        "Focus on the user's main question or topic.
        Conversation: {query}\n\n"
        Title: """
    
    response = client.chat_completion(model="microsoft/Phi-3.5-mini-instruct",messages=[{"role": "user", "content":prompt}],max_tokens=12,stream=False,stop=["\n", ".", "!", "?"],temperature=0.2,top_p=0.9)
    
    title = response.choices[0].message.content.replace('"', '')
        
    return title