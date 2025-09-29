def ask_about_data(question, df):
    import openai
    openai.api_key = 'your-openai-api-key'

    context = df.describe().to_string()
    prompt = f"""
    You are a data assistant. Here's the data summary:

    {context}

    Question: {question}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
