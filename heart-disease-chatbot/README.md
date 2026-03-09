# ❤️ Heart Disease Prediction — AI Project Chatbot

A beautiful Streamlit-powered AI chatbot that answers questions specifically about the
[Heart Disease Prediction](https://github.com/kanna-vamshi-krishna/heart-disease-prediction)
project by **Kanna Vamshi Krishna**.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
heart-disease-chatbot/
├── app.py              # Main Streamlit application
├── knowledge_base.py   # Project knowledge + AI system prompt
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🤖 How It Works

- **`knowledge_base.py`** holds all project-specific facts: dataset details, features,
  models, results, methodology, and how to run the original project.
- **`app.py`** is the Streamlit frontend. It sends user questions to the Anthropic Claude API
  with a strict system prompt that restricts the bot to project-only answers.
- If a user asks an off-topic question, the bot politely declines and redirects.

---

## 🧠 What the Chatbot Knows

| Topic          | Details                                                                              |
| -------------- | ------------------------------------------------------------------------------------ |
| Dataset        | heart.csv, 303 samples, 13 features, Kaggle source                                   |
| Features       | age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal |
| Models         | Logistic Regression, Decision Tree, Random Forest                                    |
| Best Model     | Logistic Regression with Optimal Threshold                                           |
| Metrics        | Precision 0.8788, Recall 0.9062, F1-Score 0.8923                                     |
| Visualizations | Confusion Matrix, ROC Curve                                                          |
| Tools          | Python, scikit-learn, pandas, numpy, matplotlib, seaborn, Jupyter                    |

---

## 🔑 API Key

This app uses the Anthropic Claude API. The API key is handled automatically
by the Claude.ai environment. If running locally outside Claude.ai, set your key:

```bash
# In app.py, update the headers in call_claude():
headers={
    "Content-Type": "application/json",
    "x-api-key": "YOUR_ANTHROPIC_API_KEY",
    "anthropic-version": "2023-06-01"
}
```

---

## ✨ Features

- 💬 Real-time AI chat powered by Claude
- 🎨 Beautiful dark UI with animated gradients
- 💡 Suggested question buttons for quick exploration
- 📊 Project stats displayed in the sidebar
- 🔒 Restricted to project-only answers
- 🗑️ Clear chat button

---

## Groq Router Workflow (`call_groq_router`)

```text
User Question
      │
      ▼
Create Payload
(model + system prompt + user message)
      │
      ▼
Send API Request
requests.post() → Groq API
      │
      ▼
Check HTTP Status
(resp.raise_for_status())
      │
      ▼
Extract Response Text
resp.json()["choices"][0]["message"]["content"]
      │
      ▼
Search for JSON in response
regex: { ... }
      │
 ┌────┴───────────────┐
 │                    │
JSON Found         No JSON Found
 │                    │
Parse JSON          Return Knowledge
json.loads()        {"type":"knowledge","answer":raw}
 │
 ▼
Return Routing Result
(data_query / knowledge / off_topic)
```

### Explanation

1. **User question arrives** and is sent to the Groq model.
2. A **payload dictionary** is created containing:
   - model name
   - system prompt
   - user message

3. The payload is sent using **Groq Chat Completion API**.
4. The response is checked using `raise_for_status()`.
5. The assistant's reply is extracted from the API response.
6. A **regex search** extracts JSON routing information.
7. The function returns a structured dictionary describing the query type:
   - `data_query`
   - `knowledge`
   - `off_topic`

## Result Interpretation Workflow (`call_groq_followup`)

```text
User Question + Data Result
           │
           ▼
Create Payload
(system prompt + question + dataset result)
           │
           ▼
Send API Request
requests.post() → Groq API
           │
           ▼
Check HTTP Status
(resp.raise_for_status())
           │
           ▼
Extract Assistant Response
resp.json()["choices"][0]["message"]["content"]
           │
           ▼
Return Plain-English Interpretation
(1-3 sentence explanation)
```

### Explanation

1. After executing a **pandas query on the dataset**, the result is obtained.
2. The system sends:
   - the **original question**
   - the **data result**

3. The Groq model is prompted to **interpret the result in simple language**.
4. The response is extracted and returned as a **human-friendly explanation**.

## Message Rendering Workflow

```
Message arrives
      │
      ▼
Check role
      │
 ┌────┴────┐
 │         │
User    Assistant
 │         │
Render   Check msg_type
bubble        │
        ┌─────┼───────────────┐
        │     │               │
   data_query  knowledge   off_topic   error
```

### Explanation

1. **Message arrives** – A message from the user or assistant is received.
2. **Check role** – The system checks whether the message is from the **user** or the **assistant**.
3. **User message** – Rendered as a **chat bubble** on the right side.
4. **Assistant message** – The system checks the **message type (`msg_type`)**:
   - **data_query** → Displays dataset results and optional pandas code.
   - **knowledge** → Shows a normal informational response.
   - **off_topic** → Responds that the assistant only answers project-related questions.
   - **error** → Displays an error message.

---

## 👤 Author

**Original Project:** Kanna Vamshi Krishna  
**Chatbot Layer:** Built with Streamlit + Anthropic Claude API
