# Build Your Own AI Clone: Gherman Sparrow Edition

This project demonstrates how to build a personalized AI chatbot using Retrieval-Augmented Generation (RAG). The chatbot takes on the persona of Gherman Sparrow from the novel *Lord of the Mysteries* and answers questions based *only* on a provided knowledge base about the character.

The application is built with a Streamlit front-end for user interaction and utilizes Arize Phoenix for real-time tracing and observability of the RAG pipeline.


*(To add your own screenshot: run the app, take a picture, upload to a site like [Imgur](https://imgur.com/upload), get the "Direct Link", and replace the URL above.)*

## 🚀 Features

-   **Interactive Chat Interface**: A simple and clean UI built with Streamlit.
-   **Persona-Driven Responses**: Uses advanced prompt engineering to make the LLM adopt the specific persona of Gherman Sparrow.
-   **Grounded Knowledge**: The AI's knowledge is strictly limited to the provided context, preventing it from using general knowledge and reducing hallucinations.
-   **High-Speed Inference**: Leverages the Groq API with Llama 3 for extremely fast responses.
-   **Full Observability**: Every query is traced using Arize Phoenix, allowing you to inspect the entire RAG chain: the retrieved documents, the final prompt, and the LLM's output.

## 🛠️ Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/rajathnh/Build-Your-Own-AI-Clone_HiDevs.git
cd Build-Your-Own-AI-Clone_HiDevs
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

-   **Windows:**

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

-   **macOS / Linux:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies

All required packages are listed in `requirements.txt`. Install them with a single command:

```bash
pip install -r requirements.txt
```
### 4. Configure API Keys and Secrets (Crucial Step)

This project requires API keys to function. We will store them securely using Streamlit's secrets management.

1.  **Create the secrets folder:** In the root of the project folder, create a new folder named `.streamlit`.

    ```bash
    mkdir .streamlit
    ```

2.  **Create the secrets file:** Inside the `.streamlit` folder, create a new file named `secrets.toml`.

3.  **Add your keys to `secrets.toml`:** Open the file and paste the following content, replacing the placeholder values with your actual keys.

    ```toml
    # .streamlit/secrets.toml

    GROQ_API_KEY = "gsk_YourGroqApiKeyHere"
    ARIZE_API_KEY = "your_arize_api_key_here"
    ARIZE_SPACE_KEY = "your_arize_space_key_here"
    ```
    -   Get your **Groq API Key** from the [Groq Console](https://console.groq.com/keys).
    -   Get your **Arize API & Space Keys** from your [Arize Account Settings](https://app.arize.com/account/settings).
    ```

## ▶️ How to Run the Application

Once the setup is complete, you can run the app with a single command:

```bash
streamlit run app.py
```
Your web browser will automatically open with the chatbot interface.
To view the traces and evaluate the RAG pipeline:
Check the terminal where you ran the app. You will see a URL for the Arize Phoenix UI (e.g., http://127.0.0.1:6006).
Open this URL in a new browser tab to see a detailed breakdown of each query.
