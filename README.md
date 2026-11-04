# 🎬 AI Video Assistant

An AI-powered application that transforms videos into structured knowledge. It automatically transcribes audio, generates concise summaries, extracts key decisions and action items, and allows users to interact with video content through an intelligent Q&A system.

## 🚀 Features

- **Video & Audio Input:** Upload local video/audio files or provide a video URL.
- **Automatic Transcription:** Convert video audio into text using Whisper.
- **AI-Powered Summarization:** Generate concise summaries of lengthy video content.
- **Action Item Extraction:** Automatically identify tasks and actionable points.
- **Key Decision Detection:** Extract important decisions and unresolved questions.
- **Interactive Q&A:** Ask questions about the video and receive context-aware answers using RAG.
- **Transcript Viewer:** Access the complete generated transcript.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| Whisper | Audio transcription |
| Groq LLM | AI-powered text generation |
| LangChain | LLM orchestration and RAG |
| Vector Database | Context retrieval for Q&A |
| Python-dotenv | Environment variable management |

## 🏗️ Project Structure

```text
AI-Video-Assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── core/
│   ├── transcriber.py
│   ├── summarize.py
│   ├── extract.py
│   └── rag_engine.py
│
└── utils/
    └── audio_processor.py
```

## ⚙️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/AI-Video-Assistant.git

cd AI-Video-Assistant
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the root directory and add the API credentials required by your project modules.

```env
GROQ_API_KEY=your_api_key_here
```

Never commit your `.env` file or expose API credentials publicly.

**5. Run the application**

```bash
streamlit run app.py
```

## 💡 How It Works

1. Provide a video URL or upload a video/audio file.
2. The application processes the input and extracts audio.
3. Whisper generates the video transcript.
4. The LLM generates a title, summary, action items, key decisions, and open questions.
5. A RAG pipeline is created using the transcript.
6. Users can interact with the video by asking questions through the Q&A interface.

## 📂 Supported File Formats

- **Video:** MP4, MOV, MKV, AVI, WEBM
- **Audio:** MP3, WAV, M4A

## 🔮 Future Improvements

- Support for multiple languages.
- Export summaries and transcripts as PDF.
- Conversation history and persistent storage.
- Improved processing for longer videos.
- Timestamp-based transcript navigation.

## 👨‍💻 Author

**Harshal Jaiswal**

Computer Science Engineering | AI & Machine Learning Enthusiast

[GitHub](https://github.com/Jharshal)

---

*Built to make video content easier to understand, explore, and interact with using Generative AI.*
