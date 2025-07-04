# Quick Start Guide - Obsidian AI MVP

## 🚀 Getting Started in 2 Steps (MVP Version)

### 1. Install Minimal Dependencies
```bash
pip install -r requirements_mvp.txt
```

### 2. Run the MVP Application
```bash
streamlit run app_mvp.py
```

### 3. Open in Browser
- Navigate to: `http://localhost:8501`
- Choose **AI Chat** mode
- Start asking questions!

## 💬 Example Questions

Try these queries to get started:

```
"What is the Taylor-Quinney coefficient?"
"Explain thermomechanics in materials"
"How do I analyze material properties?"
"Latest research in battery materials"
```

## 📁 Upload PDFs

1. Switch to **PDF Upload** mode
2. Upload your research papers
3. AI will learn from your documents
4. Get better, more specific answers

## 🎯 Key Features

- **Real-time AI Chat**: Instant responses
- **Specialized Knowledge**: Material science focus
- **PDF Learning**: Expand AI knowledge
- **Clean Interface**: Easy to use

## ⚠️ Troubleshooting

### "AI Model not available"
- Check if `transformers` is installed
- Verify Python version (3.8+)
- Try: `pip install torch transformers`

### Port already in use
- Change port: `streamlit run app.py --server.port 8502`
- Or kill existing process

### PDF upload fails
- Check file size (max 50MB)
- Ensure it's a valid PDF
- Try a different PDF

## 🔧 Configuration

Edit `obsidian_config.ini` to customize:
- AI model settings
- UI preferences  
- Feature toggles

---

**Need help?** Check the main README.md or open an issue.
