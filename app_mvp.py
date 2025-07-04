#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Obsidian AI Agent MVP - 最小限の実装
シンプルなAI論文レビューシステム
"""

import streamlit as st
import os
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any
import PyPDF2
import io

# ページ設定
st.set_page_config(
    page_title="Obsidian AI - MVP",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# シンプルなCSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    
    .main > div {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .user-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0084ff;
    }
    
    .ai-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b35;
        border: 1px solid #e1e5e9;
    }
</style>
""", unsafe_allow_html=True)

class SimpleAIClient:
    """シンプルなAIクライアント - OpenAI APIまたはダミーレスポンス"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.has_api = bool(self.api_key)
        
    def generate_response(self, prompt: str) -> str:
        """AIレスポンスを生成"""
        if self.has_api:
            try:
                import openai
                openai.api_key = self.api_key
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a materials science expert specializing in thermomechanics and Taylor-Quinney coefficient research. Provide detailed, technical responses."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"API Error: {e}"
        else:
            # ダミーレスポンス（デモ用）
            return self._generate_demo_response(prompt)
    
    def _generate_demo_response(self, prompt: str) -> str:
        """デモ用のダミーレスポンス"""
        demo_responses = {
            "taylor": "The Taylor-Quinney coefficient (β) represents the fraction of plastic work converted to heat during deformation. Typically ranges from 0.8-0.95 for most metals.",
            "thermomechanics": "Thermomechanics studies the coupling between mechanical deformation and thermal effects in materials under loading conditions.",
            "materials": "Materials science focuses on understanding structure-property relationships in engineering materials.",
            "default": f"This is a demo response to your query: '{prompt[:50]}...'. In a full implementation, this would be answered by an AI model trained on materials science literature."
        }
        
        prompt_lower = prompt.lower()
        for key, response in demo_responses.items():
            if key in prompt_lower:
                return response
        
        return demo_responses["default"]

def extract_text_from_pdf(uploaded_file) -> str:
    """PDFからテキストを抽出"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text[:1000]  # 最初の1000文字のみ
    except Exception as e:
        return f"PDF読み込みエラー: {e}"

def main():
    """メインアプリケーション"""
    
    # タイトル
    st.markdown("""
    <div class="title-container">
        <h1>🔬 Obsidian AI Agent MVP</h1>
        <p>Material Science Research Assistant - Minimal Viable Product</p>
    </div>
    """, unsafe_allow_html=True)
    
    # モード選択
    tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "📁 PDF Upload", "ℹ️ About"])
    
    with tab1:
        show_chat_interface()
    
    with tab2:
        show_pdf_upload()
    
    with tab3:
        show_about()

def show_chat_interface():
    """AIチャットインターフェース"""
    
    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "ai_client" not in st.session_state:
        st.session_state.ai_client = SimpleAIClient()
    
    # API状態表示
    if st.session_state.ai_client.has_api:
        st.success("✅ OpenAI API connected")
    else:
        st.warning("⚠️ OpenAI API not found - using demo mode")
        st.info("Set OPENAI_API_KEY environment variable to enable full AI features")
    
    # チャット履歴表示
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ai-message">
                <strong>🔬 Obsidian AI:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # チャット入力
    user_input = st.text_input("Ask about materials science:", key="chat_input")
    
    if st.button("Send") and user_input:
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AIレスポンスを生成
        with st.spinner("Thinking..."):
            ai_response = st.session_state.ai_client.generate_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        st.rerun()
    
    # クリアボタン
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

def show_pdf_upload():
    """PDFアップロード機能"""
    st.header("📁 PDF Document Processor")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        if st.button("Extract Text"):
            with st.spinner("Processing PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
            
            st.subheader("📄 Extracted Text (Preview)")
            st.text_area("Content:", extracted_text, height=300)
            
            # 保存オプション
            if st.button("Save to Knowledge Base"):
                # 簡単なファイル保存
                save_path = Path("uploaded_docs") 
                save_path.mkdir(exist_ok=True)
                
                with open(save_path / f"{uploaded_file.name}.txt", "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                
                st.success(f"💾 Text saved to knowledge base!")

def show_about():
    """アプリ情報"""
    st.header("ℹ️ About Obsidian AI MVP")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 MVP Features")
        st.markdown("""
        - ✅ Simple AI chat interface
        - ✅ PDF text extraction
        - ✅ Demo mode (no API required)
        - ✅ Clean, minimal UI
        - ✅ OpenAI API integration (optional)
        """)
        
        st.subheader("🔧 Technical Stack")
        st.markdown("""
        - **Frontend**: Streamlit
        - **AI**: OpenAI GPT-3.5 (optional)
        - **PDF**: PyPDF2
        - **Storage**: Local files
        """)
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Chat**: Ask materials science questions
        2. **Upload**: Process PDF documents
        3. **API**: Set OPENAI_API_KEY for full features
        """)
        
        st.subheader("🎯 Next Steps")
        st.markdown("""
        - Add vector database for PDFs
        - Implement fine-tuned models
        - Add citation tracking
        - Expand knowledge domains
        """)
    
    # システム情報
    st.markdown("---")
    st.subheader("💻 System Status")
    
    status_cols = st.columns(3)
    with status_cols[0]:
        st.metric("UI Status", "🟢 Online")
    with status_cols[1]:
        ai_status = "🟢 API" if st.session_state.get("ai_client", SimpleAIClient()).has_api else "🟡 Demo"
        st.metric("AI Status", ai_status)
    with status_cols[2]:
        st.metric("PDF Processor", "🟢 Ready")

if __name__ == "__main__":
    main()
