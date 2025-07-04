# 🔬 Obsidian AI Agent MVP

**Material Science Research Assistant** - 最小限実装版（Minimum Viable Product）

## 🎯 概要

Obsidian AI Agentは材料科学研究に特化したAIアシスタントのMVP版です。複雑な機能を排除し、核心機能のみを提供する軽量なソリューションです。

## ✨ 主要機能

- **💬 AI Chat Interface**: ChatGPT風のクリーンなチャット機能
- **🤖 Multiple AI Models**: Llama3, MaterialsBERT, matscibert対応
- **📁 PDF Upload**: PDF文書のテキスト抽出・処理
- **🔧 Demo Mode**: APIキーなしでも動作するデモモード
- **🎨 Clean UI**: 直感的で美しいStreamlit UI
- **🔑 OpenAI Integration**: オプションのGPT-3.5連携
- **🧪 Materials Science**: 材料科学特化のBERTモデル

## 🚀 クイックスタート

### 必要な環境
- Python 3.8以上
- OpenAI API key（オプション）
- HuggingFace token（オプション、高速ダウンロード用）

### インストール & 実行
```bash
# 依存関係をインストール
pip install -r requirements_mvp.txt

# アプリケーション起動
streamlit run app_mvp.py

# または簡単起動（Windows）
start_mvp.bat
```

### 環境変数（オプション）
```bash
# OpenAI APIを使用する場合
export OPENAI_API_KEY="your_api_key_here"

# HuggingFace token（高速ダウンロード用）
export HF_TOKEN="your_huggingface_token"
```

### AI Models Setup
モデルの詳細なセットアップ方法は [MODEL_SETUP.md](MODEL_SETUP.md) を参照してください。

- **自動ダウンロード**: 初回使用時に自動でモデルをダウンロード
- **ローカルセットアップ**: オフライン使用と高速化のため推奨
- **GPU対応**: CUDA使用可能時は自動でGPUを使用

## 📊 技術仕様

### アーキテクチャ
- **Frontend**: Streamlit（Web UI）
- **AI Backend**: 
  - OpenAI GPT-3.5-turbo（オプション）
  - Llama3（ローカル/API）
  - MaterialsBERT（材料科学特化）
  - matscibert（材料科学特化）
- **PDF Processing**: PyPDF2
- **Storage**: ローカルファイルシステム

### パフォーマンス
- **起動時間**: ~5秒（モデル未ロード時）
- **メモリ使用量**: ~200MB-2GB（使用モデルによる）
- **依存関係**: Transformers, PyTorch等
- **ファイルサイズ**: ~15KB

## 🎮 使用方法

### 1. AIチャット
材料科学に関する質問をAIに投げかけます：
```
"Taylor-Quinney係数について教えて"
"熱力学的結合効果とは？"
```

### 2. PDF処理
研究論文やドキュメントをアップロードしてテキスト抽出を行います。

### 3. システム状態
リアルタイムでAPI接続状況やシステム状態を監視できます。

## 📁 ファイル構成

```
Obsidian/
├── app_mvp.py              # メインアプリケーション
├── requirements_mvp.txt    # 依存関係リスト
├── start_mvp.bat          # Windows用起動スクリプト
├── README.md              # このファイル
├── README_MVP.md          # 詳細なMVPドキュメント
├── MODEL_SETUP.md         # AI モデルセットアップガイド
├── QUICKSTART.md          # 2ステップ起動ガイド
├── GITHUB_SETUP.md        # GitHubセットアップガイド
├── .gitignore             # Git除外設定（モデルファイル除外）
├── models/                # AIモデル（ローカルダウンロード、git除外）
│   ├── matscibert/       # m3rg-iitd/matscibert
│   └── MaterialsBERT/    # pranav-s/MaterialsBERT
└── LICENSE                # ライセンス情報
```

## 🔧 開発情報

### MVP設計思想
- **最小限主義**: 必要な機能のみを実装
- **高速起動**: 3秒以下での起動を実現
- **軽量**: 50MB以下のメモリ使用量
- **独立性**: 外部依存を最小限に抑制

### 削減された機能
- 複雑なAIモデル管理
- ファインチューニング機能
- CLI tools
- 過度な設定オプション

## 🎯 今後の拡張予定

### Phase 2
- Vector database integration
- Local AI model support
- Advanced PDF processing
- Citation tracking

### Phase 3
- Fine-tuning capabilities
- Multi-document chat
- Export functionality
- Advanced analytics

## 📞 サポート

MVP版は教育・プロトタイプ目的で作成されています。
本格的な研究用途には将来のフル版をご利用ください。

## 📄 ライセンス

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🔬 Material Science Research Made Simple**
