# Obsidian AI Agent MVP

**最小限実装版** - 必要最小限の機能のみを含むMVP（Minimum Viable Product）

## 🎯 MVP Features

### ✅ 実装済み
- **💬 AI Chat Interface**: シンプルなチャット機能
- **📁 PDF Upload**: PDF文書のテキスト抽出
- **🔧 Demo Mode**: APIキーなしでも動作
- **🎨 Clean UI**: 最小限のStreamlit UI
- **🔑 OpenAI Integration**: オプションのAPI連携

### 🚫 削除された複雑な機能
- 複数のAIモデルクライアント
- ファインチューニング機能
- CLI tools
- 複雑な設定ファイル
- 過度な依存関係

## 🚀 Quick Start

### 必要なもの
- Python 3.8+
- OpenAI API key (オプション)

### インストール
```bash
# MVP版の依存関係をインストール
pip install -r requirements_mvp.txt

# アプリケーション実行
streamlit run app_mvp.py
```

### 環境変数（オプション）
```bash
# OpenAI APIを使用する場合
set OPENAI_API_KEY=your_api_key_here
```

## 📁 ファイル構造（MVP）

```
scholarFetcher/
├── app_mvp.py              # メインアプリケーション
├── requirements_mvp.txt    # 最小限の依存関係
└── README_MVP.md          # このファイル
```

## 🎮 使用方法

1. **AI Chat**: 材料科学について質問
2. **PDF Upload**: PDF文書をアップロードしてテキスト抽出
3. **Demo Mode**: APIキーなしでもデモレスポンスを確認

## 🔧 技術詳細

### アーキテクチャ
- **Frontend**: Streamlit（シンプルなWeb UI）
- **AI Backend**: OpenAI GPT-3.5-turbo（オプション）
- **PDF Processing**: PyPDF2
- **Storage**: ローカルファイル

### API Modes
- **Full Mode**: OpenAI API使用（OPENAI_API_KEY設定時）
- **Demo Mode**: 事前定義されたレスポンス（APIキーなし）

## 📊 パフォーマンス

- **起動時間**: ~3秒
- **メモリ使用量**: ~50MB
- **依存関係**: 4個のパッケージのみ
- **ファイルサイズ**: ~15KB

## 🎯 次のステップ

### Phase 2で追加予定
- Vector database for PDF search
- Local AI model integration
- Advanced PDF processing
- Citation tracking

### Phase 3で追加予定  
- Fine-tuning capabilities
- Multi-document chat
- Export functionality
- Advanced analytics

## 🐛 トラブルシューティング

### よくある問題
1. **"Module not found"**: `pip install -r requirements_mvp.txt`を実行
2. **"API Error"**: OPENAI_API_KEYが正しく設定されているか確認
3. **"PDF Error"**: PDFファイルが破損していないか確認

## 📞 サポート

MVP版は教育・デモ目的です。本格的な使用には元の完全版をご利用ください。
