# 🦙 Llama3 Setup Guide

## Congratulations on getting Llama3 access! 🎉

このガイドでは、Obsidian AI AgentでLlama3を使用するための設定方法を説明します。

## 1. Hugging Face トークンの取得

1. **Hugging Face アカウント作成/ログイン**
   - [https://huggingface.co/](https://huggingface.co/) にアクセス
   - アカウントでログイン

2. **アクセストークンの作成**
   - [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) にアクセス
   - "New token" をクリック
   - Token type: "Read" を選択
   - Name: "Obsidian-AI" (任意の名前)
   - "Generate a token" をクリック
   - **重要**: トークンをコピーして安全に保存

## 2. 環境変数の設定

### Windows (PowerShell)
```powershell
# 一時的設定（セッション中のみ有効）
$env:HF_TOKEN="your_token_here"

# 永続的設定
[Environment]::SetEnvironmentVariable("HF_TOKEN", "your_token_here", "User")
```

### Windows (コマンドプロンプト)
```cmd
# 一時的設定
set HF_TOKEN=your_token_here

# 永続的設定は環境変数設定画面から行う
```

### .env ファイル使用（推奨）
プロジェクトルートに `.env` ファイルを作成：
```env
HF_TOKEN=your_token_here
```

## 3. アプリケーションの起動

1. **環境変数設定後、アプリを再起動**
   ```powershell
   python app_mvp.py
   ```

2. **Streamlit WebUIで確認**
   - サイドバーの "🦙 Llama3" セクション
   - "🔑 HF_TOKEN detected" メッセージを確認
   - "🚀 Load Llama3 Model" ボタンをクリック

## 4. 初回読み込み

- **初回は数分かかる場合があります**（モデルダウンロード）
- GPU使用時は約8GBのVRAMを使用
- CPU使用時は16GB以上のRAMを推奨

## 5. 使用開始

✅ Llama3が読み込まれたら：
- "Prefer Llama3" オプションを有効化
- 材料科学に関する複雑な質問を試してみてください
- MaterialsBERTと組み合わせて使用することで最高の結果が得られます

## トラブルシューティング

### エラー: "401 Unauthorized"
- HF_TOKENが正しく設定されていません
- トークンの権限を確認してください（Read権限が必要）

### エラー: "Repository not found"
- Meta Llamaモデルへのアクセス許可が必要です
- [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) でRequest Accessをクリック

### メモリ不足
- GPU VRAM不足: より小さなモデルを検討
- CPU RAM不足: 4-bit量子化が自動で適用されます

## サポート

問題が発生した場合は、以下を確認してください：
1. HF_TOKEN環境変数の設定
2. Meta Llamaモデルへのアクセス許可
3. システムリソース（GPU VRAM/CPU RAM）

Happy researching! 🔬✨
