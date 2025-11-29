# 【10 Days Challenge】論文解説AI「Paper Copilot」開発ログ

RAGからAgentまで、10日間で論文解説AI「Paper Copilot」を開発するチャレンジのリポジトリです。

## プロジェクト概要
LLM（大規模言語モデル）に外部知識を組み込む **RAG (Retrieval-Augmented Generation)** 技術から始め、最終的には自律的にタスクをこなす **Agent** の開発までを目指します。

## 進捗状況

- [x] **Day 1**: RAGの基本構造と実装 (Basic RAG Implementation)
- [x] **Day 2**: RAGの検索精度を向上させる「Hybrid Search」と「Re-ranking」の実装
- [x] **Day 3**: GraphRAGの基本構造と実装 (Basic GraphRAG Implementation)
- [ ] ...

## 実行方法

1. 必要なライブラリをインストール:
   ```bash
   pip install langchain langchain-openai langchain-chroma langchain-community pypdf langchain_classic
   ```

2. OpenAI APIキーを設定し、スクリプトを実行:
   ```bash
   cd Day1
   python day1_rag.py
   ```

## 備考
現在はDay 1の段階です。今後の更新をお楽しみに！
