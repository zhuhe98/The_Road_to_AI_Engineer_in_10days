# 【Day 1】LangChainで始めるRAG入門：PDFと対話するAIを作ろう

## はじめに
こんにちは！
本記事は、**「10日間でマスターするRAG（Retrieval-Augmented Generation）」**シリーズの記念すべき第1日目です。

このシリーズでは、LLM（大規模言語モデル）アプリケーション開発において現在最も注目されている技術の一つである **RAG** について、基礎から応用までをコード付きで解説していきます。

Day 1のテーマは**「RAGの基本構造と実装」**です。
LangChainを使って、PDFドキュメントの内容に基づいて回答するシンプルなRAGアプリを作成します。

## RAG（検索拡張生成）とは？
RAG (Retrieval-Augmented Generation) は、LLMが学習していない**外部データ（社内ドキュメント、最新ニュース、専門書など）**を検索し、その情報をプロンプトに含めることで、より正確で具体的な回答を生成させる技術です。

### なぜRAGが必要なのか？
LLM単体では以下の課題があります：
1. **情報の鮮度**: 学習データに含まれていない最新情報を知らない。
2. **ハルシネーション**: 嘘の情報をもっともらしく答えてしまうことがある。
3. **プライベートデータ**: 社内規定や個人のメモなど、非公開データを知らない。

RAGを使うことで、これらの課題を解決し、**「信頼性の高い、根拠に基づいた回答」**を得ることができます。

## 実装の流れ
今回は `day1_rag.py` というファイルを作成し、以下のステップで実装します。

1. **Data Ingestion**: PDFを読み込む
2. **Chunking**: テキストを適切なサイズに分割する
3. **Embedding & Vector Store**: テキストをベクトル化してデータベースに保存する
4. **Retrieval & Generation**: 質問に関連する情報を検索し、LLMに回答させる

## 環境準備
以下のライブラリを使用します。
```bash
pip install langchain langchain-openai langchain-chroma langchain-community pypdf
```

また、OpenAI API Keyが必要です。

## コード解説
それでは、実際のコードを見ていきましょう。

### 1. ライブラリのインポートと設定
```python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# API Keyの設定
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 2. PDFの読み込み (Data Ingestion)
まずは対象となるPDFファイルを読み込みます。今回はRAGの有名な論文「Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks」を使用します。

```python
def run_basic_rag(pdf_path, query):
    print(f"--- ファイル処理中: {pdf_path} ---")
    
    # PDFローダーの初期化と読み込み
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"PDFの読み込み成功、合計 {len(documents)} ページ")
```

### 3. テキスト分割 (Chunking)
LLMには一度に入力できるトークン数に制限があるため、長いドキュメントは小さな「チャンク」に分割する必要があります。

```python
    # テキスト分割の設定
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 1つのチャンクの文字数
        chunk_overlap=200 # チャンク間の重複（文脈の分断を防ぐため）
    )
    splits = text_splitter.split_documents(documents)
    print(f"ドキュメントを {len(splits)} 個のチャンクに分割しました")
```

### 4. ベクトル化と保存 (Embedding & Vector Store)
分割したテキストを、意味を理解しやすい「ベクトル（数値の羅列）」に変換し、検索可能な状態で保存します。今回は軽量な `Chroma` を使用します。

```python
    # Embeddingモデルの準備
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # ベクトルデータベースの作成
    print("ベクトルデータベースを構築中...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model
    )
    print("ベクトルデータベースの構築完了！")
```

### 5. 検索と生成 (Retrieval & Generation)
最後に、ユーザーの質問に関連するチャンクを検索し、それをコンテキストとしてLLMに渡して回答を生成します。

```python
    # LLMの準備
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Retriever（検索機）の作成
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # プロンプトの定義
    prompt = ChatPromptTemplate.from_template("""以下のコンテキストに基づいて、質問に答えてください:
    <context>{context}</context>
    質問: {input}""")

    # Chainの構築
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 実行
    result = qa_chain.invoke({"input": query})
    return result
```

## 実行結果の比較
このコードでは、RAGを使わない場合（LLMの知識のみ）と、RAGを使った場合（PDFの知識あり）を比較できます。

**質問**: 「この論文の主な貢献は何ですか？」

### No RAG (LLMのみ)
> GPT-3.5などは学習データに含まれていれば答えられますが、最新の論文や社内文書の場合は「わかりません」や一般的な回答しか返ってきません。

### Basic RAG (PDF参照)
> 論文の内容に基づき、具体的な貢献点（パラメトリックメモリとノンパラメトリックメモリの融合など）を正確に回答してくれます。また、回答の根拠となったページ番号やテキストも確認できます。

## まとめ
Day 1では、LangChainを使った基本的なRAGパイプラインを構築しました。
わずか数十行のコードで、独自のデータに基づいて回答するAIが作れることがわかりました。

次回 **Day 2** では、より高度な検索手法や、精度の向上テクニックについて掘り下げていきます。お楽しみに！

---
**参考リンク**
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API](https://platform.openai.com/docs/introduction)

https://docs.langchain.com/oss/python/migrate/langchain-v1
