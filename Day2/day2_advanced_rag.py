import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# API Keyの設定（環境変数に設定することを推奨）
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
def run_no_rag(query):
    print(f"--- LLMに直接質問 ---")

    llm = ChatOpenAI(model_name="gpt-4.1-2025-04-14", temperature=0)

    print(f"\n--- ユーザーの質問: {query} ---")
    
    message = llm.invoke(query)
    
    return message.content

def run_advanced_rag(pdf_path, query):
    print(f"--- Day 2: Advanced RAG (Hybrid + Rerank) 開始 ---")
    print(f"ファイル: {pdf_path}")

    # ---------------------------------------------------------
    # 1. データ読み込み & チャンキング (Data Ingestion & Chunking)
    # ---------------------------------------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"ドキュメントを {len(splits)} 個のチャンクに分割しました")

    # ---------------------------------------------------------
    # 2. 検索器の準備 (Retriever Setup)
    # ---------------------------------------------------------
    
    # A. Vector Retriever (意味検索)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # B. BM25 Retriever (キーワード検索)
    # 必要なライブラリ: rank_bm25
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5

    # C. Ensemble Retriever (Hybrid Search)
    # weights=[0.5, 0.5] でベクトル検索とキーワード検索を等しく評価
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )
    
    # ---------------------------------------------------------
    # 3. Re-ranking (精度向上)
    # ---------------------------------------------------------
    # 検索結果をさらに高精度なモデルで並び替える
    # 使用モデル: BAAI/bge-reranker-base (多言語対応で軽量)
    # 必要なライブラリ: sentence-transformers, langchain-huggingface
    print("Re-rankerモデルを読み込み中... (初回は時間がかかります)")
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=5) # 上位5つに絞る

    #最終的なRetriever: Ensemble -> Rerank
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    # ---------------------------------------------------------
    # 4. 生成 (Generation)
    # ---------------------------------------------------------
    llm = ChatOpenAI(model_name="gpt-4.1-2025-04-14", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""以下のコンテキストに基づいて、質問に答えてください:
    <context>{context}</context>
    質問: {input}""")

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(compression_retriever, combine_docs_chain)

    # 実行
    print(f"\n--- ユーザーの質問: {query} ---")
    result = qa_chain.invoke({"input": query})
    
    return result

if __name__ == "__main__":
    # Day1のPDFファイルを使用
    # 相対パスで指定: ../Day1/Retrieval...pdf
    current_dir = os.path.dirname(os.path.abspath(__file__))
    day1_dir = os.path.join(current_dir, "..", "Day1")
    pdf_name = 'Retrieval-Augmented_Generation_for_Knowledge-Intensive_NLP_Tasks.pdf'
    pdf_path = os.path.join(day1_dir, pdf_name)
    
    # ファイルの存在確認
    if not os.path.exists(pdf_path):
        print(f"エラー: ファイルが見つかりません: {pdf_path}")
        print("Day1ディレクトリにPDFファイルがあることを確認してください。")
    else:
        query = f"この論文{pdf_name}の主な貢献は何ですか？簡潔な言葉でまとめてください。"
        #response = run_no_rag(query)
        #print("\n=== AIの回答 (LLM Only) ===")
        #print(response)
        try:
            response = run_advanced_rag(pdf_path, query)
            print("\n=== AIの回答 ===")
            print(response["answer"])
            
            print("\n=== 参照元 ===")
            for i, doc in enumerate(response["context"]):
                print(f"[Source {i+1}] Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            print("必要なライブラリがインストールされているか確認してください: rank_bm25, sentence-transformers, langchain-huggingface")
