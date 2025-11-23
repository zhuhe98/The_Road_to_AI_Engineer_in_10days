import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. API Keyの設定（環境変数に設定する）

def run_no_rag(query):
    print(f"--- LLMに直接質問 ---")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    print(f"\n--- ユーザーの質問: {query} ---")
    
    message = llm.invoke(query)
    
    return message.content


def run_basic_rag(pdf_path, query):
    print(f"--- ファイル処理中: {pdf_path} ---")

    # ---------------------------------------------------------
    # ステップ1: データ読み込み (Data Ingestion) [cite: 12]
    # ---------------------------------------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"PDFの読み込み成功、合計 {len(documents)} ページ")

    # ---------------------------------------------------------
    # ステップ2: テキスト分割 (Chunking) [cite: 12]
    # ---------------------------------------------------------
    # チャンキングはテキストがモデルのコンテキスト制限を超えるのを防ぎ、検索精度を向上させます
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 各チャンクのサイズ
        chunk_overlap=200 # チャンク間の重複（コンテキストの断裂を防ぐ）
    )
    splits = text_splitter.split_documents(documents)
    
    print(f"ドキュメントを {len(splits)} 個のチャンク (Chunks) に分割しました")

    # ---------------------------------------------------------
    # ステップ3: ベクトル化と保存 (Embedding & Vector Storage) [cite: 13]
    # ---------------------------------------------------------
    # OpenAIのEmbeddingモデルを使用してテキストをベクトルに変換
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Chromaベクトルデータベースの作成（メモリ内に保存、再起動後は消失、テストに便利）
    print("ベクトルデータベースを構築中...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model
    )
    print("ベクトルデータベースの構築完了！")

    # ---------------------------------------------------------
    # ステップ4: 検索と生成 (Retrieval & Generation) [cite: 14]
    # ---------------------------------------------------------
    # LLMの初期化（ここではgpt-3.5-turboまたはgpt-4oを使用）
    llm = ChatOpenAI(model_name="gpt-4.1-2025-04-14", temperature=0)

    # レトリーバー (Retriever) の作成
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 質問応答チェーン (QA Chain) の構築
    # プロンプトテンプレートの作成
    prompt = ChatPromptTemplate.from_template("""以下のコンテキストに基づいて、質問に答えてください:
    <context>{context}</context>
    質問: {input}""")

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 質問の実行
    print(f"\n--- ユーザーの質問: {query} ---")
    result = qa_chain.invoke({"input": query})
    
    return result


if __name__ == "__main__":
    # --- 設定エリア ---
    # ディレクトリに "paper.pdf" という名前のファイルがあることを確認するか、ここのパスを変更してください
    paper_name = 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks'
    my_pdf_path = f"{paper_name}.pdf" 
    my_question = f"この論文{paper_name}の主な貢献は何ですか？簡潔な言葉でまとめてください。" # [cite: 14]

    # --- プログラム実行 ---
    try:
        response_no_rag = run_no_rag(my_question)
        print("\n=== AIの回答 (No RAG) ===")
        print(response_no_rag)

        response = run_basic_rag(my_pdf_path, my_question)
        
        print("\n=== AIの回答 (Basic RAG) ===")
        print(response["answer"])
        
        print("\n=== 参照元 (Source Documents) ===")
        for i, doc in enumerate(response["context"]):
            print(f"[Source {i+1}] Page {doc.metadata['page']}: {doc.page_content[:50]}...")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("ヒント: ファイル名が正しいか、およびAPI Keyが設定されているか確認してください。")