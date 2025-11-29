import os
import glob
import networkx as nx
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 共通設定 & ユーティリティ
# ==========================================

def load_documents(directory_path):
    """ディレクトリ内の全PDFを読み込む"""
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    all_splits = []
    
    print(f"--- ドキュメント読み込み開始: {len(pdf_files)} ファイル ---")
    for pdf_path in pdf_files:
        print(f"読み込み中: {os.path.basename(pdf_path)}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        all_splits.extend(splits)
        
    print(f"合計 {len(all_splits)} チャンクに分割しました。")
    return all_splits

# ==========================================
# 1. Vanilla LLM (普通のLLM)
# ==========================================

def run_vanilla_llm(query, llm):
    print("\n=== 1. Vanilla LLM (RAGなし) ===")
    print("コンテキストなしで回答を生成中...")
    response = llm.invoke(query)
    return response.content

# ==========================================
# 2. Standard RAG (普通のRAG)
# ==========================================

def run_standard_rag(splits, query, llm):
    print("\n=== 2. Standard RAG (Vector Search) ===")
    print("ベクトルデータベース構築中...")
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    template = """以下のコンテキストに基づいて質問に答えてください:
    {context}
    
    質問: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("検索と回答生成中...")
    response = chain.invoke(query)
    
    # クリーンアップ (メモリ節約のため)
    vectorstore.delete_collection()
    
    return response

# ==========================================
# 3. GraphRAG (知識グラフRAG)
# ==========================================

def extract_triples(text, llm):
    """テキストからトリプルを抽出するヘルパー関数"""
    prompt = ChatPromptTemplate.from_template("""
    以下のテキストから、重要なエンティティ（論文名、手法、概念など）とそれらの関係性を抽出してください。
    特に、異なる論文や手法の間の「比較」「関連」「進化」などの関係に注目してください。
    
    出力形式 (CSV): entity1, relation, entity2
    
    テキスト:
    {text}
    """)
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"text": text})
        triples = []
        for line in result.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 3:
                triples.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
        return triples
    except:
        return []

def run_graph_rag(splits, query, llm):
    print("\n=== 3. GraphRAG (Knowledge Graph) ===")
    
    # 1. グラフ構築 (簡易版: 重要なチャンクのみ処理、または全チャンク処理)
    # デモのため、ランダムまたは先頭/末尾のチャンクだけでなく、
    # 本来は全チャンク処理すべきだが、ここでは時間を節約するために間引くか、
    # 重要なセクション（Abstract, Introduction, Conclusionなど）に絞るのが賢い。
    # 今回はシンプルに「各PDFの最初の5チャンク」と「中間の5チャンク」を使う等の工夫も可能だが、
    # ここでは単純に先頭から一定数処理する。
    
    print("エンティティと関係性を抽出中 (これには時間がかかります)...")
    triples = []
    
    # 処理するチャンク数を制限 (デモ用)
    # 実際の運用では全データまたは要約データを使います
    process_limit = 10 
    target_chunks = splits[:process_limit] 
    
    for i, chunk in enumerate(target_chunks):
        if i % 2 == 0: print(f"Processing chunk {i+1}/{len(target_chunks)}...")
        extracted = extract_triples(chunk.page_content, llm)
        triples.extend(extracted)
        
    print(f"合計 {len(triples)} 個の関係性を抽出しました。")
    
    # 2. グラフ作成
    G = nx.DiGraph()
    for s, r, o in triples:
        G.add_edge(s, o, relation=r)
        
    # 3. 検索 (Graph Traversal / Keyword Matching)
    # クエリに関連するノードを見つける
    relevant_nodes = []
    query_terms = query.lower().split()
    
    for node in G.nodes():
        node_lower = node.lower()
        if any(term in node_lower for term in query_terms if len(term) > 2):
            relevant_nodes.append(node)
            
    # ヒットしなければ次数が高いノードを使用
    if not relevant_nodes:
        print("キーワードヒットなし。重要ノードを使用します。")
        relevant_nodes = [n for n, d in sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]]
    
    # コンテキスト構築 (1-hop 近傍)
    context_lines = []
    for node in relevant_nodes:
        if node in G:
            for neighbor in G[node]:
                rel = G[node][neighbor]['relation']
                context_lines.append(f"{node} -> [{rel}] -> {neighbor}")
    
    graph_context = "\n".join(context_lines)
    print(f"グラフコンテキストサイズ: {len(graph_context)} 文字")
    
    # 4. 回答生成
    template = """以下の知識グラフ情報に基づいて、質問に答えてください。
    特に、エンティティ間のつながりに注目して回答してください。
    
    知識グラフ:
    {context}
    
    質問: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"context": graph_context, "question": query})
    
    # 可視化保存
    try:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.3)
        nx.draw(G, pos, with_labels=True, node_size=1000, font_size=8, alpha=0.6)
        plt.savefig("comparison_graph.png")
        print("グラフ画像を 'comparison_graph.png' に保存しました。")
    except:
        pass

    return response

# ==========================================
# メイン実行
# ==========================================

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # LLM初期化
    llm = ChatOpenAI(model_name="gpt-4.1-2025-04-14", temperature=0)
    
    # ドキュメント読み込み
    splits = load_documents(current_dir)
    
    # 質問
    query = "これら2つの論文（RAGとGraphRAG）の関係性は何ですか？また、GraphRAGは従来のRAGのどのような課題を解決しようとしていますか？"
    
    print(f"\n\n質問: {query}\n" + "="*50)
    
    # 1. Vanilla LLM
    ans_vanilla = run_vanilla_llm(query, llm)
    print("\n[回答] Vanilla LLM:\n" + ans_vanilla)
    
    # 2. Standard RAG
    ans_std = run_standard_rag(splits, query, llm)
    print("\n[回答] Standard RAG:\n" + ans_std)
    
    # 3. GraphRAG
    ans_graph = run_graph_rag(splits, query, llm)
    print("\n[回答] GraphRAG:\n" + ans_graph)
