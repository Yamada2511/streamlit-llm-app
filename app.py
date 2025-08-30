# 必要なライブラリのインストール
# pip install streamlit python-dotenv langchain openai

import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# .envファイルの内容を環境変数として読み込む
load_dotenv()

# 質問内容ごとのシステムプロンプトテンプレート
SYSTEM_TEMPLATES = {
    "健康相談": """あなたは優秀な健康相談の専門家です。
以下の制約条件と入力文をもとに、適切な回答を出力してください。
制約条件:
・専門用語を使わず、わかりやすく説明すること
・一つの質問に対して、一つの回答をすること
・回答は300文字以内で簡潔にまとめること
入力文:
{user_input}""",
    "法律相談": """あなたは優秀な法律相談の専門家です。
以下の制約条件と入力文をもとに、適切な回答を出力してください。
制約条件:
・専門用語を使わず、わかりやすく説明すること
・一つの質問に対して、一つの回答をすること
・回答は300文字以内で簡潔にまとめること
入力文:
{user_input}"""
}

# .envからAPIキーを取得する関数
def get_api_key():
    # Streamlit Cloudではst.secrets、ローカルでは.env
    if os.getenv("STREAMLIT_ENVIRONMENT") == "cloud":
        api_key = st.secrets["OPEN_API_KEY"]
    else:
        api_key = os.getenv("OPEN_API_KEY")
    if not api_key:
        st.error("APIキーが設定されていません。OPEN_API_KEYを.envまたはSecretsに記載してください。")
    return api_key

# 選択された質問内容と入力文からプロンプトを生成する関数
def build_prompt(selected_item, user_input):
    system_template = SYSTEM_TEMPLATES.get(selected_item)
    if not system_template:
        st.error("質問内容の選択が不正です。")
        return None
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{user_input}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt.format_messages(user_input=user_input)

# ストリーミングで回答を表示する関数
def get_answer_stream(selected_item, user_input):
    api_key = get_api_key()
    if not api_key:
        return

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=api_key,
            streaming=True
        )
    except Exception as e:
        st.error(f"LLMの初期化に失敗しました: {e}")
        return

    messages = build_prompt(selected_item, user_input)
    if not messages:
        return

    output_placeholder = st.empty()
    full_text = ""

    try:
        # ストリーミングで応答を受け取り、順次表示
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content"):
                full_text += chunk.content
                output_placeholder.write(full_text)
    except Exception as e:
        st.error(f"回答の取得中にエラーが発生しました: {e}")

# Streamlitアプリのメイン関数
def main():
    st.title("LLM機能を搭載したWebアプリ")
    st.divider()
    st.write("このアプリは、健康相談と法律相談の2つの機能を持っています。")
    st.write("質問内容に応じて、適切な専門家として回答します。")
    st.write("質問内容を選択し、質問を入力して「質問する」ボタンを押してください。")
    st.divider()

    # 質問内容の選択
    selected_item = st.radio(
        "質問内容を選んでください。",
        ["健康相談", "法律相談"]
    )
    st.divider()

    # 質問文の入力
    user_input = st.text_area("質問を入力してください。", height=200)
    st.divider()

    # 質問するボタンが押されたときの処理
    if st.button("質問する"):
        if user_input:
            st.write(f"{selected_item}の専門家として回答します。")
            get_answer_stream(selected_item, user_input)
        else:
            st.warning("質問が入力されていません。")

# スクリプトが直接実行された場合のみmain()を呼び出す
if __name__ == "__main__":
    main()