import streamlit as st
import asyncio
from playwright.sync_api import sync_playwright
from agent_main import call_agent

with open("./css_script.css", 'r') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.write("# Web Voyager")
st.image("./frontend/Graph.png", caption="Web Voyager Components")

search_title = st.text_input("Input search title: ", placeholder="What is Capibara")

# st.set_option('server.headless', True)
# st.set_option('server.enableCORS', False)
# st.set_option('server.enableWebsocketCompression', False)


def start_playwright_session():
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True, args=None)
    page = browser.new_page()
    page.goto("https://www.google.com")

    # Store browser and page in session state
    st.session_state.playwright_context = p
    st.session_state.playwright_browser = browser
    st.session_state.playwright_page = page


def close_playwright_session():
    st.session_state.playwright_browser.close()
    st.session_state.playwright_context.stop()

    del st.session_state.playwright_browser
    del st.session_state.playwright_page
    del st.session_state.playwright_context

if search_title:
    if (
        not getattr(st.session_state, "search_title", False)
        or search_title != st.session_state.search_title
    ):
        start_playwright_session()
        setattr(st.session_state, "search_title", search_title)
        setattr(
            st.session_state,
            "output_content",
            call_agent(
                question=search_title,
                page=st.session_state.playwright_page,
                max_step=3,
            ),
        )

if getattr(st.session_state, "output_content", None):
    # Create a container with a box border
    with st.container(border=True):  # Using st.container instead of st.form
        st.write("Gemini Output:")
        try:
            for step_info, image in st.session_state.output_content:
                st.write(step_info)
                if image:
                    st.image(image=image, caption="Browser Screenshot")
        except StopIteration:
            st.write("Search completed")
            close_playwright_session()
