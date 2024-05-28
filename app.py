import streamlit as st
from prediction_page import prediction_page
import fitz  # PyMuPDF
import io
from PIL import Image

def main():
    dark_mode = st.sidebar.checkbox('Dark Mode', key="dark_mode")
    st.sidebar.markdown(
        f"<h1 style='color: {"#000000" if dark_mode else "#000000"};'>Navigation</h1>",
        unsafe_allow_html=True
    )
    add_css(dark_mode)  # Apply CSS based on dark mode
    page = st.sidebar.radio(
        "Go to",
        ["Prediction", "About"],
        index=0,
        format_func=lambda x: f"📊 {x}" if x == "Prediction" else f"ℹ️ {x}",
    )

    if page == "Prediction":
        prediction_page(dark_mode)
    elif page == "About":
        st.title("Our Model and Its Impact in combating racism")
        st.markdown("""
        <div class="about-section">
            <p>Many tweets contain harmful content that can affect individuals and communities. This model helps in identifying such racist content in tweets, enabling us to take appropriate actions to reduce the spread of harmful content on social media platforms.</p>
            <p>We utilized a combination of TF-IDF vectorization and a stacking classifier for our model. The stacking classifier includes a logistic regression and decision tree as base estimators, with a linear SVM as the final estimator. This approach allows us to effectively capture the nuances in the tweet text and improve the accuracy of our predictions.</p>
            <p>This chart provides an overview of perceptions of racism in Canada and around the world. It highlights the prevalence of racist attitudes and the need for increased awareness and interventions to combat racism in society. By understanding the extent of these perceptions, we can better address the underlying issues and work towards creating a more inclusive and equitable environment for everyone.</p>
        </div>
        """, unsafe_allow_html=True)

        # Add the PDF and description
        pdf_path = r"C:\Users\dekis\OneDrive\Desktop\DATA SCIENCE\FINAL PROJECT\ds-final-project-main\ds-final-project-main\racism graphs comparison canada and rest of the world.pdf"
        
        # Read the PDF
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(0)  # Load first page

        # Adjust the zoom level to scale down the image
        zoom_x = 2.0  # Horizontal zoom
        zoom_y = 2.0  # Vertical zoom
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert the image to a streamlit displayable format
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        st.image(img_byte_arr, use_column_width=True, caption='Perceptions of Racism in Canada and the World')

        # Add footer at the bottom of the page
        st.markdown("""
        <div class="footer">
            Developed by Sophia D
        </div>
        """, unsafe_allow_html=True)

def add_css(dark_mode):
    background_color = "#2e2e2e" if dark_mode else "#add8e6"
    text_color = "#ffffff" if dark_mode else "#000000"
    card_color = "#1e1e1e" if dark_mode else "#f0f0f0"
    title_color = "#ffffff" if dark_mode else "#000000"
    button_color = "#000000" if dark_mode else card_color
    download_button_color = "#ff0000" if dark_mode else card_color

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}
        .logo {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
        }}
        .about-section {{
            background-color: {card_color};
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: left;
            font-size: 16px;
            line-height: 1.6;
            color: {text_color};
        }}
        .footer {{
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: {background_color};
            text-align: center;
            padding: 10px;
            color: {text_color};
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
            color: {title_color};
        }}
        .stTextArea > label {{
            color: {text_color};
        }}
        .stButton > button {{
            background-color: {button_color};
            color: {text_color};
        }}
        .stDownloadButton > button {{
            background-color: {download_button_color};
            color: {text_color};
        }}
        .css-1d391kg, .css-1d391kg h1 {{
            color: {text_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
