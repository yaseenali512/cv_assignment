import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from filters import apply_low_pass_filter, apply_low_pass_butterworth_filter, apply_high_pass_laplacian_filter, histogram_matching
from streamlit_option_menu import option_menu

def main():
    selected = option_menu(
        menu_title=None,
        options=["Welcome", "Filters"],
        icons=["house", "filter"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Welcome":
        show_welcome_page()
    elif selected == "Filters":
        show_filters_page()

def show_welcome_page():
    st.title("Welcome to the Digital Image Processing App!")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='text-align: center; color: #333;'>Welcome to the Digital Image Processing App!</h2>
            <p style='font-size: 18px; text-align: center; color: #555;'>
                This application allows you to apply various image processing techniques to your images.
                Use the navigation bar to go to different pages.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_filters_page():
    filter_selected = option_menu(
        menu_title="Select a Filter",
        options=[
            "Lowpass Gaussian Filter (Spatial Domain)",
            "Lowpass Butterworth Filter (Frequency Domain)",
            "Highpass Laplacian Filter (Spatial Domain)",
            "Histogram Matching",
        ],
        icons=["circle", "filter", "highlighter", "histogram", "search"],
        menu_icon="filter",
        default_index=0,
        orientation="horizontal",
    )

    if filter_selected == "Lowpass Gaussian Filter (Spatial Domain)":
        show_low_pass_gaussian_filter_page()
    elif filter_selected == "Lowpass Butterworth Filter (Frequency Domain)":
        show_low_pass_butterworth_filter_page()
    elif filter_selected == "Highpass Laplacian Filter (Spatial Domain)":
        show_high_pass_laplacian_filter_page()
    elif filter_selected == "Histogram Matching":
        show_histogram_matching_page()

def show_low_pass_gaussian_filter_page():
    st.title("Low-Pass Gaussian Filtering Application")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 0)
        st.image(img, caption='Original Image', use_column_width=True)
        sigma = st.slider("Select Gaussian Sigma", min_value=0.1, max_value=5.0, value=1.0)
        filtered_img = apply_low_pass_filter(img, sigma)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image")  
            st.image(img, use_column_width=True)
        with col2:
            st.write("Filtered Image")
            st.image(filtered_img, use_column_width=True)

def show_low_pass_butterworth_filter_page():
    st.title("Low-Pass Butterworth Filter (Frequency Domain)")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 0)
        st.image(img, caption='Original Image', use_column_width=True)
        d0 = st.slider("Select Cutoff Frequency (D0)", min_value=10, max_value=100, value=40)
        n = st.slider("Select Order of Filter (n)", min_value=1, max_value=10, value=2)
        filtered_img = apply_low_pass_butterworth_filter(img, d0, n)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image")
            st.image(img, use_column_width=True)
        with col2:
            st.write("Filtered Image")
            st.image(filtered_img, use_column_width=True)

def show_high_pass_laplacian_filter_page():
    st.title("High-Pass Laplacian Filter (Spatial Domain)")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 0)
        st.image(img, caption='Original Image', use_column_width=True)
        filtered_img = apply_high_pass_laplacian_filter(img)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image")
            st.image(img, use_column_width=True)
        with col2:
            st.write("Filtered Image")
            st.image(filtered_img, use_column_width=True)

def show_histogram_matching_page():
    st.title("Histogram Matching")
    uploaded_file1 = st.file_uploader("Choose the source image...", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Choose the reference image...", type=["jpg", "jpeg", "png"])
    if uploaded_file1 is not None and uploaded_file2 is not None:
        file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
        source_img = cv.imdecode(file_bytes1, 0)
        file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
        reference_img = cv.imdecode(file_bytes2, 0)
        matched_img = histogram_matching(source_img, reference_img)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Source Image")
            st.image(source_img, use_column_width=True)
        with col2:
            st.write("Reference Image")
            st.image(reference_img, use_column_width=True)
        with col3:
            st.write("Matched Image")
            st.image(matched_img, use_column_width=True)

if __name__ == "__main__":
    main()
