
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from scipy.spatial.distance import cityblock, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# Đường dẫn tới mô hình
model_path = r'C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\2024_HK2_TGMT_TanDung_Phan_21T1020317\Project\Code\Flower_Recog_Model.keras'


# Kiểm tra và tải mô hình
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please check the path and try again.")
else:
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Flower names
flower_names = ['Hoa Bo Cong Anh', 'Hoa Calimerio', 'Hoa Cam Tu Cau', 'Hoa Cat Tuong', 'Hoa Cuc', 'Hoa Cuc Dai', 'Hoa Cuc PingPong', 'Hoa Hong', 'Hoa Huong Duong', 'Hoa Tana', 'Hoa Tulip']

# Functions
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(result)
    predicted_class = flower_names[predicted_class_index]
    confidence_score = np.max(result) * 100
    return predicted_class, confidence_score

def extract_features(model, image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, 0)
    features = model.predict(input_image_exp_dim)
    return features.flatten()   

def calculate_metrics(model, img_path1, img_path2):
    features1 = extract_features(model, img_path1)
    features2 = extract_features(model, img_path2)

    # L1 Distance
    l1_distance = np.abs(features1 - features2).sum()

    # L2 Distance
    l2_distance = np.sqrt(np.sum((features1 - features2)**2))

    # Cosine Similarity
    cosine_sim = cosine_similarity([features1], [features2])[0][0] * 100

    # Correlation Coefficient
    corr_coeff, _ = pearsonr(features1, features2)
    corr_coeff *= 100

    # Giả định giá trị tối đa của L1 và L2 Distance để chuẩn hóa
    max_l1_distance = 1000  # Bạn có thể điều chỉnh giá trị này
    max_l2_distance = 1000  # Bạn có thể điều chỉnh giá trị này

    l1_distance_percent = (l1_distance / max_l1_distance) * 100
    l2_distance_percent = (l2_distance / max_l2_distance) * 100

    return l1_distance_percent, l2_distance_percent, cosine_sim, corr_coeff

# File uploader
uploaded_files = st.file_uploader('Nhấn để tải ảnh lên: ', accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) >= 2:
    if not os.path.exists('upload'):
        os.makedirs('upload')
    
    img_infos = []
    img_paths = []

    st.subheader("Kết quả phân loại")
    cols = st.columns(len(uploaded_files))

    for i, uploaded_file in enumerate(uploaded_files):
        with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.read())
        img_paths.append(os.path.join('upload', uploaded_file.name))
        img_infos.append((uploaded_file.name, *classify_images(img_paths[i])))
    
    for i, img_info in enumerate(img_infos):
        with cols[i]:
            st.image(img_paths[i], caption=f'Ảnh {i + 1}', width=200)
            st.markdown(f"**Tên ảnh:** {img_info[0]}")
            st.markdown(f"**Loài hoa:** {img_info[1]}")
            st.markdown(f"**Độ chính xác:** {img_info[2]:.2f}%")

    l1_distance_percent, l2_distance_percent, cosine_sim, corr_coeff = calculate_metrics(model, img_paths[0], img_paths[1])

    st.subheader("Độ đo tương đồng giữa hai ảnh")
    st.markdown(f"- **L1 Distance:** {l1_distance_percent:.2f}%")
    st.markdown(f"- **L2 Distance:** {l2_distance_percent:.2f}%")
    st.markdown(f"- **Cosine Similarity:** {cosine_sim:.2f}%")
    st.markdown(f"- **Correlation Coefficient:** {corr_coeff:.2f}%")





# streamlit run c:/Users/Dell/OneDrive/Pictures/Documents/Code/python/OpenCV/2024_HK2_TGMT_TanDung_Phan_21T1020317/Project/Code/App.py
