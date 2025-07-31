# Face Unlock System - YOLOv8 + FaceNet + Streamlit

import cv2
import numpy as np
import streamlit as st
import os
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

# Load models
face_model = InceptionResnetV1(pretrained='vggface2').eval()
detector = YOLO("yolov8n-face.pt")  # You need to place this file in working dir

EMBEDDINGS_PATH = "embeddings/"
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

# Utility: Get embedding from face image
def get_embedding(img):
    img = cv2.resize(img, (160, 160))
    img = img / 255.0
    img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        emb = face_model(img)
    return emb.numpy()[0]

# Utility: Detect face and crop
def detect_face(image):
    results = detector(image)
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0].astype(int)
            return image[y1:y2, x1:x2]
    return None

# Save embedding to file
def save_embedding(name, emb):
    np.save(os.path.join(EMBEDDINGS_PATH, name + ".npy"), emb)

# Load embeddings
def load_embeddings():
    db = {}
    for file in os.listdir(EMBEDDINGS_PATH):
        if file.endswith(".npy"):
            name = file[:-4]
            emb = np.load(os.path.join(EMBEDDINGS_PATH, file))
            db[name] = emb
    return db

# Match face
def match_face(embedding, db, threshold=0.6):
    for name, db_emb in db.items():
        sim = cosine_similarity([embedding], [db_emb])[0][0]
        if sim > threshold:
            return name, sim
    return None, None

# Streamlit App
st.title("Face Unlock System")
mode = st.radio("Select Mode", ["Register", "Authenticate"])

cam = cv2.VideoCapture(0)
run = st.button("Start Camera")
frame_placeholder = st.empty()

if run:
    while True:
        ret, frame = cam.read()
        if not ret:
            st.warning("Camera not available")
            break

        face = detect_face(frame)
        if face is not None:
            cv2.rectangle(frame, (0, 0), (250, 40), (0, 255, 0), -1)
            if mode == "Register":
                name = st.text_input("Enter your name")
                if st.button("Register"):
                    emb = get_embedding(face)
                    save_embedding(name, emb)
                    st.success("Face Registered Successfully")
            else:
                db = load_embeddings()
                emb = get_embedding(face)
                match_name, sim = match_face(emb, db)
                if match_name:
                    cv2.putText(frame, f"Welcome {match_name}!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                else:
                    cv2.putText(frame, "Access Denied", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cam.release()
    cv2.destroyAllWindows()
