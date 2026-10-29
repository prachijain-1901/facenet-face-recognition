# FaceNet Face Recognition Web App

A Streamlit frontend for the FaceNet + triplet-loss model from `DL Mini project facenet.ipynb`.

## Features
- Upload JPG/JPEG/PNG photos.
- Detect one or more faces in the uploaded image.
- Compare normalized FaceNet embeddings with saved class centroids.
- Show predicted member names or `Unidentified`, with bounding boxes and distances.
- Loads the existing `facenet_model.pt`; it does not retrain the model.

## Required project files

Place these files in one folder:

```text
Group_x_code/
├── app.py
├── requirements.txt
├── facenet_model.pt
└── README.md
```

The notebook and `data/` folder may remain in the project for submission, but the web app inference needs the saved checkpoint. Do not upload private photos or a checkpoint to a public repository unless your group has permission to share them.

## Run locally on Windows

Open Command Prompt in this folder and run:

```cmd
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository for this app.
2. Add `app.py`, `requirements.txt`, and `README.md`.
3. Add `facenet_model.pt` only if its size is within GitHub's file limits and your group is allowed to publish it. If it is too large, use Git LFS or a private model-storage option and adapt the loading path; do not commit secrets.
4. Push the repository to GitHub.
5. Open Streamlit Community Cloud, sign in with GitHub, choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
6. Deploy and wait for dependency installation and model loading to finish.
7. Open the app URL and test with a few consented sample images.

## Deployment notes
- The first deployment can take several minutes because PyTorch and FaceNet dependencies are large.
- The free host may have limited RAM/CPU; model loading or inference may be slow.
- A hosted web app cannot access a visitor's physical webcam like a local OpenCV script. This frontend therefore uses image upload.
- The saved checkpoint must contain `state`, `classes`, `centroids`, and `threshold`, matching the notebook's save cell.
- Tune the distance threshold using validation images. Raising it may increase false matches; lowering it may reject valid members.

## Project description

This project implements a FaceNet-based face recognition system using transfer learning and triplet-loss fine-tuning. Images are preprocessed using MTCNN face detection and alignment, passed through an InceptionResnetV1 backbone to produce normalized 512-dimensional face embeddings, and compared with stored class centroids using Euclidean distance. If the nearest centroid is within the configured distance threshold, the app displays the corresponding enrolled member name; otherwise, it labels the face as `Unidentified`. The Streamlit interface allows users to upload an image and view face bounding boxes, predictions, and similarity distances. The app uses a previously trained checkpoint, so model training is not repeated during normal use.
