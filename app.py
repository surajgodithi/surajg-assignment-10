from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
import pandas as pd
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
from sklearn.decomposition import PCA

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
IMAGE_FOLDER = "coco_images_resized"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["IMAGE_FOLDER"] = IMAGE_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and data
model, _, preprocess = create_model_and_transforms("ViT-B/32", pretrained="openai")
model.eval()
df = pd.read_pickle("image_embeddings.pickle")
embeddings = torch.stack([torch.tensor(row["embedding"]) for _, row in df.iterrows()])


def apply_pca(embeddings, query, n_components):
    """
    Perform PCA dimensionality reduction on embeddings and query.
    """
    mean = torch.mean(embeddings, dim=0)
    embeddings_centered = embeddings - mean
    query_centered = query - mean

    # Compute PCA manually using SVD
    u, s, v = torch.svd(embeddings_centered)
    reduced_embeddings = torch.mm(embeddings_centered, v[:, :n_components])
    reduced_query = torch.mm(query_centered, v[:, :n_components])
    return reduced_embeddings, reduced_query


def calculate_similarity(query_embedding, top_k=5, use_pca=False, n_components=5):
    """
    Compute top-k most similar images using cosine similarity.
    """
    if use_pca:
        embeddings_reduced, query_reduced = apply_pca(embeddings, query_embedding, n_components)
        similarities = torch.matmul(query_reduced, embeddings_reduced.T)
    else:
        similarities = torch.matmul(query_embedding, embeddings.T)

    top_indices = torch.topk(similarities, top_k).indices
    results = [
        (
            os.path.join("/coco_images_resized", df.iloc[idx]["file_name"]),
            similarities[0, idx].item(),
        )
        for idx in top_indices
    ]
    return results


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route for the application.
    Handles text, image, and hybrid queries.
    """
    results = []
    if request.method == "POST":
        query_type = request.form["query_type"]
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5))
        use_pca = "use_pca" in request.form
        n_components = int(request.form.get("n_components", 5))

        image_query_path = None
        if "image_query" in request.files:
            file = request.files["image_query"]
            if file:
                filename = secure_filename(file.filename)
                image_query_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(image_query_path)

        try:
            # Text query
            if query_type == "text_query":
                text_query = request.form.get("text_query", "").strip()
                text_embedding = F.normalize(model.encode_text(tokenizer.tokenize([text_query]).to("cpu")))
                results = calculate_similarity(text_embedding, use_pca=use_pca, n_components=n_components)

            # Image query
            elif query_type == "image_query" and image_query_path:
                image = preprocess(Image.open(image_query_path)).unsqueeze(0)
                image_embedding = F.normalize(model.encode_image(image))
                results = calculate_similarity(image_embedding, use_pca=use_pca, n_components=n_components)

            # Hybrid query
            elif query_type == "hybrid_query" and image_query_path:
                text_query = request.form.get("text_query", "").strip()
                text_embedding = F.normalize(model.encode_text(tokenizer.tokenize([text_query]).to("cpu")))
                image = preprocess(Image.open(image_query_path)).unsqueeze(0)
                image_embedding = F.normalize(model.encode_image(image))
                hybrid_embedding = F.normalize(hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding)
                results = calculate_similarity(hybrid_embedding, use_pca=use_pca, n_components=n_components)

        except Exception as e:
            print(f"Error processing query: {e}")
            results = []

    return render_template("index.html", results=results)


@app.route("/coco_images_resized/<path:filename>")
def serve_images(filename):
    """
    Serve images from the resized dataset folder.
    """
    return send_from_directory(app.config["IMAGE_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
