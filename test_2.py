import torch
import json
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.express as px
import plotly.io as pio
import subprocess
import tempfile
import os
import hdbscan
import umap

from utils.token_manager import get_hf_token

class PredictionHead(torch.nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048, dtype=torch.bfloat16):
        super(PredictionHead, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            "avinot/LoLlama-3.2-1B-lora-3ep-v3-instruct", 
            token=get_hf_token("read"),
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "avinot/LoLlama-3.2-1B-lora-3ep-v3-instruct", 
            token=get_hf_token("read")
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.proj = torch.nn.Linear(input_dim, output_dim, dtype=dtype).to(self.device)

    def forward(self, input_text : list[str]) -> torch.Tensor:
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.backbone(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (layer_count, B, T, D)
            last_k_layers = hidden_states[-10:]
            stacked = torch.stack(last_k_layers)  # (k, B, T, D)
            mean_k_layers = stacked.mean(dim=0)   # (B, T, D)

            attention_mask = inputs["attention_mask"].unsqueeze(-1).to(self.device)
            masked_mean = (mean_k_layers * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        # Linear projection and L2 normalization
        projected = self.proj(masked_mean)
        normalized = torch.nn.functional.normalize(projected, p=2, dim=1)
            
        return normalized

model : PredictionHead = PredictionHead(input_dim=2048, output_dim=2048)
model.load_state_dict(torch.load("output/model_epoch_1_step_5001.pth", map_location=torch.device('cpu')))
embeddings : list[np.ndarray] = []
with open('./champion_mapping.json', 'r') as f:
    data : list[dict] = json.load(f)

champion_names : list[str] = [champion['name'] for champion in data]
for i, champion in enumerate(tqdm(champion_names)):
    embedding : torch.Tensor = model(champion).detach().cpu().to(torch.float32)
    embedding_array = embedding.numpy()
    embeddings.append(embedding_array)

embeddings_array = np.vstack(embeddings)
# # 2 Perform K-Means in the embedding space
# kmeans = KMeans(n_clusters=min(13, len(embeddings)), random_state=42)
# clusters = kmeans.fit_predict(embeddings_array)

# # 3 PCA into 3 dimensions
# pca = PCA(n_components=3)
# reduced_embeddings = pca.fit_transform(embeddings_array)

# # 2 Perform HDBSCAN clustering in the high-dimensional space
# clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
# clusters = clusterer.fit_predict(embeddings_array)

# # 3 UMAP into 3 dimensions for visualization
# umap_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
# reduced_embeddings = umap_3d.fit_transform(embeddings_array)

# # 4 Plot the clusters using plotly
# fig = px.scatter_3d(
#     x=reduced_embeddings[:, 0],
#     y=reduced_embeddings[:, 1],
#     z=reduced_embeddings[:, 2],
#     color=clusters,
#     text=champion_names,
#     title="Champion Clusters in 3D PCA Space"
# )

# 2 Perform HDBSCAN clustering in the high-dimensional space
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
clusters = clusterer.fit_predict(embeddings_array)

# 3 UMAP into 3 dimensions for visualization
umap_3d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
reduced_embeddings = umap_3d.fit_transform(embeddings_array)

# 4 Plot the clusters using plotly
fig = px.scatter(
    x=reduced_embeddings[:, 0],
    y=reduced_embeddings[:, 1],
    color=clusters.astype(str),
    text=champion_names,
    title="Champion Clusters in 3D PCA Space"
)

# Configure browser opening for WSL
def convert_to_windows_path(wsl_path):
    """Convert a WSL path to a Windows path"""
    try:
        # Use wslpath to convert the path
        result = subprocess.run(
            ["wslpath", "-w", wsl_path],
            check=True,
            capture_output=True,
            text=True
        )
        windows_path = result.stdout.strip()
        return windows_path
    except subprocess.SubprocessError:
        print(f"Failed to convert path {wsl_path} to Windows path")
        return None

def open_with_wslview(url_or_path):
    """Open a URL or file path in Windows browser from WSL"""
    try:
        # If it's a file path that starts with file://
        if url_or_path.startswith("file://"):
            file_path = url_or_path[7:]  # Remove file:// prefix
            windows_path = convert_to_windows_path(file_path)
            if windows_path:
                # Use explorer.exe directly for files
                subprocess.run(["explorer.exe", windows_path], check=True)
                return True
        else:
            # For http/https URLs use wslview if available
            try:
                subprocess.run(["which", "wslview"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subprocess.run(["wslview", url_or_path], check=True)
                return True
            except subprocess.SubprocessError:
                # If it's a URL and wslview isn't available, use powershell
                url_windows = url_or_path.replace('localhost', '127.0.0.1')
                subprocess.run(["powershell.exe", "-Command", f"Start-Process '{url_windows}'"], check=True)
                return True
    except Exception as e:
        print(f"Failed to open browser. Error: {e}")
        return False

# Save to a file in the home directory (which is accessible from Windows)
home_dir = os.path.expanduser("~")
temp_file = os.path.join(home_dir, "plotly_figure.html")
fig.write_html(temp_file)
print(f"Figure saved to {temp_file}")

# Open the file in the Windows browser
open_with_wslview(f"file://{temp_file}")


# Save the clusters to output.json
output = {}
for idx, cluster in enumerate(clusters):
    # Convert NumPy int32 to regular Python int for use as dictionary key
    cluster_key = int(cluster)
    if cluster_key not in output:
        output[cluster_key] = []
    output[cluster_key].append(champion_names[idx])

output = dict(sorted(output.items()))

with open('output.json', 'w') as f:
    json.dump(output, f, indent=4)
