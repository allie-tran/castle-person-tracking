from collections import Counter
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

preprocess_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = InceptionResnetV1(
            pretrained="vggface2",
            classify=False
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        emb = self.backbone(x)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

class KNNFaceRecognizer:
    def __init__(self, embeddings, labels, 
                 index_to_label,
                 k=5):
        self.embeddings = F.normalize(embeddings, dim=1)
        self.labels = labels
        self.k = k
        self.index_to_label = index_to_label

    @torch.no_grad()
    def predict(self, embedding, sim_threshold=0.5):
        if not isinstance(embedding, torch.Tensor):
            emb = torch.tensor(embedding)
        else:
            emb = embedding.clone()

        emb = F.normalize(emb, dim=0)

        sims = torch.mv(self.embeddings, emb)
        topk = torch.topk(sims, self.k)

        top_labels = self.labels[topk.indices]
        vote = Counter(top_labels.tolist()).most_common(1)[0]
        pred_label, count = vote

        best_sim = topk.values[0].item()
        confidence = count / self.k

        if best_sim < sim_threshold:
            return None, {
                "best_sim": best_sim,
                "confidence": confidence,
                "neighbors": top_labels.tolist()
            }
        pred_label = self.index_to_label.get(pred_label, None)
        return pred_label, {
            "best_sim": best_sim,
            "confidence": confidence,
            "neighbors": top_labels.tolist()
        }

model = FaceEmbeddingModel()
model.load_state_dict(torch.load("models/face_reid.pth"))
model.eval()
model.to("cuda")

knn_ckpt = torch.load("models/knn_face_model.pt", map_location="cpu")
knn_model = KNNFaceRecognizer(
    embeddings=knn_ckpt["embeddings"],
    labels=knn_ckpt["labels"],
    k=knn_ckpt["k"],
    index_to_label=knn_ckpt["index_to_label"]
)

PEOPLE_NAMES = list(knn_model.index_to_label.values())

def get_face_embedding_from_crop(
    face,
    device="cuda"
):
    """
    face: cv2 croped image
    returns: 512D numpy embedding or None
    """
    if isinstance(face, Image.Image):
        img = face
    else:
        img = Image.fromarray(face)

    img = preprocess_transforms(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
        emb = F.normalize(emb, dim=1)[0]

    return emb.cpu().numpy()


def classify_face_embedding(
    embedding,
    sim_threshold=0.5,
    logging=False
):
    """
    embedding: 512D numpy array
    prototypes: dict[label -> torch.Tensor or numpy]
    returns: (pred_label or None, similarity_dict)
    """
    emb = torch.tensor(embedding)
    pred_label, info = knn_model.predict(emb, sim_threshold=sim_threshold)
    if logging:
        print(f"Best sim for {pred_label}: {info['best_sim']:.4f}")
    return pred_label, info
