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
    

model = FaceEmbeddingModel()
model.load_state_dict(torch.load("best_face_reid_model.pth"))
model.eval()
model.to("cuda")
prototypes = torch.load("face_reid_prototypes.pth")
PEOPLE_NAMES = list(prototypes.keys())
NUM_KNOWN_FACES = len(prototypes)

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

    sims = {}
    for lbl, proto in prototypes.items(): 
        proto = proto if isinstance(proto, torch.Tensor) else torch.tensor(proto)
        sims[lbl] = torch.dot(emb, proto).item()
    
    pred_label = max(sims, key=sims.get) # already the name
    best_sim = sims[pred_label]
    if logging:
        print(f"Best sim for {pred_label}: {best_sim:.4f}")
    sims = list(sims.values())

    if best_sim < sim_threshold:
        return None, sims

    return pred_label, sims
