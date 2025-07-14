import imagehash
from PIL import Image
import hashlib

placeholder_hash = imagehash.phash(
    Image.open("/mnt/castle/processed/placeholder.png").convert("L")
)  # Load the placeholder image and compute its hash



def is_similar_placeholder(frame_rgb, threshold=1):
    """
    frame_rgb: RGB frame to compare against the placeholder.
    If read from cv.imread, it should be converted to RGB:

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    Returns True if the frame is similar to the placeholder image.
    """
    frame_hash = imagehash.phash(Image.fromarray(frame_rgb))
    return frame_hash - placeholder_hash < threshold

def get_frame_key(frame):
    return str(frame.tobytes())


def is_Image_similar_placeholder(frame, threshold=1):
    """
    frame_rgb: RGB frame to compare against the placeholder.
    If read from cv.imread, it should be converted to RGB:

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    Returns True if the frame is similar to the placeholder image.

    """
    frame_hash = imagehash.phash(frame)
    similar = frame_hash - placeholder_hash < threshold
    return similar


