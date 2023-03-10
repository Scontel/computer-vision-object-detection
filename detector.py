import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDetector:
    def __init__(self, threshold=0.8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights).to(self.device)
        self.model.eval()
        self.transforms = self.weights.transforms()
        self.threshold = threshold
        self.categories = self.weights.meta["categories"]

    def detect(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
            
        boxes = predictions['boxes'][predictions['scores'] > self.threshold]
        labels = predictions['labels'][predictions['scores'] > self.threshold]
        scores = predictions['scores'][predictions['scores'] > self.threshold]
        
        return image, boxes, labels, scores

    def visualize(self, image, boxes, labels, scores, output_path="output.png"):
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            category_name = self.categories[label.item()]
            ax.text(x1, y1, f'{category_name}: {score:.2f}', color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
            
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
