
from io import BytesIO

import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRMedium, RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRNano()

response = requests.get("https://media.roboflow.com/dog.jpg", timeout=30)
response.raise_for_status()
image = Image.open(BytesIO(response.content)).convert("L").convert("RGB")
image = Image.open("/home/ronbar/repo/datasets/drone-dataset-(uav)-DatasetNinja/valid/img/0014.jpg")
detections = model.predict(image, threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)


annotated_image.save("annotated_image.jpg")
