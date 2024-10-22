import cv2
import torch
import torchvision
from torchvision import transforms
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_prediction(frame, threshold):
    img = transform(frame)
    img = img.to(device)

    with torch.no_grad():
        predictions = model([img])

    # Move predictions to CPU
    predictions = [{k: v.to('cpu') for k, v in t.items()} for t in predictions]

    pred_scores = predictions[0]['scores'].numpy()
    pred_bboxes = predictions[0]['boxes'].numpy()
    pred_labels = predictions[0]['labels'].numpy()

    filtered_indices = [i for i, score in enumerate(pred_scores) if score > threshold]
    filtered_bboxes = pred_bboxes[filtered_indices]
    filtered_labels = pred_labels[filtered_indices]
    filtered_scores = pred_scores[filtered_indices]

    return filtered_bboxes, filtered_labels, filtered_scores

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fps_start_time = time.time()
    fps_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes, labels, scores = get_prediction(rgb_frame, threshold=0.7)
        for bbox, label, score in zip(bboxes, labels, scores):
            x_min, y_min, x_max, y_max = bbox.astype(int)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
            label_text = f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}"
            cv2.putText(frame, label_text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1:
            fps = fps_frame_count / elapsed_time
            fps_text = f"FPS: {fps:.2f}"
            fps_start_time = time.time()
            fps_frame_count = 0
        cv2.putText(frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Real-Time Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()