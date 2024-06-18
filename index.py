from flask import Flask, request, render_template, send_file
from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor
from PIL import Image, ImageDraw, ImageFont
import io

app = Flask(__name__)

# Load model and processor
model = LayoutLMForTokenClassification.from_pretrained("philschmid/layoutlm-funsd")
processor = LayoutLMv2Processor.from_pretrained("philschmid/layoutlm-funsd")

# Helper function to unnormalize bounding boxes
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]
label2color = {
    "B-HEADER": "blue",
    "B-QUESTION": "red",
    "B-ANSWER": "green",
    "I-HEADER": "blue",
    "I-QUESTION": "red",
    "I-ANSWER": "green",
}
# Draw bounding boxes and labels on the image
def draw_boxes(image, boxes, predictions):
    width, height = image.size
    normalized_boxes = [unnormalize_box(box, width, height) for box in boxes]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalized_boxes):
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image

# Define route for uploading image
@app.route('/')
def index():
    return render_template('index.html')

# Define route for inference
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return "No file uploaded"
        
        file = request.files['file']
        
        # Check if the file is an image
        if file.filename == '':
            return "No file selected"
        
        # Read the image and run inference
        img = Image.open(file).convert('RGB')
        encoding = processor(img, return_tensors="pt")
        del encoding["image"]
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        labels = [model.config.id2label[prediction] for prediction in predictions]
        processed_img = draw_boxes(img, encoding["bbox"][0], labels)
        
        # Save the processed image to a byte buffer
        img_byte_array = io.BytesIO()
        processed_img.save(img_byte_array, format='PNG')
        img_byte_array.seek(0)
        
        # Return the processed image
        return send_file(img_byte_array, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)