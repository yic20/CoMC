# import openai
from openai import OpenAI
import json

client = OpenAI(
    base_url='https://api.openai-proxy.org',
    api_key='YOUR_OPENAI_API_KEY',
)





import random

label_set = ['person', 'bicycle', 'car', 'motor bike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def generate_description(labels):
    prompt = f"Please make a brief sentence to describe a photo that contains {', '.join(labels)}."
    
    
    ###  use the latest gpt-3.5-turbo for generation
    chat_completion = client.chat.completions.create(      
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    max_tokens=40,
    model="gpt-3.5-turbo", 
)
    ###  use gpt3-davinci-002 (used in original paper)
    # response = client.completions.create(       
    #     model="text-davinci-002",
    #     prompt=prompt,
    #     max_tokens=30
    # )
    return chat_completion.choices[0].message.content.strip()

def sample_labels(label_set, max_labels=5):
    num_labels = random.randint(1, max_labels)
    return random.sample(label_set, num_labels)

def generate_data(label_set, num_samples=40000, max_labels=5):
    data = []
    for _ in range(num_samples):
        labels = sample_labels(label_set, max_labels)
        description = generate_description(labels)
        data.append({'labels': labels, 'description': description})
    return data



def save_to_json(data, filename='data.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

data = generate_data(label_set, num_samples=40000, max_labels=5)

save_to_json(data, 'data_coco_40k.json')

for i in range(5):
    print(data[i])
