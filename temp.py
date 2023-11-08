from l2cs import Pipeline, render
import cv2
import torch

import pandas as pd

# import timeit

gaze_pipeline = Pipeline(
    weights='models/L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu') # or 'gpu'
    # device=torch.device('mps') # or 'gpu'
)

out = []
for i in range(20):
    img = cv2.imread('test_image.jpg')
    frame = cv2.resize(img, (0,0), fx=0.125, fy=0.125)
    results = gaze_pipeline.step(frame)
    if len(results.scores) > 1:
        print("MORE THAN 1 FACE")
        continue
    out.append({
        'landmarks': results.landmarks[0],
        'bboxes': results.bboxes[0],
        'pitch': results.pitch[0],
        'yaw': results.yaw[0],
        'score': results.scores[0],
    })

print(pd.DataFrame(out))

# print(dir(results))

# print(results.landmarks, results.pitch, results.yaw)

# print(timeit.repeat("test()", setup="from __main__ import test", number=20, repeat=5))

# Process frame and visualize
# frame = render(frame, results)
