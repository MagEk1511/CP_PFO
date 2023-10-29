from ultralytics import YOLO
import cv2
import os

# load yolov8 model
model = YOLO('best.pt')
#.cuda()
# load video
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)
video_out_path = os.path.join('.', 'input.mp4')

ret, frame = cap.read()
size = (int(cap.get(3)), int(cap.get(4)))  # Get the size from the video capture
cap_out = cv2.VideoWriter(video_out_path,  
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         cap.get(cv2.CAP_PROP_FPS), size) 
ret = True
# read frames

while ret:
    ret, frame = cap.read()

    if ret:
        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        cap_out.write(frame_)


results = model.track(video_path, show=True)

d = {0: 0, 1: 0, 2: 0, 3: 0}
dd = {}
for i in results:
    for j, uu in enumerate(i.boxes.id):
        #print(j)
        dd[int(i.boxes.id[j].item())] = int(i.boxes.cls[j].item())

cc = [0]*4

for x, y in dd.items():
    cc[y] += 1
print(cc)
cap_out.release()
cv2.destroyAllWindows()
