import cv2
from style_transfer import get_image_details, superpixelate

cap = cv2.VideoCapture('./data/video.MOV')

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 

size = (frame_width, frame_height) 

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('prova.mp4', fourcc, 30.0, (frame_width,frame_height))

        

while cap.isOpened():
    ret, frame = cap.read()
    frame = get_image_details(frame)
    out.write(frame)
    cv2.imshow('window-name', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release() 
cv2.destroyAllWindows() 