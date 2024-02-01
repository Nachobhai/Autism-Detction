import time
import cv2
import mediapipe as mp
import numpy as np

"""This class is for body movements. It is using Pose module as backend.
The output is movement_log1.txt which will contain x,y,z co=ordinates and averaging(capturing 60frames per second) is done in movement_log.txt the average is calculated every second"""

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

# Initialize MediaPipe Pose module
pose = mpPose.Pose()

# cap = cv2.VideoCapture(0) #for livefeed from camera
cap = cv2.VideoCapture("PoseVideos/1    .mp4")

pTime = 0

# Define variables for movement detection
prev_coordinates = [None] * 33  # List to store previous coordinates of each body part
movement_threshold = 0  # Adjust this value as per your requirement
time_constraint = 5  # Time constraint in seconds

start_time = time.time()
movement_data = [[] for _ in range(33)]  # List to store movement coordinates for each body part
output_file = open("movement_log.txt", "a")
output_file1 = open("movement_log1.txt", "a")
frame_rate = cap.get(cv2.CAP_PROP_FPS)

print("Frame rate:", frame_rate)


previous_average_coordinates = [None] * 33  # List to store the average coordinates of previous second

def average_time(id, cx, cy, current_time):
    if current_time >= 1.0:  # Calculate the difference between consecutive second averages
        if len(movement_data[id]) > 0:
            current_average_coordinates = np.mean(movement_data[id], axis=0)
            previous_average_coordinates = np.mean(movement_data[id][-1], axis=0)#-1 for averaging at 1 second, -2 for 2 seconds, -3 for 3 secondsl
            if previous_average_coordinates is not None:
                difference = current_average_coordinates - previous_average_coordinates
                output_file.write(f"At time {round(current_time)}, Average Movement {id}: {current_average_coordinates}, Movement Difference of Body Part {id}: {difference}\n")
            previous_average_coordinates = current_average_coordinates

        movement_data[id] = []  # Reset movement data

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id in range(33):
            lm = results.pose_landmarks.landmark[id] if id < len(results.pose_landmarks.landmark) else None
            output_file1.write(f"{id, lm}")
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h) if lm is not None else (0, 0)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # Calculate movement distance for each body part
            if prev_coordinates[id] is not None:
                prev_x, prev_y = prev_coordinates[id]
                movement_distance = abs(cx - prev_x) + abs(cy - prev_y)
                current_time = time.time() - start_time

                if movement_distance > movement_threshold and current_time >= time_constraint:
                    movement_data[id].append((cx, cy))  # Store movement coordinates
                    average_time(id, cx, cy, current_time)

            # Update previous coordinates for each body part
            prev_coordinates[id] = (cx, cy)


    # not necessary
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

output_file1.close()
output_file.close()  # Close the output file
