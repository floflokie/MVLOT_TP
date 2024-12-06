import cv2
from TP_1.KalmanFilter import KalmanFilter
from TP_1.Detector import detect

# Create the object of the class Kalman filter
kalmanFilter = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_sdt_meas=0.1, y_sdt_meas=0.1)

# Create video capture object
video_capture = cv2.VideoCapture("randomball.avi")
trajectory = []

# Object Detection Integration
while True:
    # Read the video frame
    ret, frame = video_capture.read()
    if not ret or not frame.any():
        break

    # Detect the object
    centroid = detect(frame)

    # If centroid is detected then track it. Call the Kalman prediction function and the Kalman filter updating function
    if len(centroid) != 0:
        single_object = centroid[0]
        (x, y) = single_object
        estimated = kalmanFilter.update(single_object)
        predicted = kalmanFilter.predict()
        (predicted_x, predicted_y) = predicted
        (estimated_x, estimated_y) = estimated
        # Visualize for tracking results
        radius = 20
        thickness = 2
        # Draw detected circle (green color)
        green = (0, 255, 0)
        cv2.circle(img=frame, center=(int(x), int(y)), radius=radius, color=green, thickness=thickness)
        # Draw a blue rectangle as the predicted object position
        blue = (255, 0, 0)
        cv2.rectangle(img=frame, pt1=(int(predicted_x - radius), int(predicted_y - radius)),
                      pt2=(int(predicted_x + radius), int(predicted_y + radius)), color=blue, thickness=thickness)
        # Draw a red rectangle as the estimated object position
        red = (0, 0, 255)
        cv2.rectangle(img=frame, pt1=(int(estimated_x - radius), int(estimated_y - radius)),
                      pt2=(int(estimated_x + radius), int(estimated_y + radius)), color=red, thickness=thickness)
        cv2.putText(img=frame, text="Estimated Pos",
                    org=(int(estimated_x + radius), int(estimated_y + radius)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=red, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img=frame, text="Predicted Pos",
                    org=(int(predicted_x + radius), int(predicted_y + radius)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=blue, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img=frame, text="Measured Pos",
                    org=(int(x + radius), int(y + radius)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=green, thickness=1, lineType=cv2.LINE_AA)
        # Draw the trajectory (tracking path ) in an image
        trajectory.append((predicted_x, predicted_y))
        for point in trajectory:
            cv2.circle(img=frame, center=(int(point[0]), int(point[1])), radius=2, color=(255, 0, 255), thickness=-1)
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(10)

video_capture.release()
cv2.destroyAllWindows()