import cv2
import numpy as np

cam_bot = cv2.VideoCapture(1)
cam_top = cv2.VideoCapture(0)
#cam_bot.set(cv2.CAP_PROP_FPS, 120)
#cam_top.set(cv2.CAP_PROP_FPS, 120)
#print(cam_bot.get(cv2.CAP_PROP_FPS))
#print(cam_top.get(cv2.CAP_PROP_FPS))

# define affine transformation matrix
# M_top = np.float32([
# 	[1, 0, -4
# ],
# 	[0, 1, 0]
# ])

# M_bot = np.float32([
# 	[1, 0, -4.5],
# 	[0, 1, +1.9]
# ])

M_top = np.float32([
	[1, 0, 0.0],
	[0, 1, 1.1]
])

M_bot = np.float32([
	[1, 0, -1.9],
	[0, 1, -5.7]
])

while cam_bot.isOpened():    
    # frame_top = cv2.cvtColor(cam_top.read()[1], cv2.COLOR_BGR2RGB)
    # frame_bot = cv2.cvtColor(cam_bot.read()[1], cv2.COLOR_BGR2RGB)
    frame_top = cam_top.read()[1]
    # shifted_top = cv2.warpAffine(frame_top, M_top, (frame_top.shape[1], frame_top.shape[0]))
    frame_bot = cam_bot.read()[1]
    # shifted_bot = cv2.warpAffine(frame_bot, M_bot, (frame_bot.shape[1], frame_bot.shape[0]))
    # Display the resulting frame
    cv2.imshow("Top Camera", frame_top)
    # cv2.imshow("Shifted Top Camera", shifted_top)
    cv2.imshow("Bottom Camera", frame_bot)
    # cv2.imshow("Shifted Bottom Camera", shifted_bot)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.imwrite('./top_test.jpg', frame_top)
# cv2.imwrite('./shifted_top_test.jpg', shifted_top)
# cv2.imwrite('./bot_test_cylinder.jpg', frame_bot)
# cv2.imwrite('./shifted_bot_test.jpg', shifted_bot)
# When everything done, release the capture
cam_top.release()
cam_bot.release()
cv2.destroyAllWindows()

# import cv2
# import time

# cam_bot = cv2.VideoCapture(2)
# # cam_bot.set(cv2.CAP_PROP_FPS, 120)
# # print(cam_bot.get(cv2.CAP_PROP_FPS))

# while cam_bot.isOpened():    
#     frame_bot = cv2.cvtColor(cam_bot.read()[1], cv2.COLOR_BGR2RGB)
#     # Display the resulting frame
#     cv2.imshow("Bottom Camera", frame_bot)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Number of frames to capture
# num_frames = 120
# print("Capturing {0} frames".format(num_frames))
# # Start time
# start = time.time()
# # Grab a few frames
# for i in range(0, num_frames):
#     ret, frame = cam_bot.read()
# # End time
# end = time.time()
# # Time elapsed
# seconds = end - start
# print ("Time taken : {0} seconds".format(seconds))
# # Calculate frames per second
# fps  = num_frames / seconds
# print("Estimated frames per second : {0}".format(fps))
# # When everything done, release the capture
# cam_bot.release()
# cv2.destroyAllWindows()
