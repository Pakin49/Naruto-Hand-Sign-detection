import cv2
def test_video_capture():

    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))

    frame_num = 0
    types = input("Hand sign types:")
    while True:
        ret, frame = cam.read()

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv2.imshow("Camera", frame)
        path = "data/my_dataset/"+types+str(frame_num)+".png"
        cv2.imwrite(path, frame)
        frame_num += 1
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    print("Hello from naruto-hand-sign-detection!")
    test_video_capture()


if __name__ == "__main__":
    main()
