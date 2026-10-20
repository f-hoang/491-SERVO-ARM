import cv2


def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam, or a file path like "clip.mp4"
    if not cap.isOpened():
        raise SystemExit("Could not open video source")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1. Preprocess: gray -> blur -> threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Draw the ones big enough to matter
        for c in contours:
            if cv2.contourArea(c) < 800:
                continue
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 1)

        cv2.imshow("contours", frame)
        cv2.imshow("mask", mask)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # q or Esc
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()