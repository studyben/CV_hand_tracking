import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelCompex = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCompex = modelCompex
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.modelCompex, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw =True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                print(handLms)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, pos=0, handnum = 0):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        position = []

        if results.multi_hand_landmarks:
            hand = results.muti_hand_landmarks[0]:
            for id, pos in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(w*pos.x), int(h*pos.y)
                position.append([id, cx, cy])
        return position

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        if success:
            cv2.imshow("Image", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()