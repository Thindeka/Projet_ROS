import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
LINEAR_SPEED  = 0.15
ANGULAR_SPEED = 0.5
ANGLE_DEAD_ZONE = 30
STREAM_URL = "http://192.168.0.65:8080/video"
# ---------------------------------------------------------------------------


class IndexTeleop(Node):
    def __init__(self):
        super().__init__('index_teleop')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands      = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(STREAM_URL)

        # Le stream réseau peut mettre quelques secondes à s'ouvrir
        if not self.cap.isOpened():
            self.get_logger().warn('Stream non disponible immédiatement, on continue quand même...')

        self.timer = self.create_timer(0.05, self.timer_callback)

        print('[INIT] Téléopération par index')
        print('[INIT] Index vers HAUT    → AVANCER')
        print('[INIT] Index vers BAS     → RECULER')
        print('[INIT] Index vers GAUCHE  → TOURNER GAUCHE')
        print('[INIT] Index vers DROITE  → TOURNER DROITE')
        print('[INIT] Poing / rien       → STOP')

    # -----------------------------------------------------------------------
    def _index_direction(self, hand_landmarks, w, h):
        lm = hand_landmarks.landmark

        base = lm[5]
        tip  = lm[8]
        mid  = lm[6]

        dx = (tip.x - base.x) * w
        dy = (tip.y - base.y) * h

        length = np.sqrt(dx**2 + dy**2)
        if length < 30:
            return 'STOP', 0

        angle = np.degrees(np.arctan2(dy, dx))

        if   -135 < angle < -45:  direction = 'FORWARD'
        elif   45 < angle < 135:  direction = 'BACKWARD'
        elif  angle > 135 or angle < -135:  direction = 'LEFT'
        elif  -45 <= angle <= 45:  direction = 'RIGHT'
        else:  direction = 'STOP'

        return direction, angle

    # -----------------------------------------------------------------------
    def timer_callback(self):
        ret, frame = self.cap.read()

        # Si la frame échoue, tenter de reconnecter le stream
        if not ret:
            self.get_logger().warn('Frame non reçue, tentative de reconnexion...')
            self.cap.release()
            self.cap = cv2.VideoCapture(STREAM_URL)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        twist     = Twist()
        command   = 'STOP'
        angle_deg = 0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(
                frame, hand, self.mp_hands.HAND_CONNECTIONS)

            command, angle_deg = self._index_direction(hand, w, h)

            tip  = hand.landmark[8]
            base = hand.landmark[5]
            p1 = (int(base.x * w), int(base.y * h))
            p2 = (int(tip.x  * w), int(tip.y  * h))
            cv2.arrowedLine(frame, p1, p2, (0, 255, 255), 3, tipLength=0.3)

        if command == 'FORWARD':
            twist.linear.x  =  LINEAR_SPEED
        elif command == 'BACKWARD':
            twist.linear.x  = -LINEAR_SPEED
        elif command == 'LEFT':
            twist.angular.z =  ANGULAR_SPEED
        elif command == 'RIGHT':
            twist.angular.z = -ANGULAR_SPEED

        self.cmd_pub.publish(twist)

        color_map = {
            'FORWARD':  (0, 255, 0),
            'BACKWARD': (0, 0, 255),
            'LEFT':     (255, 165, 0),
            'RIGHT':    (255, 165, 0),
            'STOP':     (128, 128, 128),
        }
        color = color_map.get(command, (255, 255, 255))

        cv2.putText(frame, f'CMD: {command}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(frame, f'angle: {angle_deg:.1f} deg', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        legend = [
            'Index HAUT   → AVANCER',
            'Index BAS    → RECULER',
            'Index GAUCHE → TOURNER G',
            'Index DROITE → TOURNER D',
            'Poing/rien   → STOP',
        ]
        for i, txt in enumerate(legend):
            cv2.putText(frame, txt, (10, h - 20 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Index Teleop', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    # -----------------------------------------------------------------------
    def _publish_stop(self):
        self.cmd_pub.publish(Twist())


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = IndexTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
