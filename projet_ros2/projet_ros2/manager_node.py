import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import subprocess
import time

# ---------------------------------------------------------------------------
CHALLENGES = [
    ['ros2', 'run', 'projet_ros2', 'challenge1_follow'],  
    ['ros2', 'run', 'projet_ros2', 'challenge2_avoid'],  
    ['ros2', 'run', 'projet_ros2', 'corridor'],   
    ['ros2', 'run', 'projet_ros2', 'goal_node'],    
]

# Détection ligne bleue
BLUE_LOWER   = np.array([100, 80, 50])
BLUE_UPPER   = np.array([130, 255, 255])
BLUE_MIN_AREA       = 3000
BLUE_CONFIRM_FRAMES = 8

ROI_NEAR_TOP = 0.60
ROI_NEAR_BOT = 0.80
kernel = np.ones((5, 5), np.uint8)


class ChallengeManager(Node):

    def __init__(self):
        super().__init__('challenge_manager')

        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.image_callback, 10)

        self.challenge_index = 0
        self.blue_counter    = 0
        self.in_transition   = False
        self.current_process = None
        self._transition_timer = None

        self._launch_current()

    # -----------------------------------------------------------------------
    def _launch_current(self):
        if self.challenge_index >= len(CHALLENGES):
            self.get_logger().info('[MANAGER] Tous les challenges terminés.')
            return
        cmd = CHALLENGES[self.challenge_index]
        self.get_logger().info(
            f'[MANAGER] >>> Challenge {self.challenge_index + 1}/{len(CHALLENGES)} : {" ".join(cmd)}')
        self.current_process = subprocess.Popen(cmd)

    def _next_challenge(self):
        # Kill le node courant
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
            self.get_logger().info('[MANAGER] Node précédent arrêté.')

        self.challenge_index += 1
        time.sleep(0.5)
        self._launch_current()

        # Bloque la détection pendant 3s pour traverser la ligne bleue
        self.in_transition = True
        self.blue_counter  = 0
        if self._transition_timer is not None:
            self._transition_timer.cancel()
        self._transition_timer = self.create_timer(3.0, self._end_transition)

    def _end_transition(self):
        self.in_transition = False
        self.blue_counter  = 0
        if self._transition_timer is not None:
            self._transition_timer.cancel()
            self._transition_timer = None
        self.get_logger().info('[MANAGER] Détection réactivée.')

    # -----------------------------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        if self.in_transition:
            return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                return

            h, w = image.shape[:2]
            roi  = image[int(h * ROI_NEAR_TOP):int(h * ROI_NEAR_BOT), :]

            hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mb   = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
            mb   = cv2.morphologyEx(mb, cv2.MORPH_OPEN,  kernel)
            mb   = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, kernel)

            if cv2.countNonZero(mb) > BLUE_MIN_AREA:
                self.blue_counter += 1
                if self.blue_counter >= BLUE_CONFIRM_FRAMES:
                    self.get_logger().warn(
                        f'[MANAGER] Ligne bleue → challenge {self.challenge_index + 2}')
                    self._next_challenge()
            else:
                self.blue_counter = 0

        except Exception as e:
            self.get_logger().error(f'image_callback: {e}')


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ChallengeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.current_process and node.current_process.poll() is None:
            node.current_process.terminate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
