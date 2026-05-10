import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2

# Paramètres poru l'arrêt d'urgence
OBSTACLE_STOP_DISTANCE = 0.25
OBSTACLE_CONE_DEG      = 15

# Vitesse 
LINEAR_SPEED  = 0.15
ROUNDABOUT_SPEED = 0.10

# Multi-ROI
ROI_FAR_TOP    = 0.40;  ROI_FAR_BOT    = 0.60
ROI_NEAR_TOP   = 0.60;  ROI_NEAR_BOT   = 0.80
ROI_BOTTOM_TOP = 0.80;  ROI_BOTTOM_BOT = 1.00

# Fusion
ALPHA_STRAIGHT = 0.6
ALPHA_TURN     = 0.35

# Gains
KP_ANGULAR    = 0.007
KP_ROUNDABOUT = 0.002
DEAD_ZONE     = 25 # erreur en px par rapport aux centroides
SMOOTH        = 0.2

SPEED_TURN_REDUCTION    = 0.5
ROUNDABOUT_ENTER_FRAMES = 8
ROUNDABOUT_EXIT_FRAMES  = 5
MIN_AREA                = 600

# HSV

GREEN_LOWER = np.array([32,  44,  31]);  GREEN_UPPER = np.array([95, 255, 255])
RED_LOWER1  = np.array([  0,  33,  79]); RED_UPPER1  = np.array([ 10, 255, 255])
RED_LOWER2  = np.array([160,  33,  79]); RED_UPPER2  = np.array([179, 255, 255])

kernel = np.ones((5, 5), np.uint8)


class Challenge1Follow(Node):

    def __init__(self):
        super().__init__('challenge1_follow')

        self.declare_parameter('roundabout_direction', 'left')
        self.roundabout_direction = self.get_parameter(
            'roundabout_direction').get_parameter_value().string_value
        self.get_logger().info(f'Suivi de ligne | roundabout={self.roundabout_direction}')

        self.image_sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        self.scan_sub  = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub   = self.create_publisher(Twist, '/cmd_vel', 10)

        self.obstacle_detected  = False
        self.smooth_error       = 0.0
        self.mode               = 'follow'
        self.inversion_counter  = 0
        self.normal_counter     = 0
        self.last_cx_mid_far    = None
        self.last_cx_mid_near   = None

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        n = len(ranges)
        cone_half = int(OBSTACLE_CONE_DEG * n / 360)
        front_idx = list(range(0, cone_half + 1)) + list(range(n - cone_half, n))
        front_valid = [ranges[i] for i in front_idx
                       if not np.isnan(ranges[i]) and not np.isinf(ranges[i])
                       and ranges[i] > 0.05]
        detected = bool(front_valid and min(front_valid) < OBSTACLE_STOP_DISTANCE)
        if detected and not self.obstacle_detected:
            self.get_logger().info(f'[C1] Obstacle à {min(front_valid):.2f}m — ARRÊT')
        self.obstacle_detected = detected

    
    def _detect_roi(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mg  = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        mr  = cv2.bitwise_or(cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
                              cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
        mg  = cv2.morphologyEx(cv2.morphologyEx(mg, cv2.MORPH_OPEN,  kernel), cv2.MORPH_CLOSE, kernel)
        mr  = cv2.morphologyEx(cv2.morphologyEx(mr, cv2.MORPH_OPEN,  kernel), cv2.MORPH_CLOSE, kernel)
        Mg, Mr = cv2.moments(mg), cv2.moments(mr)
        has_g = Mg['m00'] > MIN_AREA;  has_r = Mr['m00'] > MIN_AREA
        cx_g  = int(Mg['m10'] / Mg['m00']) if has_g else None
        cx_r  = int(Mr['m10'] / Mr['m00']) if has_r else None
        return has_g, has_r, cx_g, cx_r, mg, mr

    def _cx_mid(self, has_g, has_r, cx_g, cx_r, w, fallback):
        if has_g and has_r:  return (cx_g + cx_r) // 2, 1.0
        elif has_g:          return min(cx_g + w // 3, w - 1), 0.6
        elif has_r:          return max(cx_r - w // 3, 0), 0.6
        else:                return fallback if fallback is not None else w // 2, 0.3

    
    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None: return
            h, w = image.shape[:2]

            # ARRÊT D'URGENCE
            if self.obstacle_detected:
                self._publish_velocity(0.0, 0.0)
                cv2.putText(image, 'OBSTACLE - STOP', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow('Challenge 1 - Follow', image)
                cv2.waitKey(1)
                return

            far_top  = int(h * ROI_FAR_TOP);   far_bot  = int(h * ROI_FAR_BOT)
            near_top = int(h * ROI_NEAR_TOP);  near_bot = int(h * ROI_NEAR_BOT)
            bot_top  = int(h * ROI_BOTTOM_TOP)

            hg_f, hr_f, cxg_f, cxr_f, mg_f, mr_f = self._detect_roi(image[far_top:far_bot, :])
            hg_n, hr_n, cxg_n, cxr_n, mg_n, mr_n = self._detect_roi(image[near_top:near_bot, :])
            hg_b, hr_b, cxg_b, cxr_b, _,    _    = self._detect_roi(image[bot_top:h, :])

            lines_inverted = hg_f and hr_f and (cxg_f > cxr_f)

            if self.mode == 'follow':
                if lines_inverted:
                    self.inversion_counter += 1
                    if self.inversion_counter >= ROUNDABOUT_ENTER_FRAMES:
                        self.mode = 'roundabout'
                        self.inversion_counter = 0
                        self.smooth_error = 0.0
                        self.get_logger().info(f'ROUNDABOUT ({self.roundabout_direction})')
                else:
                    self.inversion_counter = 0
            elif self.mode == 'roundabout':
                if hg_b and hr_b and not lines_inverted:
                    self.normal_counter += 1
                    if self.normal_counter >= ROUNDABOUT_EXIT_FRAMES:
                        self.mode = 'follow'
                        self.normal_counter = 0
                        self.smooth_error = 0.0
                        self.get_logger().info('FOLLOW')
                else:
                    self.normal_counter = 0

            cx_mid_far,  conf_far  = self._cx_mid(hg_f, hr_f, cxg_f, cxr_f, w, self.last_cx_mid_far)
            cx_mid_near, conf_near = self._cx_mid(hg_n, hr_n, cxg_n, cxr_n, w, self.last_cx_mid_near)
            self.last_cx_mid_far  = cx_mid_far
            self.last_cx_mid_near = cx_mid_near

            err_far  = cx_mid_far  - w // 2
            err_near = cx_mid_near - w // 2

            virage_factor = min(abs(err_far) / (w // 2), 1.0)
            alpha = ALPHA_STRAIGHT * (1 - virage_factor) + ALPHA_TURN * virage_factor

            if self.mode == 'roundabout':
                direction_bias = -30 if self.roundabout_direction == 'right' else 30
                fused_error = err_near + direction_bias
                kp = KP_ROUNDABOUT; speed = ROUNDABOUT_SPEED
            else:
                fused_error = alpha * err_far + (1.0 - alpha) * err_near
                kp = KP_ANGULAR;    speed = LINEAR_SPEED

            self.smooth_error = SMOOTH * self.smooth_error + (1 - SMOOTH) * fused_error
            angular = 0.0 if abs(self.smooth_error) < DEAD_ZONE else -kp * self.smooth_error
            turn_factor = 1.0 - SPEED_TURN_REDUCTION * min(abs(angular) / 0.5, 1.0)
            self._publish_velocity(speed * turn_factor * (conf_far + conf_near) / 2, angular)

            # Debug
            debug = image.copy()

            # Lignes ROI
            for y, c in [
                (far_top,  (255, 200,   0)),
                (far_bot,  (255, 200,   0)),
                (near_top, (  0, 200, 200)),
                (near_bot, (  0, 200, 200)),
                (bot_top,  (200,   0, 200)),
            ]:
                cv2.line(debug, (0, y), (w, y), c, 1)
            cv2.line(debug, (w // 2, far_top), (w // 2, h), (0, 255, 255), 1)

            # Centroïdes FAR
            if hg_f:
                cv2.circle(debug, (cxg_f, far_top + (far_bot - far_top) // 2), 5, (0, 255, 0), -1)
            if hr_f:
                cv2.circle(debug, (cxr_f, far_top + (far_bot - far_top) // 2), 5, (0, 0, 255), -1)
            cv2.circle(debug, (cx_mid_far, far_top + (far_bot - far_top) // 2), 7, (255, 200, 0), -1)

            # Centroïdes NEAR
            if hg_n:
                cv2.circle(debug, (cxg_n, near_top + (near_bot - near_top) // 2), 5, (0, 255, 0), -1)
            if hr_n:
                cv2.circle(debug, (cxr_n, near_top + (near_bot - near_top) // 2), 5, (0, 0, 255), -1)
            cv2.circle(debug, (cx_mid_near, near_top + (near_bot - near_top) // 2), 7, (0, 200, 200), -1)

            # Indicateur virage_factor (barre de progression)
            bar_w = int(virage_factor * 150)
            cv2.rectangle(debug, (10, h - 35), (160, h - 20), (50, 50, 50), -1)
            cv2.rectangle(debug, (10, h - 35), (10 + bar_w, h - 20), (0, 200, 255), -1)
            cv2.putText(debug, f'vf={virage_factor:.2f}', (165, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            # Indicateur inversion (rond-point détecté en cours)
            if lines_inverted:
                cv2.putText(debug,
                            f'INVERSION ({self.inversion_counter}/{ROUNDABOUT_ENTER_FRAMES})',
                            (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # Texte mode + métriques
            color = (0, 165, 255) if self.mode == 'roundabout' else (0, 255, 0)
            cv2.putText(debug, f'[C1:{self.mode.upper()}] dir={self.roundabout_direction}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(debug,
                        f'far={err_far:+d} near={err_near:+d} ang={angular:+.3f}',
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Fenêtres masques HSV
            cv2.imshow('Challenge 1 - Follow', debug)
            cv2.imshow('Mask green far',  mg_f)
            cv2.imshow('Mask red far',    mr_f)
            cv2.imshow('Mask green near', mg_n)
            cv2.imshow('Mask red near',   mr_n)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'image_callback: {e}')

    def _publish_velocity(self, linear, angular):
        t = Twist(); t.linear.x = linear; t.angular.z = angular
        self.cmd_pub.publish(t)


def main(args=None):
    rclpy.init(args=args)
    node = Challenge1Follow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_velocity(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()