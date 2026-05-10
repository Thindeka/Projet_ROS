import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2


# Obstacle
OBSTACLE_CONE_DEG    = 20
OBSTACLE_DETECT_DIST = 0.30

# Vitesse
LINEAR_SPEED = 0.15
AVOID_SPEED  = 0.10

# Multi-ROI
ROI_FAR_TOP  = 0.40;  ROI_FAR_BOT  = 0.60
ROI_NEAR_TOP = 0.60;  ROI_NEAR_BOT = 0.80

# Fusion
ALPHA_STRAIGHT = 0.6
ALPHA_TURN     = 0.35

# Gains
KP_ANGULAR = 0.007
KP_AVOID   = 0.010
DEAD_ZONE  = 25
SMOOTH     = 0.2 # garde 20% de l'ancienne erreur

SPEED_TURN_REDUCTION = 0.2

# Évitement
AVOID_BIAS        = 200
AVOID_EXIT_FRAMES = 10

MIN_AREA = 600

GREEN_LOWER = np.array([32,  44,  31]);  GREEN_UPPER = np.array([95, 255, 255])
RED_LOWER1  = np.array([  0,  33,  79]); RED_UPPER1  = np.array([ 10, 255, 255])
RED_LOWER2  = np.array([160,  33,  79]); RED_UPPER2  = np.array([179, 255, 255])
YELLOW_LOWER = np.array([18, 80, 80])
YELLOW_UPPER = np.array([35, 255, 255])
BLUE_LOWER   = np.array([100, 50, 50])
BLUE_UPPER   = np.array([130, 255, 255])

kernel = np.ones((5, 5), np.uint8)


class Challenge2Avoid(Node):

    def __init__(self):
        super().__init__('challenge2_avoid')
        self.get_logger().info("[Évitement d'obstacle ")

        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.image_callback, 10)
        self.scan_sub  = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub   = self.create_publisher(Twist, '/cmd_vel', 10)

        self.mode               = 'follow'   # 'follow' ou 'avoid'
        self.smooth_error       = 0.0
        self.avoid_exit_counter = 0
        self.last_cx_mid_far    = None
        self.last_cx_mid_near   = None

        # avoid_bias_sign : 0.0 = pas encore défini, +1.0 = droite, -1.0 = gauche 
        self.avoid_bias_sign = 0.0
        self.last_avoid_sign = 0.0
        self.current_bias    = 0.0

    # LIDAR : déclenche le mode avoid uniquement
   
    def scan_callback(self, msg: LaserScan):
        ranges   = np.array(msg.ranges)
        n        = len(ranges)
        cone_half = int(OBSTACLE_CONE_DEG * n / 360)
        front_idx = list(range(0, cone_half + 1)) + list(range(n - cone_half, n))
        front_valid = [
            ranges[i] for i in front_idx
            if not np.isnan(ranges[i]) and not np.isinf(ranges[i]) and ranges[i] > 0.05
        ]

        if not front_valid:
            return

        min_dist = min(front_valid)

        if self.mode == 'follow' and min_dist < OBSTACLE_DETECT_DIST:
            self.mode               = 'avoid'
            self.avoid_exit_counter = 0
            self.avoid_bias_sign    = 0.0   # sera défini par la caméra
            self.get_logger().warn(
                f'[AVOID déclenché | dist={min_dist:.2f}m | '
                f'côté en attente de la caméra...'
            )

    # Détection côté obstacle via caméra
    # Retourne +1.0 (biais droite) si obstacle à gauche du centre de piste,
    #          -1.0 (biais gauche) si obstacle à droite,
    #           None si obstacle non visible.

    def detect_obstacle_side(self, roi, cx_mid_piste, w):
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        M = cv2.moments(mask)
        if M['m00'] < MIN_AREA:
            return None   # obstacle jaune non visible 

        cx_obstacle = int(M['m10'] / M['m00'])

        if cx_obstacle < cx_mid_piste:
            self.get_logger().info(
                f'Obstacle à GAUCHE (cx_obs={cx_obstacle} < cx_piste={cx_mid_piste}) -> biais DROITE'
            )
            return 1.0   # obstacle à gauche -> on part à droite
        else:
            self.get_logger().info(
                f'Obstacle à DROITE (cx_obs={cx_obstacle} > cx_piste={cx_mid_piste}) -> biais GAUCHE'
            )
            return -1.0  # obstacle à droite -> on part à gauche

   
    def _check_avoid_exit(self):
        self.avoid_exit_counter += 1
        if self.avoid_exit_counter >= AVOID_EXIT_FRAMES:
            self.mode               = 'follow'
            self.avoid_exit_counter = 0
            self.smooth_error       = 0.0
            self.avoid_bias_sign    = 0.0
            self.get_logger().info('FOLLOW (obstacle passé)')

    def _detect_roi(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mg  = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        mr  = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
            cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        )
        mg = cv2.morphologyEx(cv2.morphologyEx(mg, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        mr = cv2.morphologyEx(cv2.morphologyEx(mr, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

        Mg, Mr = cv2.moments(mg), cv2.moments(mr)
        has_g  = Mg['m00'] > MIN_AREA
        has_r  = Mr['m00'] > MIN_AREA
        cx_g   = int(Mg['m10'] / Mg['m00']) if has_g else None
        cx_r   = int(Mr['m10'] / Mr['m00']) if has_r else None
        return has_g, has_r, cx_g, cx_r, mg, mr

    def _cx_mid(self, has_g, has_r, cx_g, cx_r, w, fallback):
        if has_g and has_r:  return (cx_g + cx_r) // 2,          1.0
        elif has_g:          return min(cx_g + w // 3, w - 1),   0.6
        elif has_r:          return max(cx_r - w // 3, 0),        0.6
        else:                return fallback if fallback is not None else w // 2, 0.3

    
    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                return
            h, w = image.shape[:2]

            far_top  = int(h * ROI_FAR_TOP);  far_bot  = int(h * ROI_FAR_BOT)
            near_top = int(h * ROI_NEAR_TOP); near_bot = int(h * ROI_NEAR_BOT)

            hg_f, hr_f, cxg_f, cxr_f, _, _ = self._detect_roi(image[far_top:far_bot,   :])
            hg_n, hr_n, cxg_n, cxr_n, _, _ = self._detect_roi(image[near_top:near_bot, :])

            cx_mid_far,  conf_far  = self._cx_mid(hg_f, hr_f, cxg_f, cxr_f, w, self.last_cx_mid_far)
            cx_mid_near, conf_near = self._cx_mid(hg_n, hr_n, cxg_n, cxr_n, w, self.last_cx_mid_near)
            self.last_cx_mid_far  = cx_mid_far
            self.last_cx_mid_near = cx_mid_near

            # Mise à jour du côté d'évitement par caméra 
            if self.mode == 'avoid':
                side = self.detect_obstacle_side(image[near_top:near_bot, :], cx_mid_near, w)
                if side is not None:
                    self.avoid_bias_sign = side 
                    self.last_avoid_sign = side   # mise à jour dynamique frame par frame
                # si l'obstacle n'est plus visible et que le côté n'a jamais été défini,
                # on prend l'inverse du dernier signe connu (hardcodé)
                if self.avoid_bias_sign == 0.0:
                    self.avoid_bias_sign = self.last_avoid_sign
                    self.get_logger().warn('Obstacle non visible en caméra, biais inverse par défaut')
                self._check_avoid_exit()

            # Calcul erreur et commande 
            err_far  = cx_mid_far  - w // 2
            err_near = cx_mid_near - w // 2

            target_bias   = AVOID_BIAS * self.avoid_bias_sign if self.mode == 'avoid' else 0.0
            self.current_bias = 0.6 * self.current_bias + 0.4 * target_bias

            if self.mode == 'avoid':
                fused_error = err_near + self.current_bias
                kp    = KP_AVOID
                speed = AVOID_SPEED
            else:
                virage_factor = min(abs(err_far) / (w // 2), 1.0)
                alpha         = ALPHA_STRAIGHT * (1 - virage_factor) + ALPHA_TURN * virage_factor
                fused_error   = alpha * err_far + (1.0 - alpha) * err_near
                kp    = KP_ANGULAR
                speed = LINEAR_SPEED

            self.smooth_error = SMOOTH * self.smooth_error + (1 - SMOOTH) * fused_error
            angular     = 0.0 if abs(self.smooth_error) < DEAD_ZONE else -kp * self.smooth_error
            turn_factor = 1.0 - SPEED_TURN_REDUCTION * min(abs(angular) / 0.5, 1.0)
            self._publish_velocity(speed * turn_factor * (conf_far + conf_near) / 2, angular)

            # --- Debug visuel ---
            debug = image.copy()
            for y, c in [
                (far_top,  (255, 200,   0)),
                (far_bot,  (255, 200,   0)),
                (near_top, (  0, 200, 200)),
                (near_bot, (  0, 200, 200)),
            ]:
                cv2.line(debug, (0, y), (w, y), c, 1)
            cv2.line(debug, (w // 2, far_top), (w // 2, h), (0, 255, 255), 1)

            # Centre piste estimé 
            cv2.circle(debug, (cx_mid_near, near_top + (near_bot - near_top) // 2), 8, (0, 255, 0), -1)

            color = (0, 0, 255) if self.mode == 'avoid' else (0, 255, 0)
            bias_dir = 'D' if self.avoid_bias_sign > 0 else ('G' if self.avoid_bias_sign < 0 else '?')
            cv2.putText(debug, f'[C2:{self.mode.upper()}]',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(debug,
                        f'err_near={err_near:+d} bias={self.current_bias:+.0f} '
                        f'ang={angular:+.3f} side={bias_dir}',
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow('Challenge 2 - Avoid', debug)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'image_callback: {e}')

    def _publish_velocity(self, linear, angular):
        t = Twist()
        t.linear.x  = float(linear)
        t.angular.z = float(angular)
        self.cmd_pub.publish(t)


def main(args=None):
    rclpy.init(args=args)
    node = Challenge2Avoid()
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