import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Obstacle detection
OBSTACLE_STOP_DISTANCE = 0.30
OBSTACLE_CONE_DEG      = 15

# Base speeds
LINEAR_SPEED           = 0.15
ROUNDABOUT_SPEED       = 0.10

# --- Multi-ROI layout (fractions of image height) ---
# ROI_FAR   : anticipation — détecte les virages tôt
# ROI_NEAR  : stabilisation — contrôle au plus près
# ROI_BOTTOM: détection rond-point (inversion des lignes)
ROI_FAR_TOP    = 0.40
ROI_FAR_BOT    = 0.60
ROI_NEAR_TOP   = 0.60
ROI_NEAR_BOT   = 0.80
ROI_BOTTOM_TOP = 0.80
ROI_BOTTOM_BOT = 1.00

# Fusion far/near: alpha * err_far + (1-alpha) * err_near
# Plus alpha est grand, plus le robot anticipe (bon pour virages),
# mais moins il est stable en ligne droite → ajuster selon comportement.
ALPHA_STRAIGHT    = 0.6   # ligne droite : on privilégie l'anticipation
ALPHA_TURN        = 0.35  # virage       : on privilégie la stabilité

# Gains
KP_ANGULAR    = 0.004
KP_ROUNDABOUT = 0.002
DEAD_ZONE     = 25
SMOOTH        = 0.5

# Vitesse réduite si fort virage (proportion)
SPEED_TURN_REDUCTION = 0.5   # réduction max (50 %) à full angular

# Rond-point
ROUNDABOUT_ENTER_FRAMES = 8
ROUNDABOUT_EXIT_FRAMES  = 5

# Area minimale pour considérer un blob valide
MIN_AREA = 600

# --- Plages HSV ---
# Vert (fonctionne bien en intérieur avec éclairage artificial)
GREEN_LOWER = np.array([40,  40,  40])
GREEN_UPPER = np.array([85, 255, 255])

# Rouge (deux plages car le rouge "wrape" autour de H=0/180)
RED_LOWER1  = np.array([  0,  80,  50])
RED_UPPER1  = np.array([ 10, 255, 255])
RED_LOWER2  = np.array([165,  80,  50])
RED_UPPER2  = np.array([180, 255, 255])

kernel = np.ones((5, 5), np.uint8)


# ---------------------------------------------------------------------------
class LineFollower(Node):

    def __init__(self):
        super().__init__('line_follower')

        # Paramètre : direction du rond-point ('left' ou 'right')
        self.declare_parameter('roundabout_direction', 'left')
        self.roundabout_direction = self.get_parameter(
            'roundabout_direction').get_parameter_value().string_value
        self.get_logger().info(f'[INIT] roundabout_direction={self.roundabout_direction}')

        self.image_sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed',
            self.image_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # État
        self.obstacle_detected   = False
        self.smooth_error        = 0.0
        self.mode                = 'follow'   # 'follow' | 'roundabout'
        self.inversion_counter   = 0
        self.normal_counter      = 0
        self.last_cx_mid_far     = None
        self.last_cx_mid_near    = None

    # -----------------------------------------------------------------------
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
            d = min(front_valid)
            self.get_logger().info(f'[LIDAR] Obstacle à {d:.2f} m — ARRÊT')
        self.obstacle_detected = detected

    # -----------------------------------------------------------------------
    def _detect_roi(self, roi):
        """
        Retourne (has_g, has_r, cx_g, cx_r) pour une ROI donnée.
        Applique morphologie pour réduire le bruit.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mg  = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        mr1 = cv2.inRange(hsv, RED_LOWER1,  RED_UPPER1)
        mr2 = cv2.inRange(hsv, RED_LOWER2,  RED_UPPER2)
        mr  = cv2.bitwise_or(mr1, mr2)

        mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN,  kernel)
        mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, kernel)
        mr = cv2.morphologyEx(mr, cv2.MORPH_OPEN,  kernel)
        mr = cv2.morphologyEx(mr, cv2.MORPH_CLOSE, kernel)

        Mg = cv2.moments(mg)
        Mr = cv2.moments(mr)
        has_g = Mg['m00'] > MIN_AREA
        has_r = Mr['m00'] > MIN_AREA
        cx_g = int(Mg['m10'] / Mg['m00']) if has_g else None
        cx_r = int(Mr['m10'] / Mr['m00']) if has_r else None
        return has_g, has_r, cx_g, cx_r, mg, mr

    # -----------------------------------------------------------------------
    def _cx_mid_from_detection(self, has_g, has_r, cx_g, cx_r, w, fallback):
        """
        Calcule le cx_mid et un facteur de confiance selon ce qui est détecté.
        Retourne (cx_mid, confidence).
        """
        if has_g and has_r:
            return (cx_g + cx_r) // 2, 1.0
        elif has_g:
            # Vert seul → on est trop à gauche, corriger vers la droite
            return min(cx_g + w // 3, w - 1), 0.6
        elif has_r:
            # Rouge seul → on est trop à droite, corriger vers la gauche
            return max(cx_r - w // 3, 0), 0.6
        else:
            return fallback if fallback is not None else w // 2, 0.3

    # -----------------------------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                return

            h, w = image.shape[:2]

            # --- Arrêt obstacle ---
            if self.obstacle_detected:
                self._publish_velocity(0.0, 0.0)
                cv2.putText(image, 'OBSTACLE - STOP', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow('Line Following', image)
                cv2.waitKey(1)
                return

            # --- Calcul des tranches ROI ---
            far_top    = int(h * ROI_FAR_TOP);    far_bot    = int(h * ROI_FAR_BOT)
            near_top   = int(h * ROI_NEAR_TOP);   near_bot   = int(h * ROI_NEAR_BOT)
            bottom_top = int(h * ROI_BOTTOM_TOP); bottom_bot = int(h * ROI_BOTTOM_BOT)

            roi_far    = image[far_top:far_bot,       :]
            roi_near   = image[near_top:near_bot,     :]
            roi_bottom = image[bottom_top:bottom_bot, :]

            # --- Détection dans chaque ROI ---
            hg_f, hr_f, cxg_f, cxr_f, mg_f, mr_f = self._detect_roi(roi_far)
            hg_n, hr_n, cxg_n, cxr_n, mg_n, mr_n = self._detect_roi(roi_near)
            hg_b, hr_b, cxg_b, cxr_b, _,    _    = self._detect_roi(roi_bottom)

            # --- Machine à états : détection rond-point dans ROI bottom ---
            # Inversion = vert à droite du rouge (hors du sens normal)
            lines_inverted = hg_b and hr_b and (cxg_b > cxr_b)

            if self.mode == 'follow':
                if lines_inverted:
                    self.inversion_counter += 1
                    self.normal_counter = 0
                    if self.inversion_counter >= ROUNDABOUT_ENTER_FRAMES:
                        self.mode = 'roundabout'
                        self.inversion_counter = 0
                        self.smooth_error = 0.0
                        self.get_logger().info(
                            f'[STATE] >>> ROUNDABOUT ({self.roundabout_direction})')
                else:
                    self.inversion_counter = 0

            elif self.mode == 'roundabout':
                if hg_b and hr_b and not lines_inverted:
                    self.normal_counter += 1
                    if self.normal_counter >= ROUNDABOUT_EXIT_FRAMES:
                        self.mode = 'follow'
                        self.normal_counter = 0
                        self.smooth_error = 0.0
                        self.get_logger().info('[STATE] >>> FOLLOW')
                else:
                    self.normal_counter = 0

            # --- Calcul des erreurs far et near ---
            cx_mid_far, conf_far = self._cx_mid_from_detection(
                hg_f, hr_f, cxg_f, cxr_f, w, self.last_cx_mid_far)
            cx_mid_near, conf_near = self._cx_mid_from_detection(
                hg_n, hr_n, cxg_n, cxr_n, w, self.last_cx_mid_near)

            self.last_cx_mid_far  = cx_mid_far
            self.last_cx_mid_near = cx_mid_near

            err_far  = cx_mid_far  - w // 2
            err_near = cx_mid_near - w // 2

            # --- Fusion adaptative ---
            # Plus l'erreur far est grande, plus on est dans un virage →
            # on réduit alpha pour donner plus de poids au near (stabilité).
            virage_factor = min(abs(err_far) / (w // 2), 1.0)
            alpha = ALPHA_STRAIGHT * (1 - virage_factor) + ALPHA_TURN * virage_factor
            fused_error = alpha * err_far + (1.0 - alpha) * err_near

            # En mode rond-point, on écrase l'erreur fusionnée :
            # utiliser uniquement near pour tenir la trajectoire dans le rond-point.
            if self.mode == 'roundabout':
                # Direction : gauche → bias positif, droite → bias négatif
                direction_bias = -30 if self.roundabout_direction == 'right' else 30
                fused_error = err_near + direction_bias

            # --- Lissage et commande angulaire ---
            self.smooth_error = SMOOTH * self.smooth_error + (1 - SMOOTH) * fused_error

            kp    = KP_ROUNDABOUT if self.mode == 'roundabout' else KP_ANGULAR
            speed = ROUNDABOUT_SPEED if self.mode == 'roundabout' else LINEAR_SPEED

            angular = 0.0 if abs(self.smooth_error) < DEAD_ZONE \
                      else -kp * self.smooth_error

            # Réduction de vitesse proportionnelle à l'angle commandé
            turn_factor = 1.0 - SPEED_TURN_REDUCTION * min(abs(angular) / 0.5, 1.0)
            confidence  = (conf_far + conf_near) / 2.0

            self._publish_velocity(speed * turn_factor * confidence, angular)

            # --- Debug visuel ---
            debug = image.copy()
            # Lignes de séparation ROI
            cv2.line(debug, (0, far_top),    (w, far_top),    (255, 200,   0), 1)
            cv2.line(debug, (0, far_bot),    (w, far_bot),    (255, 200,   0), 1)
            cv2.line(debug, (0, near_top),   (w, near_top),   (  0, 200, 200), 1)
            cv2.line(debug, (0, near_bot),   (w, near_bot),   (  0, 200, 200), 1)
            cv2.line(debug, (0, bottom_top), (w, bottom_top), (200,   0, 200), 1)

            # Centre image
            cv2.line(debug, (w // 2, far_top), (w // 2, h), (0, 255, 255), 1)

            # Points centroids far
            if hg_f:
                cv2.circle(debug, (cxg_f, far_top + (far_bot - far_top)//2), 5, (0, 255, 0), -1)
            if hr_f:
                cv2.circle(debug, (cxr_f, far_top + (far_bot - far_top)//2), 5, (0, 0, 255), -1)
            cv2.circle(debug, (cx_mid_far, far_top + (far_bot - far_top)//2), 7, (255, 200, 0), -1)

            # Points centroids near
            if hg_n:
                cv2.circle(debug, (cxg_n, near_top + (near_bot - near_top)//2), 5, (0, 255, 0), -1)
            if hr_n:
                cv2.circle(debug, (cxr_n, near_top + (near_bot - near_top)//2), 5, (0, 0, 255), -1)
            cv2.circle(debug, (cx_mid_near, near_top + (near_bot - near_top)//2), 7, (0, 200, 200), -1)

            # Overlay texte
            color = (0, 165, 255) if self.mode == 'roundabout' else (0, 255, 0)
            cv2.putText(debug, f'[{self.mode.upper()}] dir={self.roundabout_direction}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(debug,
                        f'err_far={err_far:+d}  err_near={err_near:+d}  '
                        f'alpha={alpha:.2f}  ang={angular:+.3f}',
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow('Line Following', debug)
            cv2.imshow('Mask green far', mg_f)
            cv2.imshow('Mask red far',   mr_f)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'image_callback: {e}')

    # -----------------------------------------------------------------------
    def _publish_velocity(self, linear: float, angular: float):
        t = Twist()
        t.linear.x  = linear
        t.angular.z = angular
        self.cmd_pub.publish(t)


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
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