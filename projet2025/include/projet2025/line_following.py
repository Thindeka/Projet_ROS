import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2


# Distance (en mètres) en dessous de laquelle on déclenche l'arrêt d'urgence
OBSTACLE_STOP_DISTANCE = 0.35

# Hauteur relative (0.0 = haut, 1.0 = bas) de la région d'intérêt (ROI)
ROI_TOP_RATIO = 0.55

# Gain proportionnel pour l'erreur latérale ? vitesse angulaire
KP_ANGULAR = 0.004

# Vitesse linéaire de base
LINEAR_SPEED = 0.12


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # --- Paramètre : direction du rond-point ('left' ou 'right') ---
        self.declare_parameter('roundabout_direction', 'left')
        self.roundabout_direction = self.get_parameter(
            'roundabout_direction').get_parameter_value().string_value
        self.get_logger().info(
            f'Roundabout direction: {self.roundabout_direction}')

        # --- Subscriptions ---
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # --- Publisher vitesse ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- État interne ---
        self.obstacle_detected = False
        self.image_width = None

        # Mode : 'follow' ou 'roundabout'
        self.mode = 'follow'

        # Compteur de frames sans ligne détectée (pour basculer en rond-point)
        self.no_line_counter = 0
        self.NO_LINE_THRESHOLD = 15  # frames consécutives avant de supposer l'entrée du rond-point

    # ------------------------------------------------------------------
    # Callback LIDAR : détection d'obstacle frontal
    # ------------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        """Vérifie si un obstacle est présent dans le cône frontal (±30°)."""
        ranges = np.array(msg.ranges)
        n = len(ranges)

        # Indices correspondant à ±30° devant le robot
        # Dans ROS2 / TurtleBot3, index 0 = devant, sens antihoraire
        cone_half = int(30 * n / 360)
        front_indices = list(range(0, cone_half + 1)) + list(range(n - cone_half, n))

        front_ranges = [ranges[i] for i in front_indices
                        if not np.isnan(ranges[i]) and not np.isinf(ranges[i])]

        if front_ranges and min(front_ranges) < OBSTACLE_STOP_DISTANCE:
            if not self.obstacle_detected:
                self.get_logger().warn(
                    f'Obstacle détecté à {min(front_ranges):.2f} m ? ARRÊT')
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

    # ------------------------------------------------------------------
    # Callback image : traitement et commande
    # ------------------------------------------------------------------
    def image_callback(self, msg: CompressedImage):
        # Décodage de l'image compressée
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().warn('Échec du décodage de l\'image compressée')
            return

        h, w = image.shape[:2]
        self.image_width = w

        # --- Arrêt d'urgence ---
        if self.obstacle_detected:
            self._publish_velocity(0.0, 0.0)
            cv2.putText(image, 'OBSTACLE - STOP', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow('Line Following', image)
            cv2.waitKey(1)
            return

        # --- Région d'intérêt (ROI) : bande horizontale basse de l'image ---
        roi_top = int(h * ROI_TOP_RATIO)
        roi = image[roi_top:h, 0:w]

        # --- Détection couleur dans HSV ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Masque VERT
        mask_green = cv2.inRange(
            hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))

        # Masque ROUGE (deux plages HSV)
        mask_red1 = cv2.inRange(
            hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(
            hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_combined = cv2.bitwise_or(mask_green, mask_red)

        # --- Calcul du centroïde via les moments d'image ---
        M = cv2.moments(mask_combined)

        twist = Twist()

        if M['m00'] > 500:  # surface minimale détectée
            # Position horizontale du centroïde
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            self.no_line_counter = 0
            self.mode = 'follow'

            # Erreur = écart du centroïde par rapport au centre de l'image
            error = cx - w // 2

            twist.linear.x = LINEAR_SPEED
            twist.angular.z = -KP_ANGULAR * error

            # Visualisation
            cv2.circle(roi, (cx, cy), 8, (255, 0, 0), -1)
            cv2.line(roi, (w // 2, 0), (w // 2, roi.shape[0]),
                     (0, 255, 255), 1)
            cv2.putText(image, f'Mode: FOLLOW  err={error:+d}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Aucune ligne détectée ? probable entrée dans le rond-point
            self.no_line_counter += 1

            if self.no_line_counter >= self.NO_LINE_THRESHOLD:
                self.mode = 'roundabout'

            if self.mode == 'roundabout':
                twist = self._roundabout_command()
                cv2.putText(image, f'Mode: ROUNDABOUT ({self.roundabout_direction})',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # Ligne brièvement perdue : continuer tout droit
                twist.linear.x = LINEAR_SPEED * 0.5
                twist.angular.z = 0.0
                cv2.putText(image, 'Mode: SEARCHING',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.cmd_pub.publish(twist)

        # Affichage debug
        debug = image.copy()
        # Remettre le masque combiné (en couleur) dans la zone ROI pour visualisation
        mask_bgr = cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
        debug[roi_top:h, 0:w] = cv2.addWeighted(
            debug[roi_top:h, 0:w], 0.6, mask_bgr, 0.4, 0)
        cv2.imshow('Line Following', debug)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Commande pour le rond-point
    # ------------------------------------------------------------------
    def _roundabout_command(self) -> Twist:
        """Génère une commande de virage pour le rond-point."""
        twist = Twist()
        twist.linear.x = 0.08

        if self.roundabout_direction == 'left':
            twist.angular.z = 0.6   # virer à gauche (antihoraire)
        else:
            twist.angular.z = -0.6  # virer à droite (horaire)

        return twist

    # ------------------------------------------------------------------
    def _publish_velocity(self, linear: float, angular: float):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)


# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_velocity(0.0, 0.0)  # stopper le robot proprement
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()