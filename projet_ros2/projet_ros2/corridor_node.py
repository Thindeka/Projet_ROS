import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Twist
import cv2
import numpy as np


class CorridorNode(Node):

    def __init__(self):
        super().__init__('corridor')

        # -------------------------
        # PARAMETRES
        # -------------------------
        self.linear_scale = 0.05
        self.angular_scale = 0.20

        # distances de déclenchement
        self.pre_turn_dist = 0.45
        self.turn_dist = 0.20

        # biais de pré-virage
        self.pre_turn_bias = 0.15

        # rotation gauche principale
        self.turn_angular = 0.35

        # distance cible au mur droit pendant le virage
        self.target_right_dist = 0.45 # 50?

        # ROS
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.current_mode = "CENTER"

        self.last_log_time = self.get_clock().now()

        self.get_logger().info('corridor_node a commencé')

    # -------------------------------------------------
    # FILTRE LIDAR
    # -------------------------------------------------
    def secu(self, values):

        vals = []

        for v in values:

            if math.isnan(v) or math.isinf(v):
                continue

            if 0.02 < v < 3.5:
                vals.append(v)

        if len(vals) == 0:
            return None

        vals.sort()

        return vals[len(vals) // 2]

    # -------------------------------------------------
    # CALLBACK LIDAR
    # -------------------------------------------------
    def scan_callback(self, msg: LaserScan):

        # -------------------------
        # SECTEURS LIDAR
        # -------------------------

        # devant
        dist_avant = self.secu(
            list(msg.ranges[345:360]) +
            list(msg.ranges[0:15])
        )

        # avant gauche
        dist_avant_gauche = self.secu(
            msg.ranges[16:75]
        )

        # gauche
        dist_gauche = self.secu(
            msg.ranges[76:120]
        )

        # droite
        dist_droite = self.secu(
            msg.ranges[240:285]
        )

        # avant droite
        dist_avant_droite = self.secu(
            msg.ranges[286:345]
        )

        cmd = Twist()

        # sécurité
        if dist_avant is None:

            self.current_mode = "STOP"

            self.cmd_pub.publish(cmd)

            return

        # fallback
        if dist_avant_gauche is None:
            dist_avant_gauche = msg.range_max

        if dist_gauche is None:
            dist_gauche = msg.range_max

        if dist_droite is None:
            dist_droite = msg.range_max

        if dist_avant_droite is None:
            dist_avant_droite = msg.range_max

        # debug camera
        self.front_min = dist_avant
        self.left_dist = dist_gauche
        self.right_dist = dist_droite

        # =========================================================
        # 1) VIRAGE FRANC
        # =========================================================
        if dist_avant < self.turn_dist:

            self.current_mode = "TURN_LEFT"

            # erreur mur droit
            erreur_droite = (
                self.target_right_dist - dist_droite
            )

            erreur_droite = max(
                min(erreur_droite, 0.20),
                -0.20
            )

            # vitesse lente
            cmd.linear.x = 0.015

            # rotation gauche + correction mur droit
            cmd.angular.z = (
                self.turn_angular
                + 0.8 * erreur_droite
            )

            cmd.angular.z = max(
                min(cmd.angular.z, 0.45),
                0.15
            )

        # =========================================================
        # 2) PRE-VIRAGE
        # =========================================================
        elif (
            dist_avant < self.pre_turn_dist
            and dist_avant_gauche > dist_avant_droite + 0.10
        ):

            self.current_mode = "PRE_TURN"

            erreur = dist_gauche - dist_droite

            erreur = max(
                min(erreur, 0.30),
                -0.30
            )

            cmd.linear.x = 0.03

            cmd.angular.z = (
                (-self.angular_scale * erreur)
                + self.pre_turn_bias
            )

            cmd.angular.z = max(
                min(cmd.angular.z, 0.35),
                -0.20
            )

        # =========================================================
        # 3) CENTRAGE NORMAL
        # =========================================================
        else:

            self.current_mode = "CENTER"

            erreur = dist_droite - dist_gauche

            erreur = max(
                min(erreur, 0.30),
                -0.30
            )

            if abs(erreur) < 0.03:
                erreur = 0.0

            cmd.linear.x = self.linear_scale

            cmd.angular.z = (
                -self.angular_scale * erreur
            )

            cmd.angular.z = max(
                min(cmd.angular.z, 0.25),
                -0.25
            )

        # publication
        self.cmd_pub.publish(cmd)

        # logs
        now = self.get_clock().now()

        if (
            now - self.last_log_time
        ).nanoseconds > 5e8:

            self.get_logger().info(
                f"mode={self.current_mode} "
                f"av={dist_avant:.2f} "
                f"avg={dist_avant_gauche:.2f} "
                f"g={dist_gauche:.2f} "
                f"d={dist_droite:.2f} "
                f"avd={dist_avant_droite:.2f} "
                f"lin={cmd.linear.x:.3f} "
                f"ang={cmd.angular.z:.3f}"
            )

            self.last_log_time = now

    # -------------------------------------------------
    # CALLBACK CAMERA
    # -------------------------------------------------
    def image_callback(self, msg: CompressedImage):

        try:

            np_arr = np.frombuffer(
                msg.data,
                np.uint8
            )

            image = cv2.imdecode(
                np_arr,
                cv2.IMREAD_COLOR
            )

            if image is None:
                return

            h, w = image.shape[:2]

            debug = image.copy()

            # ligne centrale
            cv2.line(
                debug,
                (w // 2, 0),
                (w // 2, h),
                (0, 255, 255),
                2
            )

            mode_text = self.current_mode

            color = (0, 255, 0)

            if mode_text == "PRE_TURN":
                color = (255, 200, 0)

            elif mode_text == "TURN_LEFT":
                color = (0, 165, 255)

            elif mode_text == "STOP":
                color = (0, 0, 255)

            cv2.putText(
                debug,
                f"MODE : {mode_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            if hasattr(self, "front_min"):

                cv2.putText(
                    debug,
                    f"front={self.front_min:.2f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            if hasattr(self, "left_dist"):

                cv2.putText(
                    debug,
                    f"left={self.left_dist:.2f}",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            if hasattr(self, "right_dist"):

                cv2.putText(
                    debug,
                    f"right={self.right_dist:.2f}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(
                "Corridor Navigation",
                debug
            )

            cv2.waitKey(1)

        except Exception as e:

            self.get_logger().error(
                f'image_callback: {e}'
            )


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main(args=None):

    rclpy.init(args=args)

    node = CorridorNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        stop_msg = Twist()

        node.cmd_pub.publish(stop_msg)

        node.destroy_node()

        rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()