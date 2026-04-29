import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class CorridorNode(Node):

    def __init__(self):
        super().__init__('corridor')

        # Vitesses
        self.linear_speed = 0.05
        self.slow_speed = 0.025

        # Gains
        self.center_gain = 0.9
        self.turn_bias_gain = 0.45

        # Seuils
        self.front_slow_threshold = 0.55
        self.front_turn_threshold = 0.20

        # Secteurs en degrés
        self.left_center_deg = 90.0
        self.right_center_deg = 270.0
        self.front_center_deg = 180.0
        self.front_left_center_deg = 135.0
        self.front_right_center_deg = 225.0

        # Fenêtres larges
        self.side_window_deg = 35.0
        self.front_window_deg = 20.0
        self.diag_window_deg = 20.0

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.last_log_time = self.get_clock().now()
        self.get_logger().info('corridor_node a commencé')



    def scan_callback(self, msg: LaserScan):

        gauche_large = self.mediane_secteur(
            msg,
            self.left_center_deg - self.side_window_deg,
            self.left_center_deg + self.side_window_deg
        )

        droite_large = self.mediane_secteur(
            msg,
            self.right_center_deg - self.side_window_deg,
            self.right_center_deg + self.side_window_deg
        )

        avant_min = self.min_secteur(
            msg,
            self.front_center_deg - self.front_window_deg,
            self.front_center_deg + self.front_window_deg
        )

        avant_moy = self.moyenne_secteur(
            msg,
            self.front_center_deg - self.front_window_deg,
            self.front_center_deg + self.front_window_deg
        )

        avant_gauche = self.mediane_secteur(
            msg,
            self.front_left_center_deg - self.diag_window_deg,
            self.front_left_center_deg + self.diag_window_deg
        )

        avant_droite = self.mediane_secteur(
            msg,
            self.front_right_center_deg - self.diag_window_deg,
            self.front_right_center_deg + self.diag_window_deg
        )

        cmd = Twist()

        if avant_min is None or avant_moy is None:
            self.cmd_pub.publish(cmd)
            self.get_logger().warn("avant invalide")
            return

        # Valeurs de secours si un côté manque
        if gauche_large is None:
            gauche_large = msg.range_max
        if droite_large is None:
            droite_large = msg.range_max
        if avant_gauche is None:
            avant_gauche = msg.range_max
        if avant_droite is None:
            avant_droite = msg.range_max

        # Erreur de centrage du couloir
        erreur_couloir = droite_large - gauche_large
        erreur_couloir = max(min(erreur_couloir, 0.35), -0.35)

        # Ouverture du virage
        ouverture_gauche = avant_gauche - avant_droite

        # -------------------------
        # CAS 1 : obstacle très proche devant
        # -------------------------
        if avant_min < self.front_turn_threshold:
            self.get_logger().warn("OBSTACLE TRES PROCHE DEVANT")

            ouverture_gauche = avant_gauche - avant_droite

            # 1) trop proche du mur intérieur gauche :
            # on avance pour se dégager, presque sans tourner
            if gauche_large < 0.38:
                cmd.linear.x = 0.045
                cmd.angular.z = 0.05

            # 2) vraie ouverture à gauche :
            # on avance encore, avec une rotation gauche modérée
            elif ouverture_gauche > 0.08:
                cmd.linear.x = 0.040
                cmd.angular.z = 0.16

            # 3) sinon obstacle frontal :
            # on tourne à gauche plus franchement
            else:
                cmd.linear.x = 0.025
                cmd.angular.z = 0.28

            cmd.angular.z = max(min(cmd.angular.z, 0.30), -0.05)

        # -------------------------
        # CAS 2 : pré-virage
        # -------------------------
        elif avant_moy < self.front_slow_threshold:
            self.get_logger().warn("PRE VIRAGE")
            cmd.linear.x = 0.03

            # centrage + biais gauche progressif
            biais_gauche = self.turn_bias_gain * max(min(ouverture_gauche, 0.25), -0.25)
            cmd.angular.z = -self.center_gain * erreur_couloir + biais_gauche
            cmd.angular.z = max(min(cmd.angular.z, 0.35), -0.25)

        # -------------------------
        # CAS 3 : couloir normal
        # -------------------------
        else:
            self.get_logger().warn("COULOIR NORMAL")
            cmd.linear.x = self.linear_speed
            cmd.angular.z = -self.center_gain * erreur_couloir
            cmd.angular.z = max(min(cmd.angular.z, 0.25), -0.25)

        self.cmd_pub.publish(cmd)

        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 5e8:
            self.get_logger().info(
                f"g={gauche_large:.3f} d={droite_large:.3f} "
                f"av_min={avant_min:.3f} av_moy={avant_moy:.3f} "
                f"ag={avant_gauche:.3f} ad={avant_droite:.3f} "
                f"err={erreur_couloir:.3f} ang={cmd.angular.z:.3f}"
            )
            self.last_log_time = now

    def moyenne_secteur(self, scan, deg_debut, deg_fin):
        valeurs = self.get_valeurs(scan, deg_debut, deg_fin)
        if not valeurs:
            return None
        return sum(valeurs) / len(valeurs)

    def mediane_secteur(self, scan, deg_debut, deg_fin):
        valeurs = self.get_valeurs(scan, deg_debut, deg_fin)
        if not valeurs:
            return None
        valeurs.sort()
        return valeurs[len(valeurs) // 2]

    def min_secteur(self, scan, deg_debut, deg_fin):
        valeurs = self.get_valeurs(scan, deg_debut, deg_fin)
        if not valeurs:
            return None
        return min(valeurs)

    def get_valeurs(self, scan, deg_debut, deg_fin):
        rad_debut = math.radians(deg_debut) % (2.0 * math.pi)
        rad_fin = math.radians(deg_fin) % (2.0 * math.pi)

        valeurs = []

        for i, r in enumerate(scan.ranges):
            angle = (scan.angle_min + i * scan.angle_increment) % (2.0 * math.pi)

            if rad_debut <= rad_fin:
                dans_secteur = rad_debut <= angle <= rad_fin
            else:
                dans_secteur = angle >= rad_debut or angle <= rad_fin

            if dans_secteur:
                if math.isnan(r) or math.isinf(r):
                    continue
                if scan.range_min < r < scan.range_max:
                    valeurs.append(r)

        return valeurs


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


if __name__ == '__main__':
    main()