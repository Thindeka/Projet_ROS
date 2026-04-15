import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class CorridorNode(Node):

    def __init__(self):
        super().__init__('corridor')

        # Paramètres
        self.declare_parameter('linear_scale', 0.02) # 0.05
        self.declare_parameter('angular_scale', 0.8)
        self.declare_parameter('seuil_virage', 0.36)
        self.declare_parameter('angular_scale_obstacle', 0.45)

        self.linear_scale = float(self.get_parameter('linear_scale').value)
        self.angular_scale = float(self.get_parameter('angular_scale').value)
        self.seuil_virage = float(self.get_parameter('seuil_virage').value)
        self.angular_scale_obstacle = float(self.get_parameter('angular_scale_obstacle').value)

        self.mode = "CENTER"
        self.turn_steps_remaining = 0

        # Secteurs
        self.declare_parameter('left_center_deg', 90.0)
        self.declare_parameter('right_center_deg', 270.0)
        self.declare_parameter('front_center_deg', 180.0)
        self.declare_parameter('side_window_deg', 20.0)
        self.declare_parameter('front_window_deg', 15.0)

        self.left_center_deg = float(self.get_parameter('left_center_deg').value)
        self.right_center_deg = float(self.get_parameter('right_center_deg').value)
        self.front_center_deg = float(self.get_parameter('front_center_deg').value)
        self.side_window_deg = float(self.get_parameter('side_window_deg').value)
        self.front_window_deg = float(self.get_parameter('front_window_deg').value)

        # Etat
        self.mode = "CENTER"
        self.turn_steps_remaining = 0

        # seuils
        self.declare_parameter('seuil_centrage_virage', 0.22)
        self.declare_parameter('seuil_sortie_virage', 0.40)

        self.declare_parameter('seuil_pre_virage', 0.50)
        self.declare_parameter('seuil_min_gauche_pour_tourner', 0.20)

        self.seuil_pre_virage = float(self.get_parameter('seuil_pre_virage').value)
        self.seuil_min_gauche_pour_tourner = float(self.get_parameter('seuil_min_gauche_pour_tourner').value)

        self.seuil_centrage_virage = float(self.get_parameter('seuil_centrage_virage').value)
        self.seuil_sortie_virage = float(self.get_parameter('seuil_sortie_virage').value)


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

        # mesures
        dist_gauche = self.mediane_secteur(
            msg,
            self.left_center_deg - self.side_window_deg,
            self.left_center_deg + self.side_window_deg
        )

        dist_droite = self.mediane_secteur(
            msg,
            self.right_center_deg - self.side_window_deg,
            self.right_center_deg + self.side_window_deg
        )

        dist_avant = self.min_secteur(
            msg,
            self.front_center_deg - self.front_window_deg,
            self.front_center_deg + self.front_window_deg
        )

        cmd = Twist()

        # sécurité
        if dist_avant is None :
            self.cmd_pub.publish(cmd)
            return

        if dist_gauche is None :
            dist_gauche = 1.0

        if dist_droite is None:
            dist_droite = 1.0

        erreur = dist_droite - dist_gauche

        # -------------------------
        # MODE VIRAGE (persistant)
        # -------------------------
        if self.mode == "TURN_LEFT":
            self.get_logger().warn("VIRAGE")

            # trop près du mur intérieur : avancer sans tourner
            if dist_gauche < 0.18:
                cmd.linear.x = 0.015
                cmd.angular.z = 0.0

            # fin de virage : rotation plus douce
            elif dist_avant > 0.22:
                cmd.linear.x = 0.02
                cmd.angular.z = 0.18

            # coeur du virage
            else:
                cmd.linear.x = 0.02
                cmd.angular.z = self.angular_scale_obstacle

            self.turn_steps_remaining -= 1

            if (
                self.turn_steps_remaining <= 0
                and dist_avant > self.seuil_sortie_virage
                and abs(dist_droite - dist_gauche) < 0.25
            ):
                self.mode = "CENTER"

        # -------------------------
        # DECLENCHEMENT VIRAGE
        # -------------------------
        elif (
            dist_avant < self.seuil_virage
            and dist_gauche > self.seuil_min_gauche_pour_tourner
        ):
            self.get_logger().warn("DECLENCHEMENT VIRAGE")
            self.mode = "TURN_LEFT"
            self.turn_steps_remaining = 18

            cmd.linear.x = 0.02
            cmd.angular.z = self.angular_scale_obstacle

        # -------------------------
        # PRE-VIRAGE
        # -------------------------
        elif dist_avant < self.seuil_pre_virage:
            self.get_logger().warn("PRE_VIRAGE")
            self.mode = "CENTER"

            erreur = max(min(erreur, 0.20), -0.20)

            cmd.linear.x = 0.03
            cmd.angular.z = -0.5 * self.angular_scale * erreur
            cmd.angular.z = max(min(cmd.angular.z, 0.20), -0.20)

        # -------------------------
        # MODE CENTRAGE
        # -------------------------
        else:
            self.get_logger().warn("CENTER")
            self.mode = "CENTER"

            erreur = max(min(erreur, 0.25), -0.25)

            if abs(erreur) < 0.02:
                erreur = 0.0

            cmd.linear.x = self.linear_scale
            if abs(erreur) > 0.12:
                cmd.linear.x = 0.04

            cmd.angular.z = -self.angular_scale * erreur
            cmd.angular.z = max(min(cmd.angular.z, 0.30), -0.30)

        self.cmd_pub.publish(cmd)

        # log
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 5e8:
            self.get_logger().info(
                f"mode={self.mode} gauche={dist_gauche:.3f} droite={dist_droite:.3f} "
                f"avant={dist_avant:.3f} erreur={erreur:.3f} "
                f"lin={cmd.linear.x:.3f} ang={cmd.angular.z:.3f} "
                f"steps={self.turn_steps_remaining}"
            )
            self.last_log_time = now

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