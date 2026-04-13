import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class CorridorNode(Node):

    def __init__(self):
        super().__init__('corridor')

        # Paramètres
        self.declare_parameter('linear_scale', 0.05)
        self.declare_parameter('angular_scale', 1.2)
        self.declare_parameter('seuil_arret', 0.15)
        self.declare_parameter('angular_scale_obstacle', 0.5)
        self.declare_parameter('seuil_declenchement_virage', 0.28)

        self.linear_scale = float(self.get_parameter('linear_scale').value)
        self.angular_scale = float(self.get_parameter('angular_scale').value)
        self.seuil_arret = float(self.get_parameter('seuil_arret').value)
        self.angular_scale_obstacle = float(self.get_parameter('angular_scale_obstacle').value)
        self.seuil_declenchement_virage = float(self.get_parameter('seuil_declenchement_virage').value)

        # Secteurs angulaires en degrés
        self.declare_parameter('left_center_deg', 90.0)
        self.declare_parameter('right_center_deg', 270.0)
        self.declare_parameter('front_center_deg', 180.0)
        self.declare_parameter('side_window_deg', 30.0)
        self.declare_parameter('front_window_deg', 10.0)

        self.left_center_deg = float(self.get_parameter('left_center_deg').value)
        self.right_center_deg = float(self.get_parameter('right_center_deg').value)
        self.front_center_deg = float(self.get_parameter('front_center_deg').value)
        self.side_window_deg = float(self.get_parameter('side_window_deg').value)
        self.front_window_deg = float(self.get_parameter('front_window_deg').value)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Log
        self.last_log_time = self.get_clock().now()
        self.get_logger().info('corridor_node a commencé')

    def scan_callback(self, msg: LaserScan):

        valeurs_gauche = self.get_valeurs(
            msg,
            self.left_center_deg - self.side_window_deg,
            self.left_center_deg + self.side_window_deg
        )

        valeurs_droite = self.get_valeurs(
            msg,
            self.right_center_deg - self.side_window_deg,
            self.right_center_deg + self.side_window_deg
        )

        valeurs_avant = self.get_valeurs(
            msg,
            self.front_center_deg - self.front_window_deg,
            self.front_center_deg + self.front_window_deg
        )

        dist_gauche = sum(valeurs_gauche) / len(valeurs_gauche) if valeurs_gauche else None
        dist_droite = sum(valeurs_droite) / len(valeurs_droite) if valeurs_droite else None
        dist_avant = min(valeurs_avant) if valeurs_avant else None
        dist_avant_moy = sum(valeurs_avant) / len(valeurs_avant) if valeurs_avant else None

        cmd = Twist()

        # Fallback
        if dist_avant is None or dist_avant_moy is None:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.get_logger().warn(
                f'/scan invalide | nb_gauche={len(valeurs_gauche)} '
                f'nb_droite={len(valeurs_droite)} nb_avant={len(valeurs_avant)}'
            )
            return

        # Si un côté manque, on met une grande distance par défaut
        if dist_gauche is None:
            dist_gauche = msg.range_max

        if dist_droite is None:
            dist_droite = msg.range_max

        # Déclenchement du virage :
        # - la moyenne devant montre que ça se ferme
        # - le robot est encore à peu près centré
        # - il y a de la place à gauche
        if (
            dist_avant_moy < self.seuil_declenchement_virage
            and abs(dist_gauche - dist_droite) < 0.35
            and dist_gauche > 0.22
        ):
            self.get_logger().warn('TOURNER')

            cmd.linear.x = 0.03
            cmd.angular.z = self.angular_scale_obstacle   # + = gauche dans ton simulateur

        else:
            erreur = dist_droite - dist_gauche
            erreur = max(min(erreur, 0.6), -0.6)

            if abs(erreur) < 0.03:
                erreur = 0.0

            # plus l'erreur est grande, plus on ralentit
            cmd.linear.x = self.linear_scale - 0.03 * abs(erreur)
            cmd.linear.x = max(cmd.linear.x, 0.03)

            # si obstacle très proche devant, on ralentit encore
            if dist_avant < 0.25:
                cmd.linear.x = min(cmd.linear.x, 0.03)

            cmd.angular.z = self.angular_scale * erreur
            cmd.angular.z = max(min(cmd.angular.z, 0.4), -0.4)

        self.cmd_pub.publish(cmd)

        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 5e8:
            erreur_log = dist_droite - dist_gauche
            self.get_logger().info(
                f'gauche={dist_gauche:.3f} droite={dist_droite:.3f} '
                f'avant_min={dist_avant:.3f} avant_moy={dist_avant_moy:.3f} '
                f'erreur={erreur_log:.3f}'
            )
            self.last_log_time = now

    def moyenne_secteur(self, scan, deg_debut, deg_fin):
        valeurs = self.get_valeurs(scan, deg_debut, deg_fin)
        if not valeurs:
            return None
        return sum(valeurs) / len(valeurs)

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
                if math.isnan(r):
                    continue
                if math.isinf(r):
                    valeurs.append(scan.range_max)
                elif scan.range_min < r < scan.range_max:
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