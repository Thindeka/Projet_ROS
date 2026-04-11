import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


"""
FONCTIONNEMENT LIDAR
Le LiDAR envoie un message LaserScan avec :

une liste de distances : ranges[]
chaque valeur correspond à un angle

idée de navigation :
-> utiliser les côtés pour rester centré et l'avant pour déclencher un virage plus franc
-> en utilisant la distance avant seulement le robot reste souvent coincé
"""



class CorridorNode (Node) :


    def __init__(self) :
        super().__init__('corridor')


        # Paramètres
        self.declare_parameter('linear_scale', 0.04)  
        self.declare_parameter('angular_scale', 0.30)
        self.declare_parameter('seuil_arret', 0.24)
        self.declare_parameter('angular_scale_obstacle', 0.5)
        self.declare_parameter('wall_target_dist', 0.30)
        
        self.linear_scale = float(self.get_parameter('linear_scale').value)      
        self.angular_scale = float(self.get_parameter('angular_scale').value) 
        self.seuil_arret = float(self.get_parameter('seuil_arret').value)  
        self.angular_scale_obstacle = float(self.get_parameter('angular_scale_obstacle').value) 
        self.wall_target_dist = float(self.get_parameter('wall_target_dist').value)


        # Secteurs angulaires en dégrés : on prend des secteurs au lieu d'une seule mesure pour avoir des mesures plus stables
        self.declare_parameter('left_center_deg', 90.0)
        self.declare_parameter('right_center_deg', 270.0)
        self.declare_parameter('front_center_deg', 180.0)
        self.declare_parameter('side_window_deg', 30.0)
        self.declare_parameter('front_window_deg', 20.0)
        # diagonales
        self.declare_parameter('front_left_center_deg', 135.0)
        self.declare_parameter('front_right_center_deg', 225.0)
        self.declare_parameter('diag_window_deg', 20.0)

        self.left_center_deg = float(self.get_parameter('left_center_deg').value)
        self.right_center_deg = float(self.get_parameter('right_center_deg').value)
        self.front_center_deg = float(self.get_parameter('front_center_deg').value)
        self.side_window_deg = float(self.get_parameter('side_window_deg').value)
        self.front_window_deg = float(self.get_parameter('front_window_deg').value)
        self.front_left_center_deg = float(self.get_parameter('front_left_center_deg').value)
        self.front_right_center_deg = float(self.get_parameter('front_right_center_deg').value)
        self.diag_window_deg = float(self.get_parameter('diag_window_deg').value)


        # Logique d'états
        self.mode = "SUIVRE"
        self.turn_steps_remaining = 0
        self.direction_virage = 0   # +1 gauche, -1 droite

        # filtrage pour réduire la sensibilité (filtrage exponentiel des distances)
        self.alpha = 0.7
        self.dist_gauche_f = None
        self.dist_droite_f = None
        self.dist_avant_f = None
        self.dist_avant_gauche_f = None
        self.dist_avant_droite_f = None

        # abonnenement à /scan 
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # publication sur /cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # log
        self.last_log_time = self.get_clock().now()
        self.get_logger().info('corridor_node a commencé')



    def scan_callback (self, msg : LaserScan) :

        # DEBUG 
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

        dist_avant_gauche = self.moyenne_secteur(
            msg,
            self.front_left_center_deg - self.diag_window_deg,
            self.front_left_center_deg + self.diag_window_deg
        )

        dist_avant_droite = self.moyenne_secteur(
            msg,
            self.front_right_center_deg - self.diag_window_deg,
            self.front_right_center_deg + self.diag_window_deg
        )

        # Filtrage exponentiel
        self.dist_gauche_f = dist_gauche if self.dist_gauche_f is None else self.alpha * self.dist_gauche_f + (1.0 - self.alpha) * dist_gauche
        self.dist_droite_f = dist_droite if self.dist_droite_f is None else self.alpha * self.dist_droite_f + (1.0 - self.alpha) * dist_droite
        self.dist_avant_f = dist_avant if self.dist_avant_f is None else self.alpha * self.dist_avant_f + (1.0 - self.alpha) * dist_avant
        self.dist_avant_gauche_f = dist_avant_gauche if self.dist_avant_gauche_f is None else self.alpha * self.dist_avant_gauche_f + (1.0 - self.alpha) * dist_avant_gauche
        self.dist_avant_droite_f = dist_avant_droite if self.dist_avant_droite_f is None else self.alpha * self.dist_avant_droite_f + (1.0 - self.alpha) * dist_avant_droite

        # On remplace les distances brutes par les distances filtrées
        dist_gauche = self.dist_gauche_f
        dist_droite = self.dist_droite_f
        dist_avant = self.dist_avant_f
        dist_avant_gauche = self.dist_avant_gauche_f
        dist_avant_droite = self.dist_avant_droite_f


        cmd = Twist()


        # Fallback 
        if dist_gauche is None or dist_droite is None or dist_avant is None or dist_avant_gauche is None or dist_avant_droite is None :
            cmd.linear.x = 0.0  # on n'avance pas
            cmd.angular.z = 0.0 # on ne tourne pas
            self.cmd_pub.publish(cmd)
            self.get_logger().warn(
                f'/scan invalide | nb_gauche={len(valeurs_gauche)} '
                f'nb_droite={len(valeurs_droite)} nb_avant={len(valeurs_avant)}'
            )
            return

        # --- MODE VIRAGE EN COURS ---
        if self.mode == "VIRAGE_GAUCHE":
            cmd.linear.x = 0.03
            cmd.angular.z = -0.50
            self.turn_steps_remaining -= 1

            # on sort du virage seulement si on a assez tourné
            # ET que l'avant est suffisamment dégagé
            if self.turn_steps_remaining <= 0 and dist_avant > 0.40:
                self.mode = "SUIVRE"

        # --- DÉCLENCHEMENT DU VIRAGE ---
        elif dist_avant < self.seuil_arret and dist_avant_gauche > dist_avant_droite:
            self.mode = "VIRAGE_GAUCHE"
            self.turn_steps_remaining = 25
            cmd.linear.x = 0.02
            cmd.angular.z = -0.25

        # --- SUIVI NORMAL ---
        else:
            erreur = dist_gauche - dist_droite
            erreur = max(min(erreur, 0.6), -0.6)

            if min(dist_gauche, dist_droite) < 0.20:
                cmd.linear.x = 0.025
            else:
                cmd.linear.x = self.linear_scale

            cmd.angular.z = -self.angular_scale * erreur  # faire attention au signe
            cmd.angular.z = max(min(cmd.angular.z, 0.4), -0.4)


        self.cmd_pub.publish(cmd)

        now = self.get_clock().now()

        if (now - self.last_log_time).nanoseconds > 5e8: # 0.5 s
            error = dist_gauche - dist_droite
            self.get_logger().info(
                f'mode={self.mode} gauche={dist_gauche:.3f} droite={dist_droite:.3f} '
                f'avant={dist_avant:.3f} av_g={dist_avant_gauche:.3f} '
                f'av_d={dist_avant_droite:.3f} erreur={error:.3f}'
            )
            self.last_log_time = now



    def moyenne_secteur (self, scan, deg_debut, deg_fin) :
        """
        Retourne la moyenne des angles dans un secteur
        """
        valeurs = self.get_valeurs(scan, deg_debut, deg_fin)
        if not valeurs :
            return None
        return sum(valeurs) / len(valeurs)
    


    def min_secteur (self, scan, deg_debut, deg_fin) :
        """
        Retourne le plus petit angle dans un secteur
        """
        valeurs = self.get_valeurs(scan, deg_debut, deg_fin)
        if not valeurs :
            return None
        return min(valeurs)

    def get_valeurs(self, scan, deg_debut, deg_fin):
        rad_debut = math.radians(deg_debut) % (2.0 * math.pi)
        rad_fin = math.radians(deg_fin) % (2.0 * math.pi)

        valeurs = []

        for i, r in enumerate(scan.ranges):
            angle = (scan.angle_min + i * scan.angle_increment) % (2.0 * math.pi)

            # secteur normal
            if rad_debut <= rad_fin:
                dans_secteur = rad_debut <= angle <= rad_fin
            # secteur qui traverse 0°
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

    



def main(args=None) :

    rclpy.init(args=args)
    node = CorridorNode()

    try :
        rclpy.spin(node)

    except KeyboardInterrupt :
        pass

    finally :
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()