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
"""



class CorridorNode (Node) :


    def __init__(self) :
        super().__init__('corridor')


        # Paramètres
        self.declare_parameters('linear_scale', 0.1)  
        self.declare_parameters('angular_scale', 1.2)
        self.declare_parameters('seuil_arret', 0.3)
        self.declare_parameters('angular_scale_obstacle', 0.5)

        self.linear_scale = float(self.get_parameter('linear_scale').value)      
        self.angular_scale = float(self.get_parameter('angular_scale').value) 
        self.seuil_arret = float(self.get_parameter('seuil_arret').value)  
        self.angular_scale_obstacle = float(self.get_parameter('angular_scale_obstacle').value) 

        # Secteurs angulaires en dégrés : on prend des secteurs au lieu d'une seule mesure pour avoir des mesures plus stables
        self.declare_parameter('left_center_deg', 90.0)
        self.declare_parameter('right_center_deg', -90.0)
        self.declare_parameter('side_window_deg', 20.0)
        self.declare_parameter('front_window_deg', 15.0)

        self.left_center_deg = float(self.get_parameter('left_center_deg').value)
        self.right_center_deg = float(self.get_parameter('right_center_deg').value)
        self.side_window_deg = float(self.get_parameter('side_window_deg').value)
        self.front_window_deg = float(self.get_parameter('front_window_deg').value)

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



    def scan_callback (self, msg) :

        dist_gauche = self.moyenne_secteur(
            msg,
            self.left_center_deg - self.side_window_deg,
            self.left_center_deg + self.side_window_deg
        )

        dist_droite = self.get_sector_mean(
            msg,
            self.right_center_deg - self.side_window_deg,
            self.right_center_deg + self.side_window_deg
        )

        diste_avant = self.get_sector_min(
            msg,
            -self.front_window_deg,
            self.front_window_deg
        )

        cmd = Twist()


        # Fallback 
        if dist_gauche is None or dist_droite is None or diste_avant is None :
            cmd.linear.x = 0.0  # on n'avance pas
            cmd.angular.z = 0.0 # on ne tourne pas
            self.cmd_pub.publish(cmd)
            self.get_logger().warn('Data du /scan invalide, arrêt du robot')


        # Sécurité
        if diste_avant < self.seuil_arret :
            
            cmd.linear.x = 0.0 # on n'avance pas

            # on s'éloigne en tournant du côté le plus proche 
            # A VOIR SI LE SENS EST BON
            if dist_gauche > dist_droite :
                cmd.angular.z = self.angular_scale_obstacle
            else :
                cmd.angular.z = -self.angular_scale_obstacle
        
        else :
            erreur = dist_gauche - dist_droite
            cmd.linear.x = self.linear_scale
            cmd.angular.z = self.angular_scale * erreur
            
            cmd.angular.z = max(min(cmd.angular.z, 1.5), -1.5) # clamp vitesse angulaire

        self.cmd_pub.publish(cmd)


        now = self.get_clock().now()
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 5e8:  # 0.5 s
            error = dist_gauche - dist_droite
            self.get_logger().info(
                f'gauche={dist_gauche:.3f} droite={dist_droite:.3f} '
                f'avant={diste_avant:.3f} erreur={error:.3f}'
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



    def get_valeurs (self, scan, deg_debut, deg_fin) :
        """
        Retourne toutes les distances du LiDAR dans une certaine direction (angle)
        ex : toutes les distances entre 70° et 110° (mur gauche)
        """
        
        # conversion en radians
        rad_debut = math.radians(deg_debut)
        rad_fin = math.radians(deg_fin)
        
        if rad_debut > rad_fin :
            rad_debut, rad_fin = rad_fin, rad_debut

        valeurs = []

        # on boucle sur toutes les mesures
        for i,r in enumerate(scan.ranges) :
            
            angle = scan.angle_min + i * scan.anngle_increment
            
            if rad_debut <= angle <= rad_fin : # on filtre la zone
                if math.isfinite(r) and scan.range_min < r < scan.range_max :
                    valeurs.append(r)

        return valeurs




def main(args=None) :

    rclpy.init(args=args)
    node = CorridorNode()

    try :
        rclpy.spin(Node)

    except KeyboardInterrupt :
        pass

    finally :
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()