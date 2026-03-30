import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class LineFollower(Node):


    def __init__(self):
        super().__init__('line_follower')

        # subscription a image_raw
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.listener_callback,
            10
        )
        self.subscription  # to prevent unused variable warning

        # gain proportionnel
        self.kp = 0.003

        # vitesse lineaire
        self.linear_speed = 0.15  # a revoir



    def listener_callback(self, msg):
        
        np_arr = np.frombuffer(msg.data, np.uint8) # consersion des données compressées en tableau numpy
        
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # decodage de l'image

        if image is None:
            self.get_logger().warn("Impossible de décoder l'image")
            return

    
        height, width, _ = image.shape 
        center_x = width // 2  # on recupere le cnetre de l'image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Masque pour le VERT
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Masque pour le ROUGE (le rouge est en deux plages dans HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Combiner les deux masques (rouge + vert)
        mask_combined = cv2.bitwise_or(mask_green, mask_red)

        # ROI : Region Of Interest
        # on ignore le haut de l'image, le robot ne doit suivre que ce qui est devant lui
        # on minimise le bruit/informations pas utiles
        roi_top = int(height * 0.6)
        roi = mask_combined[roi_top:height, :]

        # reduction de bruit 
        kernel = np.ones((5,5), np.uint8)
        # on essaye d'obtenir une ligne propre et continue
        roi = cv2.erode(roi, kernel, iterations=1)  # enleve les petits points blancs parasites
        roi = cv2.dilate(roi, kernel, iterations=2)  # arandit la ligne
        
        # calcul du centroide avec les moments statistiques de l'image
        M = cv2.moements(roi)

        twist = Twist() # message

        # m00 : nombre de pixels
        # m10 : somme des positions en x
        # m01 : somme des positions en y
        if M["m00"] > 0 :  # on evite la division par 0

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        
            error = cx - center_x # cx est dans la ROI, donc même repère x que l'image

            # commande proportionnelle
            twist.linear.x = self.linear_speed
            twist.angular.z = -self.kp * error

            # Dessin pour debug
        debug = image.copy()

        # Rectangle de la zone regardée
        cv2.rectangle(debug, (0, roi_top), (width, height), (255, 0, 0), 2)

        # Centre image
        cv2.line(debug, (center_x, 0), (center_x, height), (255, 255, 0), 2)

        # Centroïde dans l'image complète
        cv2.circle(debug, (cx, cy + roi_top), 8, (0, 255, 255), -1)

        # Ligne entre centre image et centroïde
        cv2.line(debug, (center_x, cy + roi_top), (cx, cy + roi_top), (255, 0, 255), 2)

        cv2.putText(debug, f"Erreur: {error}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(debug, f"angular.z: {twist.angular.z:.3f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Masque ROI", roi)
        cv2.imshow("Debug suivi ligne", debug)
        cv2.waitKey(1)
        self.get_logger().info(
            f"cx={cx}, center={center_x}, error={error}, angular.z={twist.angular.z:.3f}"
        )
    
    else:
        # Si aucune ligne n'est détectée
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.get_logger().warn("Aucune ligne détectée")

    self.cmd_pub.publish(twist)




def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()