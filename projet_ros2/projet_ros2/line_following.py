import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class CompressedImageSubscriber(Node):
    def __init__(self):
        super().__init__('compressed_image_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.listener_callback,
            10
        )
        self.subscription  # to prevent unused variable warning

    def listener_callback(self, msg):
        # Convertir les données compressées en tableau numpy
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Décoder l'image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is not None:
            #cv2.imshow("Compressed Image", image)
            #cv2.waitKey(1)
            
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

            # Combiner les deux masques
            mask_combined = cv2.bitwise_or(mask_green, mask_red)

            #cv2.imshow("Original", image)
            #cv2.imshow("Lignes vertes", mask_green)
            #cv2.imshow("Lignes rouges", mask_red)
            cv2.imshow("Toutes les lignes", mask_combined)
            cv2.waitKey(1)
            else:
                self.get_logger().warn("Failed to decode compressed image")

def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()