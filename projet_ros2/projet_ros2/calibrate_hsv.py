"""
Outil de calibration HSV Multi-couleur (Rouge, Vert, Bleu) en temps réel.
Usage : python3 calibrate_multi_hsv.py
- Utilisez le slider 'Color' pour choisir la couleur à régler (0=Rouge, 1=Vert, 2=Bleu)
- Ajustez les H/S/V pour cette couleur
- Le fenêtre 'Multi-Mask' montre les 3 détections simultanément
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class MultiHSVCalibrator(Node):
    def __init__(self):
        super().__init__('multi_hsv_calibrator')
        self.sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.callback, 10)

        # Structure pour stocker les seuils [Hmin, Hmax, Smin, Smax, Vmin, Vmax]
        # Initialisés avec des valeurs par défaut larges
        self.thresholds = [
            [0, 10, 100, 255, 100, 255],   # 0: Rouge
            [40, 80, 100, 255, 100, 255],  # 1: Vert
            [100, 130, 100, 255, 100, 255] # 2: Bleu
        ]
        self.current_color_idx = 0

        cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Multi-Mask', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Original ROI', cv2.WINDOW_NORMAL)

        # Slider de sélection de la couleur
        cv2.createTrackbar('Color (0:R, 1:G, 2:B)', 'Control Panel', 0, 2, self.on_change_color)
        
        # Sliders HSV standards
        cv2.createTrackbar('H min', 'Control Panel', 0, 179, lambda x: None)
        cv2.createTrackbar('H max', 'Control Panel', 179, 179, lambda x: None)
        cv2.createTrackbar('S min', 'Control Panel', 0, 255, lambda x: None)
        cv2.createTrackbar('S max', 'Control Panel', 255, 255, lambda x: None)
        cv2.createTrackbar('V min', 'Control Panel', 0, 255, lambda x: None)
        cv2.createTrackbar('V max', 'Control Panel', 255, 255, lambda x: None)
        
        cv2.createTrackbar('ROI top %', 'Control Panel', 50, 100, lambda x: None)

        # Initialiser les sliders avec les valeurs du Rouge (index 0)
        self.update_sliders_from_memory(0)

    def on_change_color(self, val):
        # Avant de changer, on pourrait sauvegarder, mais la sauvegarde est faite en temps réel dans le callback
        self.current_color_idx = val
        self.update_sliders_from_memory(val)
        colors = ["ROUGE", "VERT", "BLEU"]
        self.get_logger().info(f"Calibrage en cours pour : {colors[val]}")

    def update_sliders_from_memory(self, idx):
        vals = self.thresholds[idx]
        cv2.setTrackbarPos('H min', 'Control Panel', vals[0])
        cv2.setTrackbarPos('H max', 'Control Panel', vals[1])
        cv2.setTrackbarPos('S min', 'Control Panel', vals[2])
        cv2.setTrackbarPos('S max', 'Control Panel', vals[3])
        cv2.setTrackbarPos('V min', 'Control Panel', vals[4])
        cv2.setTrackbarPos('V max', 'Control Panel', vals[5])

    def callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None: return

        h, w = image.shape[:2]
        roi_top_pct = cv2.getTrackbarPos('ROI top %', 'Control Panel')
        roi_top = int(h * roi_top_pct / 100)
        roi = image[roi_top:h, 0:w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 1. Mettre à jour la mémoire pour la couleur active avec les sliders actuels
        idx = self.current_color_idx
        self.thresholds[idx] = [
            cv2.getTrackbarPos('H min', 'Control Panel'),
            cv2.getTrackbarPos('H max', 'Control Panel'),
            cv2.getTrackbarPos('S min', 'Control Panel'),
            cv2.getTrackbarPos('S max', 'Control Panel'),
            cv2.getTrackbarPos('V min', 'Control Panel'),
            cv2.getTrackbarPos('V max', 'Control Panel')
        ]

        # 2. Calculer les masques pour les 3 couleurs et créer une image combinée
        combined_display = np.zeros_like(roi)
        mask_colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # R, G, B en BGR

        for i in range(3):
            lower = np.array([self.thresholds[i][0], self.thresholds[i][2], self.thresholds[i][4]])
            upper = np.array([self.thresholds[i][1], self.thresholds[i][3], self.thresholds[i][5]])
            mask = cv2.inRange(hsv_roi, lower, upper)
            
            # Colorer le masque pour la visualisation combinée
            colored_mask = cv2.bitwise_and(
                np.full_like(roi, mask_colors_bgr[i]), 
                np.full_like(roi, mask_colors_bgr[i]), 
                mask=mask
            )
            combined_display = cv2.addWeighted(combined_display, 1.0, colored_mask, 1.0, 0)

            # Optionnel : Dessiner le centroïde de la couleur active sur l'image originale
            if i == self.current_color_idx:
                M = cv2.moments(mask)
                if M['m00'] > 500:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(roi, (cx, cy), 10, (255, 255, 255), 2)
                    cv2.putText(roi, "TARGET", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Original ROI', roi)
        cv2.imshow('Multi-Mask', combined_display)
        
        # Affichage des valeurs dans la console au clic ou changement pour aider au copier-coller
        if cv2.waitKey(1) & 0xFF == ord('s'):
            self.get_logger().info(f"CONFIG ACTUELLE : R:{self.thresholds[0]} G:{self.thresholds[1]} B:{self.thresholds[2]}")

def main(args=None):
    rclpy.init(args=args)
    node = MultiHSVCalibrator()
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