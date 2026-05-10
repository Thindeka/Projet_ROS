import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import threading


class HSVCalibration(Node):
    def __init__(self):
        super().__init__('hsv_calibration')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.listener_callback,
            10
        )

        self.image = None

        # Fenêtres séparées : une pour les trackbars, une pour les valeurs texte
        cv2.namedWindow("Original")
        cv2.namedWindow("Mask")
        cv2.namedWindow("Trackbars")   # trackbars uniquement
        cv2.namedWindow("Valeurs")     # texte uniquement — fenêtre indépendante

        # ROUGE
        cv2.createTrackbar("Red H min1", "Trackbars", 0,   179, self.nothing)
        cv2.createTrackbar("Red H max1", "Trackbars", 10,  179, self.nothing)
        cv2.createTrackbar("Red H min2", "Trackbars", 160, 179, self.nothing)
        cv2.createTrackbar("Red H max2", "Trackbars", 179, 179, self.nothing)
        cv2.createTrackbar("Red S min",  "Trackbars", 33,  255, self.nothing)
        cv2.createTrackbar("Red S max",  "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("Red V min",  "Trackbars", 79,  255, self.nothing)
        cv2.createTrackbar("Red V max",  "Trackbars", 255, 255, self.nothing)

        # VERT
        cv2.createTrackbar("Green H min", "Trackbars", 32,  179, self.nothing)
        cv2.createTrackbar("Green H max", "Trackbars", 95,  179, self.nothing)
        cv2.createTrackbar("Green S min", "Trackbars", 44,  255, self.nothing)
        cv2.createTrackbar("Green S max", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("Green V min", "Trackbars", 31,  255, self.nothing)
        cv2.createTrackbar("Green V max", "Trackbars", 255, 255, self.nothing)

        # JAUNE
        cv2.createTrackbar("Yellow H min", "Trackbars", 18,  179, self.nothing)
        cv2.createTrackbar("Yellow H max", "Trackbars", 35,  179, self.nothing)
        cv2.createTrackbar("Yellow S min", "Trackbars", 80,  255, self.nothing)
        cv2.createTrackbar("Yellow S max", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("Yellow V min", "Trackbars", 80,  255, self.nothing)
        cv2.createTrackbar("Yellow V max", "Trackbars", 255, 255, self.nothing)

        # BLEU
        cv2.createTrackbar("Blue H min", "Trackbars", 100, 179, self.nothing)
        cv2.createTrackbar("Blue H max", "Trackbars", 130, 179, self.nothing)
        cv2.createTrackbar("Blue S min", "Trackbars", 80,  255, self.nothing)
        cv2.createTrackbar("Blue S max", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("Blue V min", "Trackbars", 50,  255, self.nothing)
        cv2.createTrackbar("Blue V max", "Trackbars", 255, 255, self.nothing)

        cv2.createTrackbar("Preview 0R 1G 2Y 3B 4All", "Trackbars", 4, 4, self.nothing)

        self.get_logger().info("HSV Calibration démarré. 'p' = print terminal, 'q' = quitter.")

    def nothing(self, x):
        pass

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is not None:
            self.image = image.copy()

    def get_trackbar_values(self):
        def g(name):
            return cv2.getTrackbarPos(name, "Trackbars")

        return {
            "red1":    (g("Red H min1"),   g("Red H max1"),   g("Red S min"), g("Red S max"), g("Red V min"), g("Red V max")),
            "red2":    (g("Red H min2"),   g("Red H max2"),   g("Red S min"), g("Red S max"), g("Red V min"), g("Red V max")),
            "green":   (g("Green H min"),  g("Green H max"),  g("Green S min"), g("Green S max"), g("Green V min"), g("Green V max")),
            "yellow":  (g("Yellow H min"), g("Yellow H max"), g("Yellow S min"), g("Yellow S max"), g("Yellow V min"), g("Yellow V max")),
            "blue":    (g("Blue H min"),   g("Blue H max"),   g("Blue S min"), g("Blue S max"), g("Blue V min"), g("Blue V max")),
            "preview": g("Preview 0R 1G 2Y 3B 4All"),
        }

    def draw_values_window(self, vals):
        """Fenêtre séparée uniquement pour le texte des valeurs.
        On la rend grande pour que le texte soit lisible sans être écrasé par les trackbars."""
        r1, r2, g, y, b = vals["red1"], vals["red2"], vals["green"], vals["yellow"], vals["blue"]

        img = np.zeros((420, 520, 3), dtype=np.uint8)

        lines = [
            ("=== VALEURS COURANTES ===",          (255, 255, 255)),
            ("",                                    (0, 0, 0)),
            (f"RED1  : H {r1[0]:3d}-{r1[1]:3d}  S {r1[2]:3d}-{r1[3]:3d}  V {r1[4]:3d}-{r1[5]:3d}", (80, 80, 255)),
            (f"RED2  : H {r2[0]:3d}-{r2[1]:3d}  S {r2[2]:3d}-{r2[3]:3d}  V {r2[4]:3d}-{r2[5]:3d}", (80, 80, 200)),
            (f"GREEN : H {g[0]:3d}-{g[1]:3d}  S {g[2]:3d}-{g[3]:3d}  V {g[4]:3d}-{g[5]:3d}",        (80, 255, 80)),
            (f"YELLOW: H {y[0]:3d}-{y[1]:3d}  S {y[2]:3d}-{y[3]:3d}  V {y[4]:3d}-{y[5]:3d}",        (0, 220, 255)),
            (f"BLUE  : H {b[0]:3d}-{b[1]:3d}  S {b[2]:3d}-{b[3]:3d}  V {b[4]:3d}-{b[5]:3d}",        (255, 150, 50)),
            ("",                                    (0, 0, 0)),
            ("--- Code pret a copier (p) ---",      (200, 200, 200)),
            (f"RED_LOWER1  = np.array([{r1[0]}, {r1[2]}, {r1[4]}])", (80, 80, 255)),
            (f"RED_UPPER1  = np.array([{r1[1]}, {r1[3]}, {r1[5]}])", (80, 80, 255)),
            (f"RED_LOWER2  = np.array([{r2[0]}, {r2[2]}, {r2[4]}])", (80, 80, 200)),
            (f"RED_UPPER2  = np.array([{r2[1]}, {r2[3]}, {r2[5]}])", (80, 80, 200)),
            (f"GREEN_LOWER = np.array([{g[0]}, {g[2]}, {g[4]}])",    (80, 255, 80)),
            (f"GREEN_UPPER = np.array([{g[1]}, {g[3]}, {g[5]}])",    (80, 255, 80)),
            (f"YELLOW_LOWER= np.array([{y[0]}, {y[2]}, {y[4]}])",    (0, 220, 255)),
            (f"YELLOW_UPPER= np.array([{y[1]}, {y[3]}, {y[5]}])",    (0, 220, 255)),
            (f"BLUE_LOWER  = np.array([{b[0]}, {b[2]}, {b[4]}])",    (255, 150, 50)),
            (f"BLUE_UPPER  = np.array([{b[1]}, {b[3]}, {b[5]}])",    (255, 150, 50)),
            ("",                                    (0, 0, 0)),
            ("'p' print terminal  |  'q' quitter",  (160, 160, 160)),
        ]

        y_pos = 22
        for text, color in lines:
            if text:
                cv2.putText(img, text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            y_pos += 19

        return img

    def run(self):
        rate = self.create_rate(20)
        while rclpy.ok():
            if self.image is not None:
                vals    = self.get_trackbar_values()
                hsv     = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

                r1, r2, g, y, b = vals["red1"], vals["red2"], vals["green"], vals["yellow"], vals["blue"]

                mask_red1   = cv2.inRange(hsv, np.array([r1[0], r1[2], r1[4]]), np.array([r1[1], r1[3], r1[5]]))
                mask_red2   = cv2.inRange(hsv, np.array([r2[0], r2[2], r2[4]]), np.array([r2[1], r2[3], r2[5]]))
                mask_green  = cv2.inRange(hsv, np.array([g[0],  g[2],  g[4]]),  np.array([g[1],  g[3],  g[5]]))
                mask_yellow = cv2.inRange(hsv, np.array([y[0],  y[2],  y[4]]),  np.array([y[1],  y[3],  y[5]]))
                mask_blue   = cv2.inRange(hsv, np.array([b[0],  b[2],  b[4]]),  np.array([b[1],  b[3],  b[5]]))
                mask_red    = cv2.bitwise_or(mask_red1, mask_red2)

                preview = vals["preview"]
                if   preview == 0: mask = mask_red
                elif preview == 1: mask = mask_green
                elif preview == 2: mask = mask_yellow
                elif preview == 3: mask = mask_blue
                else:              mask = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_green),
                                                         cv2.bitwise_or(mask_yellow, mask_blue))

                result = cv2.bitwise_and(self.image, self.image, mask=mask)

                # Centroïdes
                for m, color in [
                    (mask_green,  (0, 255, 0)),
                    (mask_red,    (0, 0, 255)),
                    (mask_yellow, (0, 255, 255)),
                    (mask_blue,   (255, 100, 0)),
                ]:
                    c = self.centroid(m)
                    if c:
                        cv2.circle(result, c, 7, color, -1)

                cv2.imshow("Original", self.image)
                cv2.imshow("Mask",     result)

                # Fenêtre valeurs — complètement séparée des trackbars
                cv2.imshow("Valeurs", self.draw_values_window(vals))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.print_values(self.get_trackbar_values())
            elif key == ord('q'):
                break

            rate.sleep()

    def print_values(self, vals):
        r1, r2, g, y, b = vals["red1"], vals["red2"], vals["green"], vals["yellow"], vals["blue"]
        print("\n--- Code prêt à copier ---")
        print(f"RED_LOWER1   = np.array([{r1[0]}, {r1[2]}, {r1[4]}])")
        print(f"RED_UPPER1   = np.array([{r1[1]}, {r1[3]}, {r1[5]}])")
        print(f"RED_LOWER2   = np.array([{r2[0]}, {r2[2]}, {r2[4]}])")
        print(f"RED_UPPER2   = np.array([{r2[1]}, {r2[3]}, {r2[5]}])")
        print(f"GREEN_LOWER  = np.array([{g[0]}, {g[2]}, {g[4]}])")
        print(f"GREEN_UPPER  = np.array([{g[1]}, {g[3]}, {g[5]}])")
        print(f"YELLOW_LOWER = np.array([{y[0]}, {y[2]}, {y[4]}])")
        print(f"YELLOW_UPPER = np.array([{y[1]}, {y[3]}, {y[5]}])")
        print(f"BLUE_LOWER   = np.array([{b[0]}, {b[2]}, {b[4]}])")
        print(f"BLUE_UPPER   = np.array([{b[1]}, {b[3]}, {b[5]}])")

    def centroid(self, image_bin):
        M = cv2.moments(image_bin)
        if M["m00"] == 0:
            return None
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def main(args=None):
    rclpy.init(args=args)
    node = HSVCalibration()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__== '__main__':
    main()