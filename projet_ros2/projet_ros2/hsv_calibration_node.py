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

        # Create windows
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("HSV Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HSV Calibration", 900, 600)  # ← fenêtre plus large

        # --- Red trackbars ---
        cv2.createTrackbar("Red H min1", "HSV Calibration", 0,   179, self.nothing)
        cv2.createTrackbar("Red H max1", "HSV Calibration", 10,  179, self.nothing)
        cv2.createTrackbar("Red H min2", "HSV Calibration", 160, 179, self.nothing)
        cv2.createTrackbar("Red H max2", "HSV Calibration", 179, 179, self.nothing)
        cv2.createTrackbar("Red S min",  "HSV Calibration", 80,  255, self.nothing)
        cv2.createTrackbar("Red S max",  "HSV Calibration", 255, 255, self.nothing)
        cv2.createTrackbar("Red V min",  "HSV Calibration", 80,  255, self.nothing)
        cv2.createTrackbar("Red V max",  "HSV Calibration", 255, 255, self.nothing)

        # --- Green trackbars ---
        cv2.createTrackbar("Green H min", "HSV Calibration", 30,  179, self.nothing)
        cv2.createTrackbar("Green H max", "HSV Calibration", 85,  179, self.nothing)
        cv2.createTrackbar("Green S min", "HSV Calibration", 80,  255, self.nothing)
        cv2.createTrackbar("Green S max", "HSV Calibration", 255, 255, self.nothing)
        cv2.createTrackbar("Green V min", "HSV Calibration", 80,  255, self.nothing)
        cv2.createTrackbar("Green V max", "HSV Calibration", 255, 255, self.nothing)

        cv2.createTrackbar("Preview: 0=R 1=G 2=Both", "HSV Calibration", 2, 2, self.nothing)

        # Placeholder pour forcer l'affichage des labels
        placeholder = np.zeros((50, 900, 3), dtype=np.uint8)
        cv2.imshow("HSV Calibration", placeholder)
        cv2.waitKey(1)
        cv2.waitKey(1)

        self.get_logger().info("HSV Calibration node started. Press 'p' to print current values, 'q' to quit.")

    def nothing(self, x):
        pass

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is not None:
            self.image = image.copy()

    def get_trackbar_values(self):
        r_h_min1 = cv2.getTrackbarPos("Red H min1", "HSV Calibration")
        r_h_max1 = cv2.getTrackbarPos("Red H max1", "HSV Calibration")
        r_h_min2 = cv2.getTrackbarPos("Red H min2", "HSV Calibration")
        r_h_max2 = cv2.getTrackbarPos("Red H max2", "HSV Calibration")
        r_s_min  = cv2.getTrackbarPos("Red S min",  "HSV Calibration")
        r_s_max  = cv2.getTrackbarPos("Red S max",  "HSV Calibration")
        r_v_min  = cv2.getTrackbarPos("Red V min",  "HSV Calibration")
        r_v_max  = cv2.getTrackbarPos("Red V max",  "HSV Calibration")

        g_h_min  = cv2.getTrackbarPos("Green H min", "HSV Calibration")
        g_h_max  = cv2.getTrackbarPos("Green H max", "HSV Calibration")
        g_s_min  = cv2.getTrackbarPos("Green S min", "HSV Calibration")
        g_s_max  = cv2.getTrackbarPos("Green S max", "HSV Calibration")
        g_v_min  = cv2.getTrackbarPos("Green V min", "HSV Calibration")
        g_v_max  = cv2.getTrackbarPos("Green V max", "HSV Calibration")

        preview  = cv2.getTrackbarPos("Preview: 0=R 1=G 2=Both", "HSV Calibration")

        return {
            "red1":   (r_h_min1, r_h_max1, r_s_min, r_s_max, r_v_min, r_v_max),
            "red2":   (r_h_min2, r_h_max2, r_s_min, r_s_max, r_v_min, r_v_max),
            "green":  (g_h_min,  g_h_max,  g_s_min, g_s_max, g_v_min, g_v_max),
            "preview": preview
        }

    def print_values(self, vals):
        r1 = vals["red1"]
        r2 = vals["red2"]
        g  = vals["green"]
        print("\n--- Current HSV values ---")
        print(f"lower_red1  = np.array([{r1[0]}, {r1[2]}, {r1[4]}])")
        print(f"upper_red1  = np.array([{r1[1]}, {r1[3]}, {r1[5]}])")
        print(f"lower_red2  = np.array([{r2[0]}, {r2[2]}, {r2[4]}])")
        print(f"upper_red2  = np.array([{r2[1]}, {r2[3]}, {r2[5]}])")
        print(f"lower_green = np.array([{g[0]},  {g[2]},  {g[4]}])")
        print(f"upper_green = np.array([{g[1]},  {g[3]},  {g[5]}])")
        print("--------------------------\n")

    def run(self):
        rate = self.create_rate(20)

        while rclpy.ok():
            image = self.image
            if image is not None:
                vals = self.get_trackbar_values()
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                r1 = vals["red1"]
                r2 = vals["red2"]
                g  = vals["green"]

                mask_red1  = cv2.inRange(image_hsv,
                    np.array([r1[0], r1[2], r1[4]]),
                    np.array([r1[1], r1[3], r1[5]]))
                mask_red2  = cv2.inRange(image_hsv,
                    np.array([r2[0], r2[2], r2[4]]),
                    np.array([r2[1], r2[3], r2[5]]))
                mask_green = cv2.inRange(image_hsv,
                    np.array([g[0],  g[2],  g[4]]),
                    np.array([g[1],  g[3],  g[5]]))

                mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                preview = vals["preview"]
                if preview == 0:
                    mask = mask_red
                elif preview == 1:
                    mask = mask_green
                else:
                    mask = cv2.bitwise_or(mask_red, mask_green)

                result = cv2.bitwise_and(image, image, mask=mask)

                cv2.imshow("Original", image)
                cv2.imshow("Mask", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.print_values(self.get_trackbar_values())
            elif key == ord('q'):
                break

            rate.sleep()


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


if __name__ == '__main__':
    main()