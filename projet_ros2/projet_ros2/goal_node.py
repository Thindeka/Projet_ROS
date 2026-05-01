import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import math


class GoalScorer(Node):

    def __init__(self):
        super().__init__('goal_scorer')

        self.image_sub = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.linear_speed = 0.06
        self.push_speed = 0.08
        self.angular_gain = 0.003
        self.push_angular_gain = 0.0015
        self.dead_zone = 25

        self.min_ball_area = 250
        self.push_area_threshold = 25000

        self.get_logger().info("goal_scorer démarré")

    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                return

            h, w = image.shape[:2]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_ball = np.array([22, 45, 35])
            upper_ball = np.array([85, 255, 255])

            mask = cv2.inRange(hsv, lower_ball, upper_ball)

            kernel_small = np.ones((5, 5), np.uint8)
            kernel_big = np.ones((13, 13), np.uint8)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
            mask = cv2.dilate(mask, kernel_small, iterations=1)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cmd = Twist()
            mode = "SEARCH"

            best = None
            best_score = -1.0

            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_ball_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh)

                if aspect < 0.45 or aspect > 1.8:
                    continue

                (cx_f, cy_f), radius = cv2.minEnclosingCircle(c)

                if radius < 8:
                    continue

                circle_area = math.pi * radius * radius
                fill_ratio = area / circle_area if circle_area > 0 else 0.0

                # Une balle avec ombre peut avoir un fill_ratio faible.
                # Une ligne fine aura un fill_ratio très faible + forme allongée.
                if fill_ratio < 0.25:
                    continue

                score = area + 300.0 * fill_ratio + 5.0 * radius

                if score > best_score:
                    best_score = score
                    best = (c, int(cx_f), int(cy_f), int(radius), area, fill_ratio)

            if best is not None:
                c, cx, cy, radius, area, fill_ratio = best
                error = cx - w // 2

                if area < self.push_area_threshold:
                    mode = "APPROACH"

                    if abs(error) < self.dead_zone:
                        cmd.linear.x = self.linear_speed
                        cmd.angular.z = 0.0
                    else:
                        cmd.linear.x = 0.025
                        cmd.angular.z = -self.angular_gain * error
                else:
                    mode = "PUSH"
                    cmd.linear.x = self.push_speed
                    cmd.angular.z = -self.push_angular_gain * error

                cv2.circle(image, (cx, cy), radius, (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)

                cv2.putText(
                    image,
                    f"{mode} area={area:.0f} fill={fill_ratio:.2f} err={error}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            else:
                mode = "SEARCH"
                cmd.linear.x = 0.0
                cmd.angular.z = 0.12

            self.cmd_pub.publish(cmd)

            cv2.putText(
                image,
                f"MODE: {mode}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Ball detection", image)
            cv2.imshow("Ball mask", mask)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"image_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GoalScorer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()