import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
import cv2
import numpy as np


class GoalScorer(Node):

    def __init__(self):
        super().__init__('goal_scorer')

        self.image_sub = self.create_subscription(
            CompressedImage,
            'camera/image_raw/compressed',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.mode = "SEARCH_BALL"
        self.front_dist = None

        # différentes vitesses
        self.search_angular = 0.12
        self.approach_speed = 0.045
        self.push_speed = 0.09
        self.goal_search_angular = 0.06

        # pour la correction
        self.k_ball = 0.003
        self.k_goal = 0.0025

        self.dead_zone_ball = 35
        self.min_ball_area = 30

        self.push_duration_sec = 15.0
        self.push_start_time = None

        # pour quand on a perdu le gaol
        self.goal_lost_counter = 0
        self.goal_lost_max = 25

        # pour quand on a perdu la balle
        self.last_ball_error = 0.0
        self.last_goal_error = 0.0

        self.goal_search_max_time1 = 10.0 
        self.goal_search_max_time2 = 20.0 # le double dans l'autre sens pour qu'il ne s'arrête pas là où il a commencé
        
        self.goal_search_start_time = None
        self.goal_search_direction = 1.0
        self.dead_zone_goal = 45

        # recul
        self.backup_speed = -0.035
        self.backup_duration_sec = 1.2
        self.backup_start_time = None
        
        self.dead_zone_goal = 45

        self.get_logger().info("goal_scorer démarré")



    def secu(self, values):
        vals = [] # contient les mesures LiDAR valides
        for v in values:
            if math.isnan(v) or math.isinf(v):
                continue
            if 0.03 < v < 3.5: # on conserve les valeurs dans une plage réaliste du capteur
                vals.append(v)

        if not vals:
            return None

        vals.sort() # tri
        return vals[len(vals) // 2] # retour de la médiane



    def scan_callback(self, msg: LaserScan):
        self.front_dist = self.secu(
            list(msg.ranges[345:360]) + list(msg.ranges[0:15])
        )



    def make_ball_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_ball = np.array([20, 35, 100])
        upper_ball = np.array([75, 255, 255])

        mask = cv2.inRange(hsv, lower_ball, upper_ball)

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_big = np.ones((7, 7), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
        mask = cv2.dilate(mask, kernel_small, iterations=1)

        return mask


    def make_red_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Rouge plus tolérant si les poteaux paraissent pâles
        lower_red1 = np.array([0, 35, 35])
        upper_red1 = np.array([12, 255, 255])

        lower_red2 = np.array([155, 35, 35])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_big = np.ones((7, 7), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)

        return mask

    def detect_ball(self, image, mask):
        h, w = image.shape[:2]

        contours, _ = cv2.findContours( # recherche des contours dans le masque de la balle
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best = None
        best_score = -1.0

        # analyse de chaque constour rencontré
        for c in contours:
            area = cv2.contourArea(c)  # aire du contour
            if area < self.min_ball_area:  # ignore le bruit 
                continue

            x, y, bw, bh = cv2.boundingRect(c)  # rectangle englobant
            if bw <= 0 or bh <= 0: # sécurité
                continue

            aspect = bw / float(bh) # Rapport largeur / hauteur, permet de rejeter certaines formes aberrantes
            extent = area / float(bw * bh) # Taux de remplissage du rectangle, plus la valeur est élevée, plus l'objet est compact
            cy = y + bh // 2 # position verticale du centre

            if aspect < 0.25 or aspect > 4.0: # ignore formes trop extrêmes
                continue
            if extent < 0.10:  # ignore formes bruitées
                continue
            if cy < 0.25 * h:  # ignore objets trop hauts dans l'image
                continue

            score = area + 700.0 * extent + 0.8 * cy # utilisé pour choisir le meilleur candidat

            if score > best_score:
                best_score = score
                best = (x, y, bw, bh, area, extent)

        if best is None:
            return None

        x, y, bw, bh, area, extent = best  # extraction du meilleur candidat
        # centre de la balle
        cx = x + bw // 2
        cy = y + bh // 2
        error = cx - w // 2 # Erreur horizontale par rapport au centre image

        return {
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "cx": cx,
            "cy": cy,
            "area": area,
            "extent": extent,
            "error": error
        }

    
    def detect_goal(self, image, red_mask):
        h, w = image.shape[:2]

        # on cherche surtout dans la partie haute/milieu.
        roi_y1 = 0
        roi_y2 = int(0.75 * h)

        mask_roi = red_mask[roi_y1:roi_y2, :]

        # detection des contours rouges
        contours, _ = cv2.findContours(
            mask_roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        posts = [] # liste des poteaux détectés

        for c in contours: # même idée que pour self.detect_ball()
            area = cv2.contourArea(c)
            if area < 60:
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            if bw <= 0 or bh <= 0:
                continue

            aspect = bh / float(bw)

            if aspect < 1.15:
                continue

            if bh < 18:
                continue

            posts.append({
                "x": x,
                "y": y + roi_y1,
                "w": bw,
                "h": bh,
                "cx": x + bw // 2,
                "cy": y + roi_y1 + bh // 2,
                "area": area
            })

        if len(posts) < 2:
            return None

        posts = sorted(posts, key=lambda p: p["area"], reverse=True)[:2]
        posts = sorted(posts, key=lambda p: p["cx"])

        left = posts[0]
        right = posts[1]

        goal_cx = (left["cx"] + right["cx"]) // 2
        goal_error = goal_cx - w // 2

        return {
            "left": left,
            "right": right,
            "cx": goal_cx,
            "error": goal_error
        }


    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                return

            h, w = image.shape[:2]

            ball_mask = self.make_ball_mask(image)
            red_mask = self.make_red_mask(image)

            ball = self.detect_ball(image, ball_mask)
            goal = self.detect_goal(image, red_mask)

            cmd = Twist()

            if ball is not None:
                self.last_ball_error = ball["error"]

            if goal is not None:
                self.last_goal_error = goal["error"]

            # 1) SEARCH_BALL
            if self.mode == "SEARCH_BALL":
                if ball is not None:
                    self.mode = "APPROACH_BALL"
                else:
                    cmd.linear.x = 0.0
                    cmd.angular.z = -self.search_angular


            # 2) APPROACH_BALL
            elif self.mode == "APPROACH_BALL":
                if ball is not None:
                    ball_error = ball["error"]

                    if abs(ball_error) < self.dead_zone_ball:
                        cmd.linear.x = self.approach_speed
                        cmd.angular.z = 0.0
                    else:
                        cmd.linear.x = 0.025
                        cmd.angular.z = -self.k_ball * ball_error
                        cmd.angular.z = max(min(cmd.angular.z, 0.22), -0.22)

                else:
                    self.mode = "BACK_UP"
                    self.backup_start_time = self.get_clock().now()

                    cmd.linear.x = self.backup_speed
                    cmd.angular.z = 0.0

            # 3) BACK_UP
            elif self.mode == "BACK_UP":
                cmd.linear.x = self.backup_speed
                cmd.angular.z = 0.0

                elapsed = (
                    self.get_clock().now() - self.backup_start_time
                ).nanoseconds / 1e9

                if elapsed > self.backup_duration_sec:
                    self.mode = "SEARCH_GOAL"
                    self.goal_search_start_time = self.get_clock().now()
                    self.goal_search_direction = 1.0


            # 4) SEARCH_GOAL
            elif self.mode == "SEARCH_GOAL":
                elapsed = (
                    self.get_clock().now() - self.goal_search_start_time
                ).nanoseconds / 1e9

                if goal is not None:
                    goal_error = goal["error"]

                    if abs(goal_error) < self.dead_zone_goal:
                        self.mode = "PUSH"
                        self.push_start_time = self.get_clock().now()
                        self.goal_lost_counter = 0

                        cmd.linear.x = self.push_speed
                        cmd.angular.z = 0.0
                    else:
                        cmd.linear.x = 0.0
                        cmd.angular.z = -self.k_goal * goal_error
                        cmd.angular.z = max(min(cmd.angular.z, 0.08), -0.08)

                else:
                    cmd.linear.x = 0.0

                    # scan limité : droite puis gauche, pas rotation infinie
                    if elapsed < self.goal_search_max_time1:
                        cmd.angular.z = -self.goal_search_direction * self.goal_search_angular
                    elif elapsed < 2 * self.goal_search_max_time2:
                        cmd.angular.z = self.goal_search_direction * self.goal_search_angular
                    else:
                        cmd.angular.z = 0.0


            # 3) PUSH
            elif self.mode == "PUSH":
                cmd.linear.x = self.push_speed

                if goal is not None:
                    self.goal_lost_counter = 0
                    self.last_goal_error = goal["error"]
                    cmd.angular.z = -self.k_goal * goal["error"]
                else:
                    self.goal_lost_counter += 1
                    cmd.angular.z = 0.0

                cmd.angular.z = max(min(cmd.angular.z, 0.08), -0.08)

                elapsed = (
                    self.get_clock().now() - self.push_start_time
                ).nanoseconds / 1e9

                if goal is None and elapsed > self.push_duration_sec and self.goal_lost_counter > self.goal_lost_max:
                    self.mode = "DONE"
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0

        

            # =====================================================
            # 4) DONE
            # =====================================================
            elif self.mode == "DONE":
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

            self.cmd_pub.publish(cmd)

            # DEBUG VISUEL
            if ball is not None:
                cv2.rectangle(
                    image,
                    (ball["x"], ball["y"]),
                    (ball["x"] + ball["w"], ball["y"] + ball["h"]),
                    (0, 255, 0),
                    2
                )
                cv2.circle(image, (ball["cx"], ball["cy"]), 6, (0, 0, 255), -1)

            if goal is not None:
                left = goal["left"]
                right = goal["right"]

                cv2.rectangle(
                    image,
                    (left["x"], left["y"]),
                    (left["x"] + left["w"], left["y"] + left["h"]),
                    (0, 0, 255),
                    2
                )

                cv2.rectangle(
                    image,
                    (right["x"], right["y"]),
                    (right["x"] + right["w"], right["y"] + right["h"]),
                    (0, 0, 255),
                    2
                )

                cv2.circle(image, (goal["cx"], h // 2), 7, (255, 0, 255), -1)

            cv2.putText(
                image,
                f"MODE: {self.mode}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if self.front_dist is not None:
                cv2.putText(
                    image,
                    f"lidar_front={self.front_dist:.2f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.imshow("Goal scorer", image)
            cv2.imshow("Ball mask", ball_mask)
            cv2.imshow("Goal red mask", red_mask)
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