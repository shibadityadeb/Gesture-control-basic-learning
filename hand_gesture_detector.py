import cv2
import numpy as np

def detect_hand_gesture(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use the Canny edge detector to find edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    is_open = False  # Default to closed palm

    if contours:
        # Get the largest contour (which should be the hand)
        largest_contour = contours[0]

        # Simplify the contour to avoid self-intersections
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Create a convex hull around the approximated contour
        hull = cv2.convexHull(approx_contour)

        # Draw the convex hull on the original frame
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

        # Validate that convexity defects are computed
        try:
            defects = cv2.convexityDefects(approx_contour, cv2.convexHull(approx_contour, returnPoints=False))
        except cv2.error as e:
            print(f"Error in convexityDefects: {e}")
            defects = None

        if defects is not None:
            is_open = True
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx_contour[s][0])
                end = tuple(approx_contour[e][0])
                far = tuple(approx_contour[f][0])

                # Draw the lines for defects
                cv2.line(frame, start, end, (0, 0, 255), 2)
                cv2.circle(frame, far, 5, (0, 0, 255), -1)

            # If the number of defects is greater than a threshold, assume it's an open palm
            if len(defects) > 4:
                is_open = True
            else:
                is_open = False
        else:
            is_open = False

    return is_open, frame
