import numpy as np
import cv2
import matplotlib.pyplot as plt


class LaneDetector:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.left_fit_sliding = None
        self.right_fit_sliding = None
        self.left_fit_world_sliding = None
        self.right_fit_world_sliding = None
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        self.ema_alpha = 0.85

    def reset(self):
        self.left_fit_sliding = None
        self.right_fit_sliding = None

    def calibrate(self, images):
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                # img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                # plt.imshow(img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
        self.mtx = mtx
        self.dist = dist

    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Grayscale
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        # np.pi/10, np.pi - np.pi/10
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Apply each of the thresholding functions
    def binary_image(self, img):
        ksize = 7  # Choose a larger odd number to smooth gradient measurements
        img = np.copy(img)
        gradx = self.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 255))
        mag_binary = self.mag_thresh(img, sobel_kernel=ksize, mag_thresh=(10, 255))
        dir_binary = self.dir_threshold(img, sobel_kernel=ksize, thresh=(0.0 * np.pi, np.pi * 0.4))

        combined = np.zeros_like(dir_binary)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        combined[
            ((s_channel >= 15) & (h_channel >= 10) & (h_channel <= 80) | ((l_channel >= 170) & (l_channel <= 255)) & (
                    mag_binary == 1) & (dir_binary == 1) & (gradx == 1))] = 1

        return combined

    def warp_matrices(self, image):
        h = image.shape[0]
        w = image.shape[1]
        trapezoid_top = 456
        trapezoid_top_margin = 595
        trapezoid_bottom_margin = 253

        trapezoid = np.float32([[trapezoid_bottom_margin, h], [trapezoid_top_margin, trapezoid_top],
                                [w - trapezoid_top_margin, trapezoid_top], [w - trapezoid_bottom_margin, h]])
        new_top_left = np.array([trapezoid[0, 0], 0])
        new_top_right = np.array([trapezoid[3, 0], 0])
        offset = [50, 0]

        src = np.float32([trapezoid[0], trapezoid[1], trapezoid[2], trapezoid[3]])
        dst = np.float32([trapezoid[0] + offset, new_top_left + offset, new_top_right - offset, trapezoid[3] - offset])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def warp_perspective(self, image, m):
        return cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]))

    def hist(self, img):
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = img[img.shape[0] // 2:, :]

        # Sum across image pixels vertically - make sure to set an `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        return histogram

    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = self.hist(binary_warped)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_lane_lines(self, warped):
        if self.left_fit_sliding is None or self.right_fit_sliding is None:
            return self.fit(warped)
        else:
            return self.search_around_previous_fit(warped)

    def fit(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]


        return ploty, left_fitx, right_fitx, out_img

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):

        if lefty.size != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fit_world = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        else:
            left_fit = self.left_fit_sliding
            left_fit_world = self.left_fit_world_sliding

        if righty.size != 0:
            right_fit = np.polyfit(righty, rightx, 2)
            right_fit_world = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)
        else:
            right_fit = self.right_fit_sliding
            right_fit_world = self.right_fit_world_sliding

        if self.left_fit_sliding is None or self.right_fit_sliding is None:
            self.left_fit_sliding = left_fit
            self.right_fit_sliding = right_fit

            self.left_fit_world_sliding = left_fit_world
            self.right_fit_world_sliding = right_fit_world
        else:
            alpha = self.ema_alpha
            one_minus_alpha = (1 - alpha)

            self.left_fit_sliding = self.left_fit_sliding * alpha + left_fit * one_minus_alpha
            self.right_fit_sliding = self.right_fit_sliding * alpha + right_fit * one_minus_alpha

            self.left_fit_world_sliding = self.left_fit_world_sliding * alpha + left_fit * one_minus_alpha
            self.right_fit_world_sliding = self.right_fit_world_sliding * alpha + right_fit * one_minus_alpha

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        try:
            left_fitx = self.left_fit_sliding[0] * ploty ** 2 + self.left_fit_sliding[1] * ploty + \
                        self.left_fit_sliding[2]
            right_fitx = self.right_fit_sliding[0] * ploty ** 2 + self.right_fit_sliding[1] * ploty + \
                         self.right_fit_sliding[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        return left_fitx, right_fitx, ploty

    def search_around_previous_fit(self, binary_warped):
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
                (nonzerox > (self.left_fit_sliding[0] * (nonzeroy ** 2) + self.left_fit_sliding[1] * nonzeroy +
                             self.left_fit_sliding[2] - margin)) & (
                        nonzerox < (self.left_fit_sliding[0] * (nonzeroy ** 2) +
                                    self.left_fit_sliding[1] * nonzeroy + self.left_fit_sliding[
                                        2] + margin)))
        right_lane_inds = (
                (nonzerox > (self.right_fit_sliding[0] * (nonzeroy ** 2) + self.right_fit_sliding[1] * nonzeroy +
                             self.right_fit_sliding[2] - margin)) & (
                        nonzerox < (self.right_fit_sliding[0] * (nonzeroy ** 2) +
                                    self.right_fit_sliding[1] * nonzeroy + self.right_fit_sliding[
                                        2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##

        return ploty, left_fitx, right_fitx, result

    def draw_lane_data(self, image, ploty):
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (
                2 * self.left_fit_world_sliding[0] * y_eval * self.ym_per_pix + self.left_fit_world_sliding[
            1]) ** 2) ** 1.5) / np.absolute(
            2 * self.left_fit_world_sliding[0])
        right_curverad = ((1 + (
                2 * self.right_fit_world_sliding[0] * y_eval * self.ym_per_pix + self.right_fit_world_sliding[
            1]) ** 2) ** 1.5) / np.absolute(
            2 * self.right_fit_world_sliding[0])

        curv_rad = (left_curverad + right_curverad) / 2

        cv2.putText(image, 'Radius of Curvature = {:.2f}(m)'.format(curv_rad), (32, 64), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        h = image.shape[0]
        w = image.shape[1]
        center_horizontal = w / 2

        left_fit_bottom = self.left_fit_sliding[0] * h ** 2 + self.left_fit_sliding[1] * h + \
                          self.left_fit_sliding[2]
        right_fit_bottom = self.right_fit_sliding[0] * h ** 2 + self.right_fit_sliding[1] * h + \
                           self.right_fit_sliding[2]

        lane_center = (left_fit_bottom + right_fit_bottom) / 2

        # Debug
        # image = cv2.line(image, (int(left_fit_bottom), h - 5), (int(lane_center), h - 5), [255, 0, 0], 10)

        dist_from_lane_center = (center_horizontal - lane_center) * self.xm_per_pix

        if dist_from_lane_center > 0:
            direction = 'right'
        else:
            direction = 'left'

        cv2.putText(image, 'Vehicle is {:04.3f}m '.format(abs(dist_from_lane_center)) + direction + ' of center',
                    (32, 96), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image
