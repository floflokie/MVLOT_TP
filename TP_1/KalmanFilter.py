import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        """
        The class will be initialized with six parameters
        :param dt: time for one cycle used to estimate state (sampling time)
        :param u_x: accelerations in the x-direction
        :param u_y: accelerations in the y-direction
        :param std_acc: process noise magnitude
        :param x_sdt_meas: standard deviations of the measurement in the x-direction
        :param y_sdt_meas: standard deviations of the measurement in the y-direction
        """
        super(KalmanFilter).__init__()
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_sdt_meas = x_sdt_meas
        self.y_sdt_meas = y_sdt_meas

        # control input variables u = [u_x, u_y]
        self.u = np.array([[self.u_x], [self.u_y]])
        # initial state matrix (𝑥̂)=[x0=0, y0=0, vx=0,vy=0]
        self.x = np.array([[0], [0], [0], [0]])
        # Matrices describing the system model A,B with respect to the sampling time dt (∆t) :
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[(1 / 2) * (self.dt ** 2), 0],
                           [0, (1 / 2) * (self.dt ** 2)],
                           [self.dt, 0],
                           [0, self.dt]])
        # Measurement mapping matrix H
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # Initial process noise covariance matrix Q with respect to the standard deviation of acceleration (std_acc) σa :
        self.Q = np.array([[(1 / 4) * (self.dt ** 4), 0, (1 / 2) * (self.dt ** 3), 0],
                           [0, (1 / 4) * (self.dt ** 4), 0, (1 / 2) * (self.dt ** 3)],
                           [(1 / 2) * (self.dt ** 3), 0, self.dt ** 2, 0],
                           [0, (1 / 2) * (self.dt ** 3), 0, self.dt ** 2]]) * (self.std_acc ** 2)
        # Initial measurement noise covariance R. Suppose that the measurements z (x, y) are both independent
        # (so that covariance x and y is 0), and look only the variance in the x and y: x_sdt_meas =𝜎𝑥2 , y_sdt_meas=𝜎𝑦2
        self.R = np.array([[self.x_sdt_meas**2, 0],
                           [0, self.y_sdt_meas**2]])
        # Initialize covariance matrix P for prediction error as an identity matrix whose shape is the same as
        # the shape of the matrix A.
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        """
        This function does the prediction of the state estimate 𝑥̂𝑘− and the error prediction 𝑃𝑘− . This task also
        call the time update process (u) because it projects forward the current state to the next time step.
        :return:
        """
        # Update time state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[:2]

    def update(self, z):
        """
        This function takes measurements 𝑧𝑘 as input (centroid coordinates x,y of detected circles)
        :param z: measurements 𝑧𝑘
        :return:
        """
        # Compute Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Update the predicted state estimate 𝑥̂𝑘 and predicted error covariance 𝑃𝑘

        # take the first 2 of x
        self.x = self.x + np.dot(K, z - np.dot(self.H, self.x))
        I = np.eye(self.P.shape[1])
        self.P = (I - np.dot(K, self.H)) * self.P
        return self.x[:2]
