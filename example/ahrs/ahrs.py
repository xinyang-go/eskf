import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from ahrs import ahrs


def view(result):
    plt.figure(0)
    plt.title("roll")
    plt.plot(result[:, 0], result[:, 1], label="gravity")
    plt.plot(result[:, 0], result[:, 2], label="eskf")
    plt.plot(result[:, 0], result[:, 3], label="reference")
    plt.legend()

    plt.figure(1)
    plt.title("pitch")
    plt.plot(result[:, 0], result[:, 4], label="gravity")
    plt.plot(result[:, 0], result[:, 5], label="eskf")
    plt.plot(result[:, 0], result[:, 6], label="reference")
    plt.legend()

    result[:, 7] += np.mean(result[:, 8] - result[:, 7])
    plt.figure(2)
    plt.title("yaw")
    plt.plot(result[:, 0], result[:, 7], label="eskf")
    plt.plot(result[:, 0], result[:, 8], label="reference")
    plt.legend()

    plt.show()
   

def main():
    kf = ahrs()
    kf.x = np.array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ])
    kf.Q = np.diag([1e0, 1e0, 1e0])
    V = np.diag([1e6, 1e6, 1e6])

    data = np.loadtxt("ahrs.txt")

    last_t = None

    result = []
    for t, ax, ay, az, gx, gy, gz, qx, qy, qz, qw in data:
        acc = np.array([ax, ay, az]) 
        acc /= np.linalg.norm(acc)
        gyr = np.array([gx, gy, gz]) 
        q = np.array([qx, qy, qz, qw])
        Rr = Rotation.from_quat(q).as_matrix().T
        
        if last_t is None:
            last_t = t
            continue
        dt = (t - last_t) * 1e-3
        last_t = t

        kf.process(gyr, dt)
        kf.update(acc, V)
        Rk = kf.x[:9].reshape(3, 3).T

        result.append((
            # timestamp
            t,
            # roll
            np.arctan2(ay, az),
            np.arctan2(Rk[1,2], Rk[2,2]), 
            np.arctan2(Rr[1,2], Rr[2,2]), 
            # pitch
            np.arctan2(-ax, np.sqrt(ay*ay+az*az)),
            np.arctan2(-Rk[0,2], np.sqrt(Rk[1,2]*Rk[1,2]+Rk[2,2]*Rk[2,2])),
            np.arctan2(-Rr[0,2], np.sqrt(Rr[1,2]*Rr[1,2]+Rr[2,2]*Rr[2,2])),
            # yaw
            np.arctan2(Rk[0,1], Rk[0,0]),
            np.arctan2(Rr[0,1], Rr[0,0]),
        ))

    result = np.array(result)
    result[:, 1:9] *= 180 / np.pi
    view(result)


if __name__ == "__main__":
    main()
