import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares
import cv2
import json
import os
from data.loader import DataLoader

class ZhangCalibration:
    def __init__(self, data):
        self.object_points = data['object_points']
        self.image_points = data['image_points']
        self.image_size = data['image_size']
        self.images = data['images']
        self.n_images = len(self.images)

    # Hartley normalization for numerical stability
    def normalize_points(self, points):
        pts = points.reshape(-1, 2)
        centroid = np.mean(pts, axis=0)
        std = np.std(pts)
        scale = np.sqrt(2) / (std + 1e-10)
        T = np.array([[scale, 0, -scale*centroid[0]], 
                      [0, scale, -scale*centroid[1]], 
                      [0, 0, 1]])
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        pts_norm = (T @ pts_h.T).T
        return pts_norm[:, :2], T
    
    # homography estimation using DLT with normalization
    def estimate_homography(self, obj_pts, img_pts):
        img_pts_norm, T_img = self.normalize_points(img_pts)
        obj_pts_norm, T_obj = self.normalize_points(obj_pts[:, :2].reshape(-1, 1, 2))
        
        # construct matrix A
        A = []
        for i in range(len(obj_pts)):
            X, Y = obj_pts_norm[i]
            u, v = img_pts_norm[i]
            A.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
            A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
        
        _, _, Vt = svd(np.array(A))
        H = Vt[-1, :].reshape(3, 3)

        # denormalize
        H = np.linalg.inv(T_img) @ H @ T_obj
        return H / H[2, 2]
    
    # compute intrinsic matrix K from multiple homographies
    def compute_intrinsics(self):
        self.homographies = [self.estimate_homography(self.object_points[i], self.image_points[i]) 
                            for i in range(self.n_images)]
        
        # constraint matrix V
        V = []
        for H in self.homographies:
            h = [H[:, i] for i in range(3)]
            v = lambda i, j: np.array([h[i][0]*h[j][0], 
                                       h[i][0]*h[j][1] + h[i][1]*h[j][0],
                                       h[i][1]*h[j][1],
                                       h[i][2]*h[j][0] + h[i][0]*h[j][2],
                                       h[i][2]*h[j][1] + h[i][1]*h[j][2],
                                       h[i][2]*h[j][2]])
            V.extend([v(0,1), v(0,0) - v(1,1)])
        
        _, _, Vt = svd(np.array(V))
        b = Vt[-1, :]
        B11, B12, B22, B13, B23, B33 = b
        
        # intrinsic parameters
        denom = B11*B22 - B12**2
        v0 = (B12*B13 - B11*B23) / denom
        lambda_ = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
        fx = np.sqrt(abs(lambda_ / B11))
        fy = np.sqrt(abs(lambda_ * B11 / denom))
        cx = -B13 * fx**2 / lambda_
        cy = v0
        
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        print("Intrinsic Matrix (K):")
        print(self.K)
        print(f"\nfx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}\n")

    # extract rotation and translation for each image
    def compute_extrinsics(self):
        K_inv = np.linalg.inv(self.K)
        self.rvecs, self.tvecs = [], []
        
        for H in self.homographies:
            h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
            lambda_ = 1.0 / np.linalg.norm(K_inv @ h1)
            r1, r2 = lambda_ * K_inv @ h1, lambda_ * K_inv @ h2
            r3 = np.cross(r1, r2)
            t = lambda_ * K_inv @ h3
            # enforce orthogonality
            R = np.column_stack([r1, r2, r3])
            U, _, Vt = svd(R)
            R = U @ Vt
            # ensure det(R) = 1
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt

            # convert to rotation vector (axis-angle)
            rvec, _ = cv2.Rodrigues(R)
            
            self.rvecs.append(rvec)
            self.tvecs.append(t.reshape(3, 1))

    # estimate distortion coefficients k1, k2 using linear least squares
    def estimate_distortion(self):
        D, d = [], []
        for i in range(self.n_images):
            rvec_flat = self.rvecs[i].ravel()  # (3,1) → (3,)
            theta = np.linalg.norm(rvec_flat)
            if theta > 1e-10:
                r = rvec_flat / theta
                K_mat = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
                R = np.eye(3) + np.sin(theta)*K_mat + (1-np.cos(theta))*(K_mat @ K_mat)
            else:
                R = np.eye(3)
            
            tvec_flat = self.tvecs[i].ravel()  # (3,1) → (3,)
            obj_h = np.column_stack([self.object_points[i], np.ones(len(self.object_points[i]))])
            p_cam = (np.column_stack([R, tvec_flat]) @ obj_h.T).T
            x, y = p_cam[:, 0]/p_cam[:, 2], p_cam[:, 1]/p_cam[:, 2]
            
            fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
            u_ideal, v_ideal = fx*x + cx, fy*y + cy
            u_obs, v_obs = self.image_points[i][:, 0, 0], self.image_points[i][:, 0, 1]
            
            r2, r4 = x**2 + y**2, (x**2 + y**2)**2
            for j in range(len(x)):
                D.extend([[(u_ideal[j]-cx)*r2[j], (u_ideal[j]-cx)*r4[j]], 
                        [(v_ideal[j]-cy)*r2[j], (v_ideal[j]-cy)*r4[j]]])
                d.extend([u_obs[j]-u_ideal[j], v_obs[j]-v_ideal[j]])
        
        k = np.linalg.lstsq(np.array(D), np.array(d), rcond=None)[0]
        self.dist_coeffs = k
        print(f"Distortion: k1={k[0]:.6f}, k2={k[1]:.6f}\n")
    
    ## non-linear refinement via Levenberg-Marquardt optimization ##
    def refine_parameters(self):
        def pack():
            p = [self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]]
            p.extend(self.dist_coeffs)
            for rv, tv in zip(self.rvecs, self.tvecs):
                p.extend(rv.ravel())
                p.extend(tv.ravel())
            return np.array(p)
        
        def unpack(params):
            K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
            dist = params[4:6]
            rvs, tvs = [], []
            idx = 6
            for _ in range(self.n_images):
                rvs.append(params[idx:idx+3])
                tvs.append(params[idx+3:idx+6])
                idx += 6
            return K, dist, rvs, tvs
        
        def residuals(params):
            K, dist, rvs, tvs = unpack(params)
            res = []
            for i in range(self.n_images):
                # convert to rotation matrix
                R, _ = cv2.Rodrigues(rvs[i])

                # transform to camera coordinates
                obj_h = np.column_stack([self.object_points[i], 
                                         np.ones(len(self.object_points[i]))])
                p = (np.column_stack([R, tvs[i]]) @ obj_h.T).T
                
                # perspective division
                x = p[:, 0] / p[:, 2]
                y = p[:, 1] / p[:, 2]
                
                # apply radial distortion
                r2 = x**2 + y**2
                radial = 1 + dist[0]*r2 + dist[1]*r2**2
                x *= radial
                y *= radial
                
                # apply intrinsics
                u = K[0,0]*x + K[0,2]
                v = K[1,1]*y + K[1,2]
                
                # compute residuals
                obs = self.image_points[i][:, 0, :]
                res.extend((obs - np.column_stack([u, v])).ravel())
            
            return np.array(res)
        
        result = least_squares(residuals, pack(), method='lm', verbose=0)
        self.K, self.dist_coeffs, self.rvecs, self.tvecs = unpack(result.x)
        self.rvecs = [r.reshape(3, 1) for r in self.rvecs]
        self.tvecs = [t.reshape(3, 1) for t in self.tvecs]
        
        print("\nRefined Intrinsic Matrix (K):")
        print(self.K)
        print(f"fx={self.K[0,0]:.4f}, fy={self.K[1,1]:.4f}, "
              f"cx={self.K[0,2]:.4f}, cy={self.K[1,2]:.4f}")
        print(f"\nRefined distortion: k1={self.dist_coeffs[0]:.6f}, "
              f"k2={self.dist_coeffs[1]:.6f}")


    def calibrate(self):
        self.compute_intrinsics()
        self.compute_extrinsics()
        self.estimate_distortion()
        self.refine_parameters()

        dist_opencv = np.zeros((5, 1), dtype=np.float64)
        dist_opencv[0, 0] = self.dist_coeffs[0]
        dist_opencv[1, 0] = self.dist_coeffs[1]
        
        return {
            'camera_matrix': self.K,
            'dist_coeffs': dist_opencv,
            'rvecs': self.rvecs,
            'tvecs': self.tvecs
        }
    
    def compute_reprojection_error(self, result):
        errors = []
        for i in range(self.n_images):
            proj, _ = cv2.projectPoints(self.object_points[i], result['rvecs'][i], 
                                       result['tvecs'][i], result['camera_matrix'], 
                                       result['dist_coeffs'])
            error = cv2.norm(self.image_points[i], proj, cv2.NORM_L2) / len(proj)
            errors.append(error)
            print(f"Image {i:03d}: {error:.4f} pixels")
        
        print(f"\nMean: {np.mean(errors):.6f} px | Std: {np.std(errors):.6f} px")
        print(f"Min:  {np.min(errors):.6f} px | Max: {np.max(errors):.6f} px")
        return errors
    
    ## save visualization results ##
    def visualize_reprojection(self, result, output_dir="results/reprojection/zhang"):
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(len(self.images)):
            img = self.images[i].copy()
            
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i], result['rvecs'][i], result['tvecs'][i],
                result['camera_matrix'], result['dist_coeffs']
            )
            
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            
            for point in self.image_points[i]:
                cv2.circle(img, tuple(point.ravel().astype(int)), 5, (0, 0, 255), -1)
            
            for point in imgpoints2:
                cv2.circle(img, tuple(point.ravel().astype(int)), 3, (0, 255, 0), -1)

            cv2.rectangle(img, (5, 5), (550, 140), (0, 0, 0), -1)
            cv2.putText(img, "Red: Detected", (15, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(img, "Green: Reprojected", (15, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(img, f"Error: {error:.4f} px", (15, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            cv2.imwrite(os.path.join(output_dir, f"reproj_{i+1:03d}.jpg"), img)
        
        print(f"Saved {len(self.images)} reprojection images")
    
    def undistort_images(self, result, output_dir="results/undistorted/zhang"):
        os.makedirs(output_dir, exist_ok=True)
        
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            result['camera_matrix'], result['dist_coeffs'],
            self.image_size, 1, self.image_size
        )
        
        for i, img in enumerate(self.images):
            dst = cv2.undistort(img, result['camera_matrix'], 
                               result['dist_coeffs'], None, newcameramtx)
            comparison = np.hstack([img, dst])
            h, w = img.shape[:2]
            
            cv2.rectangle(comparison, (5, 5), (450, 70), (0, 0, 0), -1)
            cv2.rectangle(comparison, (w + 5, 5), (w + 350, 70), (0, 0, 0), -1)
            
            cv2.putText(comparison, "Original (Distorted)", (15, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(comparison, "Undistorted", (w + 15, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            cv2.imwrite(os.path.join(output_dir, f"undist_{i+1:03d}.jpg"), comparison)
        
        print(f"Saved {len(self.images)} undistorted images")
    
    ## save numerical results to json ##
    def save_results(self, result, errors, filename="results/zhang_calibration.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        dist_flat = result['dist_coeffs'].ravel()
        rotation_matrices = []
        for rvec in result['rvecs']:
            rmat, _ = cv2.Rodrigues(rvec)
            rotation_matrices.append(rmat.tolist())
        
        data = {
            'method': "Zhang's Method (Custom Implementation)",
            'num_images': self.n_images,
            'image_size': {
                'width': self.image_size[0], 
                'height': self.image_size[1]
            },
            'intrinsics': {
                'camera_matrix': result['camera_matrix'].tolist(),
                'fx': float(result['camera_matrix'][0, 0]),
                'fy': float(result['camera_matrix'][1, 1]),
                'cx': float(result['camera_matrix'][0, 2]),
                'cy': float(result['camera_matrix'][1, 2])
            },
            'distortion': {
                'coefficients': dist_flat.tolist(),
                'k1': float(dist_flat[0]),
                'k2': float(dist_flat[1])
            },
            'extrinsics': [
                {
                    'image_id': i + 1,
                    'rotation_vector': result['rvecs'][i].ravel().tolist(),
                    'rotation_matrix': rotation_matrices[i],
                    'translation_vector': result['tvecs'][i].ravel().tolist()
                }
                for i in range(len(result['rvecs']))
            ],
            'reprojection_errors': {
                'per_image': [float(e) for e in errors],
                'statistics': {
                    'mean': float(np.mean(errors)),
                    'std': float(np.std(errors)),
                    'min': float(np.min(errors)),
                    'max': float(np.max(errors))
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    loader = DataLoader(pattern_size=(13, 9), square_size=20.0)
    data = loader.load_data("data/raw_img")
    
    calibrator = ZhangCalibration(data)
    result = calibrator.calibrate()
    errors = calibrator.compute_reprojection_error(result)
    calibrator.visualize_reprojection(result)
    calibrator.undistort_images(result)
    calibrator.save_results(result, errors)
    
    