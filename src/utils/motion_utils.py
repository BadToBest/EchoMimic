from skimage.transform import estimate_transform, AffineTransform
import numpy as np
from IPython import embed
import mediapipe as mp
import copy

mp_face_mesh = mp.solutions.face_mesh

  
FACEMESH_LEFT_EYE = [i for i in mp_face_mesh.FACEMESH_LEFT_EYE] 
FACEMESH_RIGHT_EYE = [i for i in mp_face_mesh.FACEMESH_RIGHT_EYE]
FACEMESH_LEFT_EYEBROW = [i for i in mp_face_mesh.FACEMESH_LEFT_EYEBROW]
FACEMESH_RIGHT_EYEBROW = [i for i in mp_face_mesh.FACEMESH_RIGHT_EYEBROW]
             
# copy from draw_utils
FACEMESH_LIPS_OUTER_BOTTOM_LEFT = [(61,146),(146,91),(91,181),(181,84),(84,17)]
FACEMESH_LIPS_OUTER_BOTTOM_RIGHT = [(17,314),(314,405),(405,321),(321,375),(375,291)]
FACEMESH_LIPS_INNER_BOTTOM_LEFT = [(78,95),(95,88),(88,178),(178,87),(87,14)]
FACEMESH_LIPS_INNER_BOTTOM_RIGHT = [(14,317),(317,402),(402,318),(318,324),(324,308)]
FACEMESH_LIPS_OUTER_TOP_LEFT = [(61,185),(185,40),(40,39),(39,37),(37,0)]
FACEMESH_LIPS_OUTER_TOP_RIGHT = [(0,267),(267,269),(269,270),(270,409),(409,291)]
FACEMESH_LIPS_INNER_TOP_LEFT = [(78,191),(191,80),(80,81),(81,82),(82,13)]
FACEMESH_LIPS_INNER_TOP_RIGHT = [(13,312),(312,311),(311,310),(310,415),(415,308)]
FACEMESH_MOUSE = \
    FACEMESH_LIPS_OUTER_BOTTOM_LEFT + \
    FACEMESH_LIPS_OUTER_BOTTOM_RIGHT + \
    FACEMESH_LIPS_INNER_BOTTOM_LEFT + \
    FACEMESH_LIPS_INNER_BOTTOM_RIGHT + \
    FACEMESH_LIPS_OUTER_TOP_LEFT + \
    FACEMESH_LIPS_OUTER_TOP_RIGHT + \
    FACEMESH_LIPS_INNER_TOP_LEFT + \
    FACEMESH_LIPS_INNER_TOP_RIGHT

LANDMARK_IDXES_DICT = {
    "left_eye" : sorted(list(set([j for i in FACEMESH_LEFT_EYE for j in i])) + [473]),
    "right_eye" : sorted(list(set([j for i in FACEMESH_RIGHT_EYE for j in i])) + [468]),
    "mouse" : sorted(list(set([j for i in FACEMESH_MOUSE for j in i]))),
    "nose" : sorted(list(set([1,4,5,274,275,281,44,45,51,220,440]))),
    "left_eyebow" : sorted(list(set([j for i in FACEMESH_LEFT_EYEBROW for j in i]))),
    "right_eyebow" : sorted(list(set([j for i in FACEMESH_RIGHT_EYEBROW for j in i]))),
}

def create_perspective_matrix(aspect_ratio):
    kDegreesToRadians = np.pi / 180.
    near = 1
    far = 10000
    perspective_matrix = np.zeros(16, dtype=np.float32)

    # Standard perspective projection matrix calculations.
    f = 1.0 / np.tan(kDegreesToRadians * 63 / 2.)

    denom = 1.0 / (near - far)
    perspective_matrix[0] = f / aspect_ratio
    perspective_matrix[5] = f
    perspective_matrix[10] = (near + far) * denom
    perspective_matrix[11] = -1.
    perspective_matrix[14] = 1. * far * near * denom

    # If the environment's origin point location is in the top left corner,
    # then skip additional flip along Y-axis is required to render correctly.

    perspective_matrix[5] *= -1.
    return perspective_matrix


def project_points_with_trans(points_3d, transformation_matrix, image_shape):
    P = create_perspective_matrix(image_shape[1] / image_shape[0]).reshape(4, 4).T
    L, N, _ = points_3d.shape
    projected_points = np.zeros((L, N, 2))
    #embed()
    for i in range(L):
        points_3d_frame = points_3d[i]
        ones = np.ones((points_3d_frame.shape[0], 1))
        points_3d_homogeneous = np.hstack([points_3d_frame, ones])  
        transformed_points = points_3d_homogeneous @ transformation_matrix[i].T @ P
        projected_points_frame = transformed_points[:, :2] / transformed_points[:, 3, np.newaxis] # -1 ~ 1
        projected_points_frame[:, 0] = (projected_points_frame[:, 0] + 1) * 0.5 * image_shape[1] 
        projected_points_frame[:, 1]  = (projected_points_frame[:, 1] + 1) * 0.5 * image_shape[0]
        projected_points[i] = projected_points_frame
    return projected_points

def project_vertices_from_ref2tgt(ref_lmks3d, tgt_trans_mat):
    #eye_point_idxes
    projected_vertices = project_points_with_trans(ref_lmks3d[np.newaxis, ...], tgt_trans_mat[np.newaxis, ...], [512, 512])[0]
    return projected_vertices


def old_motion_sync_old(sequence_driver_det, reference_det):
    assert type(sequence_driver_det) is list
    assert type(sequence_driver_det[0]) is type(reference_det) 

    lmks3d_mean = sum([i["lmks3d"] for i in sequence_driver_det]) / len(sequence_driver_det)
    overall_transform = estimate_transform('affine', lmks3d_mean, reference_det["lmks3d"])

    eye_idxes_all = LANDMARK_IDXES_DICT["left_eye"] + LANDMARK_IDXES_DICT["right_eye"]
    for det_id in range(len(sequence_driver_det)):
        trans = estimate_transform('affine', sequence_driver_det[det_id]["lmks"][eye_idxes_all], sequence_driver_det[det_id]["lmks3d"][eye_idxes_all])
        sequence_driver_det[det_id]["lmks3d"] = np.vstack([
            sequence_driver_det[det_id]["lmks3d"],
            trans(sequence_driver_det[det_id]["lmks"][-10:])
        ])

    trans_mats = [] 
    for det in sequence_driver_det: 
        trans_mats.append(det["trans_mat"] @ np.linalg.inv(sequence_driver_det[0]["trans_mat"]))

    trans_mats_smooth = []
    smooth_margin = 2
    for tm_itx in range(len(trans_mats)):
        smooth_idxes = [i for i in range(tm_itx - smooth_margin, tm_itx + smooth_margin + 1) if i >= 0 and i < len(trans_mats)]
        tm = sum([trans_mats[i] for i in smooth_idxes]) / len(smooth_idxes)
        trans_mats_smooth.append(tm)

    lmks3d_smooth = []
    smooth_margin = 1
    for det_itx in range(len(sequence_driver_det)):
        smooth_idxes = [i for i in range(det_itx - smooth_margin, det_itx + smooth_margin + 1) if i >= 0 and i < len(sequence_driver_det)]
        lmks3d_smooth.append(sum([sequence_driver_det[i]["lmks3d"] for i in smooth_idxes]) / len(smooth_idxes))

    for det_itx, lmks3d in enumerate(lmks3d_smooth):
        sequence_driver_det[det_itx]["lmks3d"] = lmks3d

    projected_vertices_list = []
    for det_itx in range(len(sequence_driver_det)):
        aligned_3d = overall_transform(sequence_driver_det[det_itx]["lmks3d"])
        tmat = reference_det["trans_mat"] @ trans_mats_smooth[det_itx]
        projected_vertices = project_vertices_from_ref2tgt(aligned_3d, tmat)
        projected_vertices_list.append(projected_vertices)
    
    # note : use normed=False after motion_sync, when draw(ing)_landmarks
    # kps_image = vis.draw_landmarks((512, 512), projected_vertices, normed=False) 
    return  projected_vertices_list


def motion_sync(sequence_driver_det, reference_det, per_landmark_align=True):
    assert type(sequence_driver_det) is list
    assert type(sequence_driver_det[0]) is type(reference_det) 

    eye_idxes_all = [i for i in sorted(list(set(LANDMARK_IDXES_DICT["left_eye"] + LANDMARK_IDXES_DICT["right_eye"]))) if i < len(reference_det["lmks3d"])]
    for det_id in range(len(sequence_driver_det)):
        trans_iris = estimate_transform('affine', sequence_driver_det[det_id]["lmks"][eye_idxes_all], sequence_driver_det[det_id]["lmks3d"][eye_idxes_all])
        sequence_driver_det[det_id]["lmks3d"] = np.vstack([
            sequence_driver_det[det_id]["lmks3d"],
            trans_iris(sequence_driver_det[det_id]["lmks"][-10:])
        ])

    trans_iris = estimate_transform('affine', reference_det["lmks"][eye_idxes_all], reference_det["lmks3d"][eye_idxes_all])
    reference_det["lmks3d"] = np.vstack([
        reference_det["lmks3d"],
        trans_iris(reference_det["lmks"][-10:])
    ])

    lmks3d_mean = sum([i["lmks3d"] for i in sequence_driver_det]) / len(sequence_driver_det)

    landmark_trans_dict = {}
    for landmark_name, landmark_idxes in LANDMARK_IDXES_DICT.items():
        rf_lm = reference_det["lmks3d"][landmark_idxes]
        dr_lm = lmks3d_mean[landmark_idxes]
        landmark_trans_dict[landmark_name] = estimate_transform('affine', dr_lm, rf_lm)

    #embed()
    overall_transform = estimate_transform('affine', lmks3d_mean, reference_det["lmks3d"])
    #embed()
    #lmks3d_mean = sum([i["lmks3d"] for i in sequence_driver_det]) / len(sequence_driver_det)
    #overall_transform = estimate_transform('affine', lmks3d_mean, reference_det["lmks3d"])
    
    #driver_start_center = sequence_driver_det[0]["lmks3d"].mean(axis=0)
    #reference_center = reference_det["lmks3d"].mean(axis=0)
    #driver_start_size = ((sequence_driver_det[0]["lmks3d"] - driver_start_center)**2).sum()**(0.5)
    #reference_size = ((reference_det["lmks3d"] - reference_center)**2).sum()**(0.5)

    #reference_det_lmks3d_rescale = (reference_det["lmks3d"] - reference_center) / reference_size * driver_start_size + driver_start_center
    #reference_transform_back = estimate_transform('affine', reference_det_lmks3d_rescale, reference_det["lmks3d"])

    #driver_lmks3d_mean = sum([i["lmks3d"] for i in sequence_driver_det]) / len(sequence_driver_det)
    #facial_transform = estimate_transform('affine', driver_lmks3d_mean, reference_det_lmks3d_rescale)


    #for det_id in range(len(sequence_driver_det)):
    #    trans = estimate_transform('affine', sequence_driver_det[det_id]["lmks"][:-10], sequence_driver_det[det_id]["lmks3d"])
    #    sequence_driver_det[det_id]["lmks3d"] = trans(sequence_driver_det[det_id]["lmks"])

    trans_mats = [] 
    for det in sequence_driver_det: 
        trans_mats.append(det["trans_mat"] @ np.linalg.inv(sequence_driver_det[0]["trans_mat"]))

    trans_mats_smooth = []
    smooth_margin = 2
    for tm_itx in range(len(trans_mats)):
        smooth_idxes = [i for i in range(tm_itx - smooth_margin, tm_itx + smooth_margin + 1) if i >= 0 and i < len(trans_mats)]
        tm = sum([trans_mats[i] for i in smooth_idxes]) / len(smooth_idxes)
        trans_mats_smooth.append(tm)

    lmks3d_smooth = []
    smooth_margin = 1
    for det_itx in range(len(sequence_driver_det)):
        smooth_idxes = [i for i in range(det_itx - smooth_margin, det_itx + smooth_margin + 1) if i >= 0 and i < len(sequence_driver_det)]
        lmks3d_smooth.append(sum([sequence_driver_det[i]["lmks3d"] for i in smooth_idxes]) / len(smooth_idxes))

    for det_itx, lmks3d in enumerate(lmks3d_smooth):
        sequence_driver_det[det_itx]["lmks3d"] = lmks3d

    projected_vertices_list = []
    for det_itx in range(len(sequence_driver_det)):
        #aligned_3d = overall_transform(sequence_driver_det[det_itx]["lmks3d"])
        aligned_3d = copy.deepcopy(sequence_driver_det[det_itx]["lmks3d"])
        if per_landmark_align:
            for landmark_name, landmark_idxes in LANDMARK_IDXES_DICT.items():
                dr_lm = sequence_driver_det[det_itx]["lmks3d"][landmark_idxes]
                lm_trans = landmark_trans_dict[landmark_name]
                aligned_3d[landmark_idxes] = lm_trans(dr_lm)

        #aligned_3d = lmks3d_mean
        tmat = trans_mats_smooth[det_itx] @ reference_det["trans_mat"]
        projected_vertices = project_vertices_from_ref2tgt(aligned_3d, tmat)
        projected_vertices_list.append(projected_vertices)

        continue

        trans_ref_aligned_to_driver = (sequence_driver_det[det_itx]["trans_mat"]) @ np.linalg.inv(reference_det["trans_mat"])
        ref_aligned_to_driver = AffineTransform(trans_ref_aligned_to_driver)(reference_det["lmks3d"])
        det["trans_mat"] @ np.linalg.inv(sequence_driver_det[0]["trans_mat"])
        aligned_3d = sequence_driver_det[det_itx]["lmks3d"]

        
        #facial_transform(sequence_driver_det[det_itx]["lmks3d"])
        #tmat = reference_det["trans_mat"] @ trans_mats_smooth[det_itx]

        tmat = sequence_driver_det[det_itx]["trans_mat"] @ trans_mats_smooth[det_itx] #@ reference_transform_back.params
        projected_vertices = project_vertices_from_ref2tgt(aligned_3d, tmat)
        #embed()
        #projected_vertices = reference_transform_back(projected_vertices)
        projected_vertices_list.append(projected_vertices)
    
    # note : use normed=False after motion_sync, when draw(ing)_landmarks
    # kps_image = vis.draw_landmarks((512, 512), projected_vertices, normed=False) 
    return  projected_vertices_list
    
     