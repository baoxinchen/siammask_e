# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.bbox import cxy_wh_2_rect
from pysot.tracker.siamrpn_tracker import SiamRPNTracker

class SiamMaskETracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamMaskETracker, self).__init__(model)
       
        assert hasattr(self.model, 'mask_head'), \
                "SiamMaskETracker must have mask_head"
        assert hasattr(self.model, 'refine_head'), \
                "SiamMaskETracker must have refine_head"
        
        
    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THERSHOLD)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, 
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, 
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)] 
            polygon = contour.reshape(-1, 2)

            ## the following code estimate the shape angle with ellipse
            ## then fit a axis-aligned bounding box on the rotated image
            
            ellipseBox = cv2.fitEllipse(polygon)
            # get the center of the ellipse and the angle
            angle = ellipseBox[-1]
            #print(angle)
            center = np.array(ellipseBox[0])
            axes = np.array(ellipseBox[1])
            
            # get the ellipse box
            ellipseBox = cv2.boxPoints(ellipseBox)
            
            #compute the rotation matrix
            rot_mat = cv2.getRotationMatrix2D((center[0],center[1]), angle, 1.0)
            
            # rotate the ellipse box
            one = np.ones([ellipseBox.shape[0],3,1])
            one[:,:2,:] = ellipseBox.reshape(-1,2,1)
            ellipseBox = np.matmul(rot_mat, one).reshape(-1,2)
            
            # to xmin ymin xmax ymax
            xs = ellipseBox[:,0]
            xmin, xmax = np.min(xs), np.max(xs)
            ys = ellipseBox[:,1]
            ymin, ymax = np.min(ys), np.max(ys)
            ellipseBox = [xmin, ymin, xmax, ymax]
            
            # rotate the contour
            one = np.ones([polygon.shape[0],3,1])
            one[:,:2,:] = polygon.reshape(-1,2,1)
            polygon = np.matmul(rot_mat, one).astype(int).reshape(-1,2)
            
            # remove points outside of the ellipseBox
            logi = polygon[:,0]<=xmax
            logi = np.logical_and(polygon[:,0]>=xmin, logi)
            logi = np.logical_and(polygon[:,1]>=ymin, logi)
            logi = np.logical_and(polygon[:,1]<=ymax, logi)
            polygon = polygon[logi,:]
            
            x,y,w,h = cv2.boundingRect(polygon)
            bRect = [x, y, x+w, y+h]
            
            # get the intersection of ellipse box and the rotated box
            x1, y1, x2, y2 = ellipseBox[0], ellipseBox[1], ellipseBox[2], ellipseBox[3]
            tx1, ty1, tx2, ty2 = bRect[0], bRect[1], bRect[2], bRect[3]
            xx1 = min(max(tx1, x1, 0), target_mask.shape[1]-1)
            yy1 = min(max(ty1, y1, 0), target_mask.shape[0]-1)
            xx2 = max(min(tx2, x2, target_mask.shape[1]-1), 0)
            yy2 = max(min(ty2, y2, target_mask.shape[0]-1), 0)
            
            rotated_mask = cv2.warpAffine(target_mask, rot_mat,(target_mask.shape[1],target_mask.shape[0]))
            
            #refinement
            alpha_factor = cfg.TRACK.FACTOR
            while True:
                if np.sum(rotated_mask[int(yy1):int(yy2),int(xx1)]) < (yy2-yy1)*alpha_factor:
                    temp = xx1+(xx2-xx1)*0.02
                    if not (temp >= target_mask.shape[1]-1 or xx2-xx1 < 1):
                        xx1 = temp
                    else:
                        break
                else:
                    break
            while True:
                if np.sum(rotated_mask[int(yy1):int(yy2),int(xx2)]) < (yy2-yy1)*alpha_factor:
                    temp = xx2-(xx2-xx1)*0.02
                    if not (temp <= 0 or xx2-xx1 < 1):
                        xx2 = temp
                    else:
                        break
                else:
                    break
            while True:
                if np.sum(rotated_mask[int(yy1),int(xx1):int(xx2)]) < (xx2-xx1)*alpha_factor:
                    temp = yy1+(yy2-yy1)*0.02
                    if not (temp >= target_mask.shape[0]-1 or yy2-yy1 < 1):
                        yy1 = temp
                    else:
                        break
                else:
                    break
            while True:
                if np.sum(rotated_mask[int(yy2),int(xx1):int(xx2)]) < (xx2-xx1)*alpha_factor:
                    temp = yy2-(yy2-yy1)*0.02
                    if not (temp <= 0 or yy2-yy1 < 1):
                        yy2 = temp
                    else:
                        break
                else:
                    break
            
            prbox = np.array([[xx1,yy1],[xx2,yy1],[xx2,yy2],[xx1,yy2]])
            
            # inverse of the rotation matrix
            M_inv = cv2.invertAffineTransform(rot_mat)
            # project the points back to image coordinate
            one = np.ones([prbox.shape[0],3,1])
            one[:,:2,:] = prbox.reshape(-1,2,1)
            prbox = np.matmul(M_inv, one).reshape(-1,2)
            
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                        [location[0] + location[2], location[1]],
                        [location[0] + location[2], location[1] + location[3]],
                        [location[0], location[1] + location[3]]])
        return rbox_in_img

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        s_x = round(s_x)
        
        x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE,
                s_x, self.channel_average)
        crop_box = [self.center_pos[0] - s_x / 2, 
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]
       
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
            
        # scale penalty
        s_c = change(sz(pred_bbox[2,:], pred_bbox[3,:]) / 
                (sz(self.size[0]*scale_z, self.size[1]*scale_z))) 
        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2,:]/pred_bbox[3,:]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty 
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
   
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        # processing mask
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        mask = self.model.mask_refine((delta_y, delta_x)).sigmoid().squeeze()
        out_size = cfg.TRACK.MASK_OUTPUT_SIZE
        mask = mask.view(out_size, out_size).cpu().data.numpy()
        
        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        base_size = cfg.TRACK.BASE_SIZE
        stride = cfg.ANCHOR.STRIDE 
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * cfg.TRACK.EXEMPLAR_SIZE,
                   s * cfg.TRACK.EXEMPLAR_SIZE]
        s = out_size / sub_box[2]

        im_h, im_w = img.shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()
        
        return {
                'bbox': bbox,
                'best_score': best_score,
                'mask': mask_in_img,
                'polygon': polygon
               }
