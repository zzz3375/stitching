import math
from dataclasses import dataclass

import cv2 as cv
import numpy as np


# 定义空匹配对象的默认值（模拟cv.detail.MatchInfo）
@dataclass
class EmptyMatchInfo:
    confidence: float = 0.0
    matches: list = None
    inliers_mask: np.ndarray = None

    def getMatches(self):
        return [] if self.matches is None else self.matches

    def getInliers(self):
        return [] if self.inliers_mask is None else self.inliers_mask


class FeatureMatcher:
    """https://docs.opencv.org/4.x/da/d87/classcv_1_1detail_1_1FeaturesMatcher.html"""

    MATCHER_CHOICES = ("homography", "affine")
    DEFAULT_MATCHER = "homography"
    DEFAULT_RANGE_WIDTH = -1

    def __init__(
        self, matcher_type=DEFAULT_MATCHER, range_width=DEFAULT_RANGE_WIDTH,** kwargs
    ):
        if matcher_type == "affine":
            self.matcher = cv.detail_AffineBestOf2NearestMatcher(**kwargs)
        elif range_width == -1:
            self.matcher = cv.detail_BestOf2NearestMatcher(**kwargs)
        else:
            self.matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, **kwargs)

    def match_features(self, features, *args, **kwargs):
        """
        匹配特征，跳过特征数≤1的图像，并将匹配结果映射回原始图像索引
        :param features: 图像特征列表（cv.detail.ImageFeatures 类型）
        :param args: 其他参数
        :param kwargs: 其他关键字参数
        :return: 原始尺寸的配对匹配结果列表（按矩阵展平顺序）
        """
        # 1. 过滤特征数≤1的图像，记录原始索引映射
        original_indices = []  # 保存有效图像的原始索引
        valid_features = []    # 保存有效图像的特征
        invalid_indices = set()# 记录无效图像的索引（特征数≤1）
        
        for idx, feat in enumerate(features):
            kp_count = len(feat.getKeypoints()) if feat.getKeypoints() is not None else 0
            if kp_count > 1:
                valid_features.append(feat)
                original_indices.append(idx)
            else:
                invalid_indices.add(idx)

        # 2. 处理无有效图像/仅1张有效图像的情况
        num_original = len(features)
        num_valid = len(valid_features)
        if num_valid < 2:
            # 返回全空的匹配结果（原始尺寸）
            empty_match = EmptyMatchInfo()
            empty_matches = [empty_match] * (num_original * num_original)
            self.matcher.collectGarbage()
            return empty_matches

        # 3. 对有效特征执行匹配
        valid_matches = self.matcher.apply2(valid_features, *args, **kwargs)
        self.matcher.collectGarbage()

        # 4. 将有效匹配结果映射回原始索引矩阵
        # 4.1 构建有效匹配的矩阵形式（过滤后）
        valid_matrix = self.array_in_square_matrix(valid_matches)
        
        # 4.2 构建原始尺寸的匹配矩阵（初始化为空匹配）
        original_matrix = np.full((num_original, num_original), EmptyMatchInfo(), dtype=object)
        
        # 4.3 填充有效匹配到原始矩阵（按原始索引映射）
        for valid_i, orig_i in enumerate(original_indices):
            for valid_j, orig_j in enumerate(original_indices):
                original_matrix[orig_i, orig_j] = valid_matrix[valid_i, valid_j]

        # 5. 将原始矩阵展平为列表（与OpenCV返回格式一致）
        original_matches = original_matrix.flatten().tolist()

        return original_matches

    @staticmethod
    def draw_matches_matrix(
        imgs, features, matches, conf_thresh=1, inliers=False, **kwargs
    ):
        matches_matrix = FeatureMatcher.get_matches_matrix(matches)
        for idx1, idx2 in FeatureMatcher.get_all_img_combinations(len(imgs)):
            # 检查图像是否有足够特征，跳过无效图像对
            kp1_count = len(features[idx1].getKeypoints()) if features[idx1].getKeypoints() is not None else 0
            kp2_count = len(features[idx2].getKeypoints()) if features[idx2].getKeypoints() is not None else 0
            if kp1_count <= 1 or kp2_count <= 1:
                continue
                
            match = matches_matrix[idx1, idx2]
            # 跳过空匹配或低置信度匹配
            if isinstance(match, EmptyMatchInfo) or match.confidence < conf_thresh or len(match.getMatches()) == 0:
                continue
            
            if inliers:
                kwargs["matchesMask"] = match.getInliers()
            yield idx1, idx2, FeatureMatcher.draw_matches(
                imgs[idx1], features[idx1], imgs[idx2], features[idx2], match,** kwargs
            )

    @staticmethod
    def draw_matches(img1, features1, img2, features2, match1to2, **kwargs):
        kwargs.setdefault("flags", cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        keypoints1 = features1.getKeypoints()
        keypoints2 = features2.getKeypoints()
        matches = match1to2.getMatches()

        return cv.drawMatches(
            img1, keypoints1, img2, keypoints2, matches, None, **kwargs
        )

    @staticmethod
    def get_matches_matrix(pairwise_matches):
        return FeatureMatcher.array_in_square_matrix(pairwise_matches)

    @staticmethod
    def get_confidence_matrix(pairwise_matches):
        matches_matrix = FeatureMatcher.get_matches_matrix(pairwise_matches)
        match_confs = []
        for row in matches_matrix:
            row_confs = []
            for m in row:
                # 兼容空匹配对象
                row_confs.append(m.confidence if not isinstance(m, EmptyMatchInfo) else 0.0)
            match_confs.append(row_confs)
        return np.array(match_confs)

    @staticmethod
    def array_in_square_matrix(array):
        matrix_dimension = int(math.sqrt(len(array)))
        rows = []
        for i in range(0, len(array), matrix_dimension):
            rows.append(array[i : i + matrix_dimension])
        return np.array(rows, dtype=object)

    @staticmethod
    def get_all_img_combinations(number_imgs):
        ii, jj = np.triu_indices(number_imgs, k=1)
        for i, j in zip(ii, jj):
            yield i, j

    @staticmethod
    def get_match_conf(match_conf, feature_detector_type):
        if match_conf is None:
            match_conf = FeatureMatcher.get_default_match_conf(feature_detector_type)
        return match_conf

    @staticmethod
    def get_default_match_conf(feature_detector_type):
        if feature_detector_type == "orb":
            return 0.3
        return 0.65