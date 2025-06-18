# tracker/settings.py

from typing import Union, Dict, Tuple

def get_detector_path_and_im_size() -> Tuple[str, Tuple[int, int]]:
    detector_path = "external/weights/my_detector.pth"
    size = (720, 1280)
    return detector_path, size

class GeneralSettings:
    values: Dict[str, Union[float, bool, int, str]] = {
        "max_age": 30,
        "min_hits": 3,
        "det_thresh": 0.5,
        "iou_threshold": 0.3,
        "use_ecc": False,
        "min_box_area": 10,
        "aspect_ratio_thresh": 1.6,
    }

    @staticmethod
    def get(key: str) -> Union[float, bool, int, str]:
        return GeneralSettings.values.get(key, None)
    def max_age(seq_name: str) -> int:
        return 30

class BoostTrackSettings:
    values: Dict[str, Union[float, bool, int, str]] = {
        "lambda_iou": 0.5, # 0 to turn off
        "lambda_mhd": 0.25, # 0 to turn off
        "lambda_shape": 0.25, # 0 to turn off
        "use_dlo_boost": True, # False to turn off
        "use_duo_boost": True, # False to turn off
        "dlo_boost_coef": 0.6, # Irrelevant if use_dlo_boost == False
        "s_sim_corr": False, # Which shape similarity function should be used (True == corrected version)
    }

    @staticmethod
    def get(key: str) -> Union[float, bool, int, str]:
        return BoostTrackSettings.values.get(key, None)

class BoostTrackPlusPlusSettings:
    values: Dict[str, bool] = {
        "use_rich_s": True,
        "use_sb": True,
        "use_vt": True,
    }

    @staticmethod
    def get(key: str) -> bool:
        return BoostTrackPlusPlusSettings.values.get(key, False)