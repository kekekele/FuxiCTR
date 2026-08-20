from importlib import import_module

_MODEL_IMPORTS = {
    "AFM": ".AFM.src",
    "AFN": ".AFN.src",
    "AOANet": ".AOANet.src",
    "AutoInt": ".AutoInt.src",
    "BST": ".BST.src",
    "CCPM": ".CCPM.src",
    "DCN": ".DCN.DCN_torch.src",
    "DCNv2": ".DCNv2.src",
    "DeepCrossing": ".DeepCrossing.src",
    "DeepFM": ".DeepFM.DeepFM_torch.src",
    "DeepIM": ".DeepIM.src",
    "DESTINE": ".DESTINE.src",
    "DIEN": ".DIEN.src",
    "DIN": ".DIN.src",
    "DLRM": ".DLRM.src",
    "DMIN": ".DMIN.src",
    "DMR": ".DMR.src",
    "DNN": ".DNN.DNN_torch.src",
    "DSSM": ".DSSM.src",
    "EDCN": ".EDCN.src",
    "FFM": ".FFM.src",
    "FFMv2": ".FFM.src",
    "FGCNN": ".FGCNN.src",
    "FiBiNET": ".FiBiNET.src",
    "FiGNN": ".FiGNN.src",
    "FinalMLP": ".FinalMLP.src",
    "FinalNet": ".FinalNet.src",
    "FLEN": ".FLEN.src",
    "FM": ".FM.src",
    "FmFM": ".FmFM.src",
    "FwFM": ".FwFM.src",
    "HFM": ".HFM.src",
    "HOFM": ".HOFM.src",
    "InterHAt": ".InterHAt.src",
    "LorentzFM": ".LorentzFM.src",
    "LR": ".LR.src",
    "MaskNet": ".MaskNet.src",
    "NFM": ".NFM.src",
    "ONN": ".ONN.ONN_torch.src",
    "ONNv2": ".ONN.ONN_torch.src",
    "PNN": ".PNN.src",
    "SAM": ".SAM.src",
    "WideDeep": ".WideDeep.WideDeep_torch.src",
    "xDeepFM": ".xDeepFM.src",
    "PPNet": ".PEPNet.src",
    "TransAct": ".TransAct.src",
    "ShareBottom": ".multitask",
    "MMoE": ".multitask",
    "PLE": ".multitask",
    "EulerNet": ".EulerNet.src",
    "WuKong": ".WuKong.src",
    "GDCN": ".GDCN.src",
}

__all__ = list(_MODEL_IMPORTS)


def __getattr__(name):
    if name not in _MODEL_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODEL_IMPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
