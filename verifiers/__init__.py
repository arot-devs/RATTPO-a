from .Aesthetic import AestheticVerifier
from .ImageReward import ImageRewardVerifier
from .DSGScore import DSGScoreVerifier
from .CompBench import CompBenchVerifier
from .VLM import VLMVerifier
from .CLIPScore import CLIPScoreVerifier
# from .prompt_adaptation import PromptAdaptationVerifier

SUPPORTED_VERIFIERS = {
    "imagereward": ImageRewardVerifier,
    "aesthetic": AestheticVerifier,
    "clipscore": CLIPScoreVerifier,
    "dsgscore": DSGScoreVerifier,
    "compbench": CompBenchVerifier,
    # "prompt_adaptation": PromptAdaptationVerifier
    'vlmscore': VLMVerifier,
}

SUPPORTED_METRICS = {k: getattr(v, "SUPPORTED_METRIC_CHOICES", None) for k, v in SUPPORTED_VERIFIERS.items()}
