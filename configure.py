# configure.py – central configuration for the **BirdCLEF 2025** pipeline
# =============================================================================
# One authoritative source of truth for every tunable used by the codebase.
# Edit paths & hyper‑parameters here, re‑run the affected stage, and *never*
# hunt for magic numbers scattered across modules again.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

class CFG:  # pylint: disable=too-few-public-methods
    """Global, immutable configuration namespace."""
    # ────────────────────────────────────────────────────────────────────────
    # Reproducibility & runtime
    # ────────────────────────────────────────────────────────────────────────
    SEED: int = 42              # master RNG seed
    DEVICE: str = "cuda"         # "cuda" | "gpu" | "cpu"
    # ────────────────────────────────────────────────────────────────────────
    # Filesystem layout  – adjust to your environment / Kaggle dataset mount
    # ────────────────────────────────────────────────────────────────────────
    DATA_ROOT: Path = Path("/data/birdclef")

    TRAIN_AUDIO_DIR: Path = DATA_ROOT / "train_audio"
    TRAIN_SOUNDSCAPE_DIR: Path = DATA_ROOT / "train_soundscapes"
    TRAIN_CSV: Path = DATA_ROOT / "train.csv"
    TAXONOMY_CSV: Path = DATA_ROOT / "taxonomy.csv"
    TEST_DIR: Path = DATA_ROOT / "test_soundscapes/"
    SAMPLE_SUBMISSION: Path = DATA_ROOT / "sample_submission.csv"

    # Pre‑processing outputs
    PROCESSED_DIR: Path = DATA_ROOT / "processed"  # mels with labels for training data

    # Model checkpoints
    MODELS_DIR: Path = DATA_ROOT / "models"

    # Inference artifact
    SUBMISSION_OUT: Path = DATA_ROOT / "submission.csv"

    # ────────────────────────────────────────────────────────────────────────
    # Audio / spectrogram params – **keep consistent across modules**
    # ────────────────────────────────────────────────────────────────────────
    SAMPLE_RATE: int = 32_000
    N_FFT: int = 1024
    HOP_LENGTH: int = 500
    N_MELS: int = 128
    FMIN: int = 40
    FMAX: int = 15_000
    POWER: float = 2.0

    TARGET_SHAPE: tuple[int, int] = (256, 256)  # resize for CNN

    # ────────────────────────────────────────────────────────────────────────
    # Chunking strategy
    # ────────────────────────────────────────────────────────────────────────
    TRAIN_CHUNK_SEC: int = 10
    TRAIN_CHUNK_HOP_SEC: int = 5
    SC_SEG_SEC: int = 5  # evaluation granularity
    FOLD0_RATIO: float = 0.80  # % of cleanest clips used for training

    # ────────────────────────────────────────────────────────────────────────
    # Pre‑processing heuristics
    # ────────────────────────────────────────────────────────────────────────
    TRIM_TOP_DB: int = 20
    RMS_THRESHOLD: float = 0.01
    MIN_RATING: int = 1
    PSEUDO_THRESHOLD: float = 0.50

    USE_SOFT_LABELS: bool = True
    LABEL_WEIGHT_PRIMARY: float = 0.7
    LABEL_WEIGHT_SECONDARY: float = 0.2
    LABEL_WEIGHT_BENCH: float = 0.1

    RARE_COUNT_THRESHOLD: int = 20
    RARE_WEIGHT: float = 2.0
    PSEUDO_WEIGHT: float = 0.5

    MEL_CACHE_SIZE: int = 2048

    # ────────────────────────────────────────────────────────────────────────
    # Data‑augmentation
    # ────────────────────────────────────────────────────────────────────────
    SPEC_AUG_FREQ_MASK_PARAM: int = 10
    SPEC_AUG_TIME_MASK_PARAM: int = 50
    SPEC_AUG_NUM_MASKS: int = 2
    CUTMIX_PROB: float = 0.5

    # ────────────────────────────────────────────────────────────────────────
    # EfficientNet‑B0
    # ────────────────────────────────────────────────────────────────────────
    EFF_NUM_MODELS: int = 2
    EFF_BATCH_SIZE: int = 32
    EFF_EPOCHS: int = 10
    EFF_LR: float = 2e-3
    EFF_WEIGHT_DECAY: float = 1e-4
    EFF_NUM_WORKERS: int = 4

    # ────────────────────────────────────────────────────────────────────────
    # RegNetY‑0.8GF
    # ────────────────────────────────────────────────────────────────────────
    REG_NUM_MODELS: int = 2
    REG_BATCH_SIZE: int = 32
    REG_EPOCHS: int = 10
    REG_LR: float = 2e-3
    REG_WEIGHT_DECAY: float = 1e-4
    REG_NUM_WORKERS: int = 4

    # ────────────────────────────────────────────────────────────────────────
    # DiffWave minority‑class synthesis
    # ────────────────────────────────────────────────────────────────────────
    DIFF_RARE_THRESHOLD: int = 10  # if real recordings < 10 ⇒ target for synth

    # ────────────────────────────────────────────────────────────────────────
    # Inference / ensemble parameters
    # ────────────────────────────────────────────────────────────────────────
    INF_BATCH_SIZE: int = 16
    INF_NUM_WORKERS: int = 4
    INF_SMOOTH_NEIGHBORS: int = 2  # ±2 × 5 s segments

    # ────────────────────────────────────────────────────────────────────────
    # CPU optimisation (OpenVINO / ONNX)
    # ────────────────────────────────────────────────────────────────────────
    USE_OPENVINO: bool = True
    OV_NUM_THREADS: int | None = None

    # ────────────────────────────────────────────────────────────────────────
    # Class mapping
    # ────────────────────────────────────────────────────────────────────────
    SPECIES: Dict[int, str] = {
        0: '1139490',
        1: '1192948',
        2: '1194042',
        3: '126247',
        4: '1346504',
        5: '134933',
        6: '135045',
        7: '1462711',
        8: '1462737',
        9: '1564122',
        10: '21038',
        11: '21116',
        12: '21211',
        13: '22333',
        14: '22973',
        15: '22976',
        16: '24272',
        17: '24292',
        18: '24322',
        19: '41663',
        20: '41778',
        21: '41970',
        22: '42007',
        23: '42087',
        24: '42113',
        25: '46010',
        26: '47067',
        27: '476537',
        28: '476538',
        29: '48124',
        30: '50186',
        31: '517119',
        32: '523060',
        33: '528041',
        34: '52884',
        35: '548639',
        36: '555086',
        37: '555142',
        38: '566513',
        39: '64862',
        40: '65336',
        41: '65344',
        42: '65349',
        43: '65373',
        44: '65419',
        45: '65448',
        46: '65547',
        47: '65962',
        48: '66016',
        49: '66531',
        50: '66578',
        51: '66893',
        52: '67082',
        53: '67252',
        54: '714022',
        55: '715170',
        56: '787625',
        57: '81930',
        58: '868458',
        59: '963335',
        60: 'amakin1',
        61: 'amekes',
        62: 'ampkin1',
        63: 'anhing',
        64: 'babwar',
        65: 'bafibi1',
        66: 'banana',
        67: 'baymac',
        68: 'bbwduc',
        69: 'bicwre1',
        70: 'bkcdon',
        71: 'bkmtou1',
        72: 'blbgra1',
        73: 'blbwre1',
        74: 'blcant4',
        75: 'blchaw1',
        76: 'blcjay1',
        77: 'blctit1',
        78: 'blhpar1',
        79: 'blkvul',
        80: 'bobfly1',
        81: 'bobher1',
        82: 'brtpar1',
        83: 'bubcur1',
        84: 'bubwre1',
        85: 'bucmot3',
        86: 'bugtan',
        87: 'butsal1',
        88: 'cargra1',
        89: 'cattyr',
        90: 'chbant1',
        91: 'chfmac1',
        92: 'cinbec1',
        93: 'cocher1',
        94: 'cocwoo1',
        95: 'colara1',
        96: 'colcha1',
        97: 'compau',
        98: 'compot1',
        99: 'cotfly1',
        100: 'crbtan1',
        101: 'crcwoo1',
        102: 'crebob1',
        103: 'cregua1',
        104: 'creoro1',
        105: 'eardov1',
        106: 'fotfly',
        107: 'gohman1',
        108: 'grasal4',
        109: 'grbhaw1',
        110: 'greani1',
        111: 'greegr',
        112: 'greibi1',
        113: 'grekis',
        114: 'grepot1',
        115: 'gretin1',
        116: 'grnkin',
        117: 'grysee1',
        118: 'gybmar',
        119: 'gycwor1',
        120: 'labter1',
        121: 'laufal1',
        122: 'leagre',
        123: 'linwoo1',
        124: 'littin1',
        125: 'mastit1',
        126: 'neocor',
        127: 'norscr1',
        128: 'olipic1',
        129: 'orcpar',
        130: 'palhor2',
        131: 'paltan1',
        132: 'pavpig2',
        133: 'piepuf1',
        134: 'pirfly1',
        135: 'piwtyr1',
        136: 'plbwoo1',
        137: 'plctan1',
        138: 'plukit1',
        139: 'purgal2',
        140: 'ragmac1',
        141: 'rebbla1',
        142: 'recwoo1',
        143: 'rinkin1',
        144: 'roahaw',
        145: 'rosspo1',
        146: 'royfly1',
        147: 'rtlhum',
        148: 'rubsee1',
        149: 'rufmot1',
        150: 'rugdov',
        151: 'rumfly1',
        152: 'ruther1',
        153: 'rutjac1',
        154: 'rutpuf1',
        155: 'saffin',
        156: 'sahpar1',
        157: 'savhaw1',
        158: 'secfly1',
        159: 'shghum1',
        160: 'shtfly1',
        161: 'smbani',
        162: 'snoegr',
        163: 'sobtyr1',
        164: 'socfly1',
        165: 'solsan',
        166: 'soulap1',
        167: 'spbwoo1',
        168: 'speowl1',
        169: 'spepar1',
        170: 'srwswa1',
        171: 'stbwoo2',
        172: 'strcuc1',
        173: 'strfly1',
        174: 'strher',
        175: 'strowl1',
        176: 'tbsfin1',
        177: 'thbeup1',
        178: 'thlsch3',
        179: 'trokin',
        180: 'tropar',
        181: 'trsowl',
        182: 'turvul',
        183: 'verfly',
        184: 'watjac1',
        185: 'wbwwre1',
        186: 'whbant1',
        187: 'whbman1',
        188: 'whfant1',
        189: 'whmtyr1',
        190: 'whtdov',
        191: 'whttro1',
        192: 'whwswa1',
        193: 'woosto',
        194: 'y00678',
        195: 'yebela1',
        196: 'yebfly1',
        197: 'yebsee1',
        198: 'yecspi2',
        199: 'yectyr1',
        200: 'yehbla2',
        201: 'yehcar1',
        202: 'yelori1',
        203: 'yeofly1',
        204: 'yercac1',
        205: 'ywcpar'}

# EOF