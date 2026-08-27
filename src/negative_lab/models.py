from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json


@dataclass
class AnchorSample:
    rgb: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    spread: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    source_path: str = ""
    x: float = 0.5
    y: float = 0.5
    radius: int = 9
    valid: bool = False


@dataclass
class FilmProfile:
    name: str = "Custom two-point"
    manufacturer: str = ""
    stock: str = ""
    slope_rgb: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    orange_mask_compensation: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    description: str = "Session anchors determine density range."
    toe: float = 0.0
    shoulder: float = 0.0
    grain: float = 0.0

    @classmethod
    def builtins(cls):
        return [
            cls(),
            cls("Neutral color negative", "Generic", "Color negative", [1.0, .96, .90], [1.0, 1.0, 1.0], "Gentle neutral starting profile."),
            cls("Warm portrait negative", "Generic", "Portrait", [.94, 1.0, .92], [1.0, 1.0, 1.0], "Restrained warm portrait response."),
            cls("High-contrast negative", "Generic", "High contrast", [1.12, 1.08, 1.02], [1.0, 1.0, 1.0], "Stronger density separation; verify clipping."),
        ]


@dataclass
class NegativeRecipe:
    exposure: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    shadows: float = 0.0
    highlights: float = 0.0
    gamma: float = 1.0
    crop: list[float] | None = None
    rotation: int = 0
    fine_rotation: float = 0.0
    perspective_corners: list[list[float]] | None = None
    curves: dict = field(default_factory=lambda: {"RGB": [[0,0],[1,1]]})
    show_clipping: bool = False
    enabled: bool = True
    fade_correction: float = 0.0
    dust_removal: float = 0.0
    dust_max_radius: int = 10
    sprocket_mask: bool = False
    local_adjustments: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    film_toe: float = 0.0
    film_shoulder: float = 0.0
    film_grain: float = 0.0

    @classmethod
    def from_dict(cls, data):
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in allowed})


@dataclass
class FrameRecord:
    path: str
    recipe: NegativeRecipe = field(default_factory=NegativeRecipe)
    frame_number: int = 0
    warnings: list[str] = field(default_factory=list)
    rating: int = 0
    color_label: str = "None"
    notes: str = ""
    rejected: bool = False

    @classmethod
    def from_dict(cls, data):
        return cls(str(data.get("path", "")), NegativeRecipe.from_dict(data.get("recipe")),
                   int(data.get("frame_number", 0)), list(data.get("warnings", [])),
                   int(data.get("rating",0)),str(data.get("color_label","None")),
                   str(data.get("notes","")),bool(data.get("rejected",False)))


@dataclass
class CalibrationOptions:
    dark_frame_path: str = ""
    flat_field_path: str = ""
    input_matrix: list[list[float]] = field(default_factory=lambda: [[1,0,0],[0,1,0],[0,0,1]])
    input_profile_name: str = "Identity / uncharacterized"
    input_profile_path: str = ""
    working_space: str = "Scene-linear RGB"
    output_space: str = "sRGB"
    output_icc_path: str = ""
    embed_icc: bool = True
    dpi: int = 300


@dataclass
class PerformanceOptions:
    preview_dimension: int = 1800
    cpu_threads: int = 0
    use_opencl: bool = True
    disk_preview_cache: bool = True
    cache_directory: str = ""
    recover_failed_frames: bool = True
    memory_warning_percent: int = 70


@dataclass
class OutputOptions:
    preset: str = "Archival"
    format: str = "TIFF"
    bit_depth: int = 16
    resize_percent: int = 100
    overwrite_existing: bool = False
    filename_suffix: str = "_positive"


@dataclass
class RollProject:
    name: str = "Untitled Roll"
    frames: list[FrameRecord] = field(default_factory=list)
    clear_base: AnchorSample = field(default_factory=AnchorSample)
    dense_leader: AnchorSample = field(default_factory=lambda: AnchorSample(rgb=[.05, .05, .05]))
    film_profile: FilmProfile = field(default_factory=FilmProfile)
    mode: str = "two_point"
    output_directory: str = ""
    project_version: int = 1
    calibration: CalibrationOptions = field(default_factory=CalibrationOptions)
    performance: PerformanceOptions = field(default_factory=PerformanceOptions)
    output: OutputOptions = field(default_factory=OutputOptions)

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=str(data.get("name", "Untitled Roll")),
            frames=[FrameRecord.from_dict(x) for x in data.get("frames", [])],
            clear_base=AnchorSample(**data.get("clear_base", {})),
            dense_leader=AnchorSample(**data.get("dense_leader", {"rgb":[.05,.05,.05]})),
            film_profile=FilmProfile(**data.get("film_profile", {})),
            mode=str(data.get("mode", "two_point")),
            output_directory=str(data.get("output_directory", "")),
            project_version=int(data.get("project_version", 1)),
            calibration=CalibrationOptions(**data.get("calibration", {})),
            performance=PerformanceOptions(**data.get("performance", {})),
            output=OutputOptions(**data.get("output", {})),
        )

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
