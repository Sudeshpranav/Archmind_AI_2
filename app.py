import io
import math
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import trimesh
from PIL import Image
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")


# ============================================================
# ARCHMIND PRO V2.5.1
# AI-POWERED CONSTRUCTION MATERIAL INTELLIGENCE
# ============================================================

APP_VERSION = "V2.8.0"

HF_REPO_ID = "AloneMrY/archmind-pro-v2-2-models"

TARGET_FILES = {
    "cement_bags": "cement_bags_model.joblib",
    "steel_tonnes": "steel_tonnes_model.joblib",
    "brick_count": "brick_count_model.joblib",
    "aac_block_count": "aac_block_count_model.joblib",
}

FEATURES = [
    "floor_area_sqft",
    "built_up_area_sqft",
    "num_floors",
    "num_rooms",
    "wall_length_ft",
    "wall_thickness_in",
    "wall_height_ft",
    "foundation_depth_ft",
    "foundation_area_sqft",
    "door_area_sqft",
    "window_area_sqft",
    "slab_area_sqft",
    "roof_area_sqft",
    "concrete_volume_m3",
    "slab_thickness_in",
    "soil_bearing_capacity_kpa",
    "structural_intensity",
]

SOIL_CAPACITY = {
    "weak": 100,
    "normal": 180,
    "good": 300,
}

MASONRY_TYPES = [
    "brick",
    "aac",
    "mixed",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ArchMind Pro",
    page_icon="🏗️",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🏗️ ArchMind Pro")

st.caption(
    "AI-powered construction material intelligence "
    "from 3D structural geometry"
)

st.success(
    f"🟢 System Online — {APP_VERSION}"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Project Configuration")

building_type = st.sidebar.selectbox(
    "Building Type",
    [
        "Small Residential",
        "Medium Residential",
        "Large Residential",
        "Small Commercial",
        "Medium Commercial",
        "Large Commercial",
        "Industrial",
        "Institutional",
    ],
)

num_floors_input = st.sidebar.number_input(
    "Selected Number of Floors",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
)

floor_height = st.sidebar.number_input(
    "Floor Height (ft)",
    min_value=7.0,
    max_value=30.0,
    value=10.0,
    step=0.5,
)

foundation_depth = st.sidebar.number_input(
    "Foundation Depth (ft)",
    min_value=1.0,
    max_value=30.0,
    value=5.0,
    step=0.5,
)

wall_thickness = st.sidebar.number_input(
    "Wall Thickness (in)",
    min_value=3.0,
    max_value=24.0,
    value=9.0,
    step=0.5,
)

soil_condition = st.sidebar.selectbox(
    "Soil Condition",
    ["weak", "normal", "good"],
    index=1,
)

masonry_type = st.sidebar.selectbox(
    "Masonry Type",
    MASONRY_TYPES,
)

structural_intensity = st.sidebar.slider(
    "Structural Intensity",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.05,
)

st.sidebar.divider()

uploaded_stl = st.sidebar.file_uploader(
    "Upload STL Building Model",
    type=["stl"],
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Blueprint → 3D",
        "STL Material Prediction",
        "Geometry Analysis",
        "Feature Preview",
        "About",
    ],
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading ArchMind ML models...")
def load_models_from_huggingface():

    models = {}
    errors = []

    for target, filename in TARGET_FILES.items():

        try:

            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
            )

            models[target] = joblib.load(
                model_path
            )

        except Exception as e:

            errors.append(
                f"{filename}: {str(e)}"
            )

    return models, errors


models, model_errors = (
    load_models_from_huggingface()
)


# ============================================================
# STL LOADING
# ============================================================

def load_stl_mesh(uploaded_file):

    if uploaded_file is None:
        raise ValueError(
            "No STL file provided."
        )

    uploaded_file.seek(0)

    mesh = trimesh.load(
        uploaded_file,
        file_type="stl",
        force="mesh",
    )

    if isinstance(mesh, trimesh.Scene):

        if len(mesh.geometry) == 0:
            raise ValueError(
                "STL scene contains no geometry."
            )

        mesh = trimesh.util.concatenate(
            tuple(
                mesh.geometry.values()
            )
        )

    if mesh is None:
        raise ValueError(
            "Unable to load STL."
        )

    if len(mesh.vertices) == 0:
        raise ValueError(
            "STL contains no vertices."
        )

    if len(mesh.faces) == 0:
        raise ValueError(
            "STL contains no faces."
        )

    return mesh


# ============================================================
# HELPERS
# ============================================================

def safe_div(a, b):

    if b is None or abs(b) < 1e-9:
        return 0.0

    return float(a / b)


def estimate_floor_count(
    height_ft,
    floor_height,
):

    if height_ft <= 0:
        return 1

    if floor_height <= 0:
        return 1

    estimated = int(
        round(
            height_ft /
            floor_height
        )
    )

    return max(
        1,
        min(
            estimated,
            100,
        )
    )


# ============================================================
# VERTICAL PROFILE ANALYSIS
# ============================================================

def analyze_vertical_profile(mesh):

    vertices = np.asarray(
        mesh.vertices
    )

    if len(vertices) < 10:
        return {
            "profile_area_cv": 1.0,
            "profile_width_cv": 1.0,
            "profile_depth_cv": 1.0,
            "top_bottom_area_ratio": 0.0,
            "profile_consistency": 0.0,
            "severe_taper": True,
        }

    mins = vertices.min(
        axis=0
    )

    maxs = vertices.max(
        axis=0
    )

    dims = maxs - mins

    vertical_axis = int(
        np.argmax(dims)
    )

    horizontal_axes = [
        i for i in range(3)
        if i != vertical_axis
    ]

    z_values = vertices[
        :, vertical_axis
    ]

    h1 = float(
        z_values.min()
    )

    h2 = float(
        z_values.max()
    )

    height = h2 - h1

    if height <= 0:
        return {
            "profile_area_cv": 1.0,
            "profile_width_cv": 1.0,
            "profile_depth_cv": 1.0,
            "top_bottom_area_ratio": 0.0,
            "profile_consistency": 0.0,
            "severe_taper": True,
        }

    # --------------------------------------------------------
    # Divide object vertically into slices.
    # --------------------------------------------------------

    num_slices = 12

    ratios = []

    widths = []

    depths = []

    for i in range(num_slices):

        lower = (
            h1 +
            height *
            i /
            num_slices
        )

        upper = (
            h1 +
            height *
            (i + 1) /
            num_slices
        )

        mask = (
            (z_values >= lower) &
            (z_values <= upper)
        )

        pts = vertices[mask]

        if len(pts) < 5:
            continue

        x_values = pts[
            :, horizontal_axes[0]
        ]

        y_values = pts[
            :, horizontal_axes[1]
        ]

        width = (
            x_values.max() -
            x_values.min()
        )

        depth = (
            y_values.max() -
            y_values.min()
        )

        area = width * depth

        widths.append(
            width
        )

        depths.append(
            depth
        )

        ratios.append(
            area
        )

    if len(ratios) < 4:

        return {
            "profile_area_cv": 1.0,
            "profile_width_cv": 1.0,
            "profile_depth_cv": 1.0,
            "top_bottom_area_ratio": 0.0,
            "profile_consistency": 0.0,
            "severe_taper": True,
        }

    ratios = np.asarray(
        ratios,
        dtype=float
    )

    widths = np.asarray(
        widths,
        dtype=float
    )

    depths = np.asarray(
        depths,
        dtype=float
    )

    area_mean = ratios.mean()

    width_mean = widths.mean()

    depth_mean = depths.mean()

    area_cv = safe_div(
        ratios.std(),
        area_mean,
    )

    width_cv = safe_div(
        widths.std(),
        width_mean,
    )

    depth_cv = safe_div(
        depths.std(),
        depth_mean,
    )

    bottom_area = np.mean(
        ratios[:max(1, len(ratios)//3)]
    )

    top_area = np.mean(
        ratios[-max(1, len(ratios)//3):]
    )

    top_bottom_ratio = safe_div(
        top_area,
        bottom_area,
    )

    # --------------------------------------------------------
    # Strong taper detection.
    #
    # Buildings generally maintain a reasonably consistent
    # footprint across floors.
    # Objects such as statues/towers can taper dramatically.
    # --------------------------------------------------------

    severe_taper = (
        top_bottom_ratio < 0.35
        or
        top_bottom_ratio > 1.80
    )

    # --------------------------------------------------------
    # Consistency score
    # --------------------------------------------------------

    inconsistency = (
        area_cv * 0.50
        +
        width_cv * 0.25
        +
        depth_cv * 0.25
    )

    consistency = (
        100 *
        max(
            0.0,
            min(
                1.0,
                1.0 -
                inconsistency,
            )
        )
    )

    return {
        "profile_area_cv": area_cv,
        "profile_width_cv": width_cv,
        "profile_depth_cv": depth_cv,
        "top_bottom_area_ratio":
            top_bottom_ratio,
        "profile_consistency":
            consistency,
        "severe_taper":
            severe_taper,
    }


# ============================================================
# FLOOR / HEIGHT ANALYSIS
# ============================================================

def calculate_height_metrics(
    height_ft,
    selected_floors,
    floor_height,
):

    expected_height = (
        selected_floors *
        floor_height
    )

    height_ratio = safe_div(
        height_ft,
        expected_height,
    )

    estimated_floors = (
        estimate_floor_count(
            height_ft,
            floor_height,
        )
    )

    detected_height = (
        estimated_floors *
        floor_height
    )

    detected_ratio = safe_div(
        height_ft,
        detected_height,
    )

    return {
        "expected_height":
            expected_height,

        "height_ratio":
            height_ratio,

        "estimated_floors":
            estimated_floors,

        "detected_expected_height":
            detected_height,

        "detected_ratio":
            detected_ratio,
    }


# ============================================================
# BUILDING CLASSIFICATION
# ============================================================

def classify_building_geometry(
    geometry,
):

    score = 100.0

    reasons = []

    # --------------------------------------------------------
    # Extract signals
    # --------------------------------------------------------

    fill = geometry[
        "volume_fill_ratio"
    ]

    components = geometry[
        "connected_components"
    ]

    profile = geometry[
        "vertical_profile"
    ]

    profile_consistency = profile[
        "profile_consistency"
    ]

    top_bottom_ratio = profile[
        "top_bottom_area_ratio"
    ]

    severe_taper = profile[
        "severe_taper"
    ]

    surface_volume = geometry[
        "surface_volume_ratio"
    ]

    height_ft = geometry[
        "height_ft"
    ]

    footprint_area = geometry[
        "footprint_area_sqft"
    ]

    # --------------------------------------------------------
    # Basic footprint sanity
    # --------------------------------------------------------

    if footprint_area < 50:

        score -= 25

        reasons.append(
            "Very small footprint"
        )

    # --------------------------------------------------------
    # Volume fill
    # --------------------------------------------------------

    if fill < 0.08:

        score -= 35

        reasons.append(
            "Very sparse geometry"
        )

    elif fill < 0.15:

        score -= 20

        reasons.append(
            "Low volume density"
        )

    elif fill < 0.22:

        score -= 8

        reasons.append(
            "Moderately sparse geometry"
        )

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    if components >= 15:

        score -= 25

        reasons.append(
            "Many disconnected components"
        )

    elif components >= 8:

        score -= 15

        reasons.append(
            "Several disconnected components"
        )

    elif components >= 5:

        score -= 5

        reasons.append(
            "Multiple disconnected components"
        )

    # --------------------------------------------------------
    # Surface / volume
    # --------------------------------------------------------

    if surface_volume > 0.20:

        score -= 15

        reasons.append(
            "High surface-to-volume ratio"
        )

    elif surface_volume > 0.12:

        score -= 5

    # --------------------------------------------------------
    # Vertical profile
    # --------------------------------------------------------

    if profile_consistency < 30:

        score -= 35

        reasons.append(
            "Highly inconsistent vertical footprint"
        )

    elif profile_consistency < 50:

        score -= 20

        reasons.append(
            "Inconsistent vertical footprint"
        )

    elif profile_consistency < 70:

        score -= 8

        reasons.append(
            "Moderately variable vertical footprint"
        )

    else:

        reasons.append(
            "Consistent vertical footprint"
        )

    # --------------------------------------------------------
    # Tapering
    # --------------------------------------------------------

    if severe_taper:

        score -= 30

        reasons.append(
            "Severe vertical taper detected"
        )

    elif (
        top_bottom_ratio < 0.60
        or
        top_bottom_ratio > 1.50
    ):

        score -= 12

        reasons.append(
            "Strong vertical taper detected"
        )

    # --------------------------------------------------------
    # Extreme height/footprint ratio
    # --------------------------------------------------------

    footprint_width = math.sqrt(
        max(
            footprint_area,
            1.0,
        )
    )

    height_ratio = safe_div(
        height_ft,
        footprint_width,
    )

    if height_ratio > 8:

        score -= 20

        reasons.append(
            "Extreme height-to-footprint ratio"
        )

    elif height_ratio > 5:

        score -= 8

        reasons.append(
            "High height-to-footprint ratio"
        )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    score = max(
        0,
        min(
            100,
            score,
        )
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if score >= 70:

        classification = (
            "Likely Building"
        )

    elif score >= 45:

        classification = (
            "Uncertain Structure"
        )

    else:

        classification = (
            "Likely Non-Building Object"
        )

    return {
        "confidence":
            score,

        "classification":
            classification,

        "reasons":
            reasons,
    }


# ============================================================
# COMPLETE GEOMETRY ANALYSIS
# ============================================================

def analyze_geometry(
    mesh,
    selected_floors,
    floor_height,
):

    bounds = mesh.bounds

    dimensions = (
        bounds[1] -
        bounds[0]
    )

    dimensions = np.asarray(
        dimensions,
        dtype=float,
    )

    sorted_dims = np.sort(
        dimensions
    )

    footprint_x = float(
        sorted_dims[0]
    )

    footprint_y = float(
        sorted_dims[1]
    )

    height = float(
        sorted_dims[2]
    )

    footprint_area = (
        footprint_x *
        footprint_y
    )

    volume = float(
        abs(mesh.volume)
    )

    surface_area = float(
        mesh.area
    )

    bounding_volume = float(
        np.prod(dimensions)
    )

    volume_fill = safe_div(
        volume,
        bounding_volume,
    )

    surface_volume = safe_div(
        surface_area,
        max(
            volume,
            1e-6,
        ),
    )

    components = len(
        mesh.split(
            only_watertight=False
        )
    )

    height_metrics = (
        calculate_height_metrics(
            height,
            selected_floors,
            floor_height,
        )
    )

    geometry = {

        "dimensions":
            dimensions,

        "footprint_x_ft":
            footprint_x,

        "footprint_y_ft":
            footprint_y,

        "footprint_area_sqft":
            footprint_area,

        "height_ft":
            height,

        "volume":
            volume,

        "surface_area":
            surface_area,

        "bounding_volume":
            bounding_volume,

        "volume_fill_ratio":
            volume_fill,

        "surface_volume_ratio":
            surface_volume,

        "connected_components":
            components,

        "watertight":
            bool(mesh.is_watertight),

        "vertices":
            len(mesh.vertices),

        "faces":
            len(mesh.faces),

    }

    geometry.update(
        height_metrics
    )

    # --------------------------------------------------------
    # New V2.5.1 vertical profile intelligence
    # --------------------------------------------------------

    geometry[
        "vertical_profile"
    ] = analyze_vertical_profile(
        mesh
    )

    # --------------------------------------------------------
    # Building classification
    # --------------------------------------------------------

    classification = (
        classify_building_geometry(
            geometry
        )
    )

    geometry.update(
        classification
    )

    return geometry


# ============================================================
# FLOOR CONFIRMATION
# ============================================================

def handle_floor_confirmation(
    geometry,
    selected_floors,
):

    estimated = geometry[
        "estimated_floors"
    ]

    mismatch = (
        abs(
            estimated -
            selected_floors
        )
        >= 2
    )

    if not mismatch:

        st.session_state[
            "confirmed_floors"
        ] = selected_floors

        return True

    st.warning(
        f"⚠️ Geometry suggests approximately "
        f"**{estimated} floors**, while the project "
        f"configuration currently has "
        f"**{selected_floors} floors**."
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            f"✅ Use detected {estimated} floors",
            type="primary",
            key=f"detected_{estimated}",
        ):

            st.session_state[
                "confirmed_floors"
            ] = estimated

            st.session_state[
                "floor_confirmation"
            ] = True

            st.rerun()

    with c2:

        if st.button(
            f"↩️ Keep {selected_floors} floors",
            key=f"keep_{selected_floors}",
        ):

            st.session_state[
                "confirmed_floors"
            ] = selected_floors

            st.session_state[
                "floor_confirmation"
            ] = False

            st.rerun()

    confirmed = st.session_state.get(
        "confirmed_floors"
    )

    return (
        confirmed is not None
        and
        confirmed == estimated
    )


# ============================================================
# BUILDING CLASSIFICATION DISPLAY
# ============================================================

def show_building_classification(
    geometry,
):

    st.subheader(
        "🏢 Building Classification"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Building Confidence",
            f"{geometry['confidence']:.0f}%",
        )

    with c2:

        st.metric(
            "Classification",
            geometry[
                "classification"
            ],
        )

    with c3:

        st.metric(
            "Estimated Floors",
            geometry[
                "estimated_floors"
            ],
        )

    classification = geometry[
        "classification"
    ]

    if classification == "Likely Building":

        st.success(
            "🏢 Geometry is structurally plausible "
            "as a building."
        )

    elif classification == "Uncertain Structure":

        st.warning(
            "⚠️ Geometry has mixed structural "
            "characteristics. Manual verification "
            "is recommended."
        )

    else:

        st.error(
            "🚫 Geometry is unlikely to represent "
            "a conventional building."
        )

    with st.expander(
        "🔎 Classification reasoning"
    ):

        for reason in geometry[
            "reasons"
        ]:

            st.write(
                f"• {reason}"
            )

        profile = geometry[
            "vertical_profile"
        ]

        st.write(
            f"Vertical profile consistency: "
            f"**{profile['profile_consistency']:.1f}%**"
        )

        st.write(
            f"Top / bottom footprint ratio: "
            f"**{profile['top_bottom_area_ratio']:.2f}×**"
        )


# ============================================================
# GEOMETRY METRICS DISPLAY
# ============================================================

def show_geometry_metrics(
    geometry,
):

    st.subheader(
        "📐 Geometry Analysis"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Footprint",
            f"{geometry['footprint_area_sqft']:,.0f} sqft",
        )

    with c2:

        st.metric(
            "Model Height",
            f"{geometry['height_ft']:.1f} ft",
        )

    with c3:

        st.metric(
            "Estimated Floors",
            geometry[
                "estimated_floors"
            ],
        )

    c4, c5, c6 = st.columns(3)

    with c4:

        st.metric(
            "Volume Fill",
            f"{geometry['volume_fill_ratio']:.2f}",
        )

    with c5:

        st.metric(
            "Components",
            geometry[
                "connected_components"
            ],
        )

    with c6:

        st.metric(
            "Faces",
            f"{geometry['faces']:,}",
        )

    st.divider()

    data = {
        "Metric": [
            "Footprint X",
            "Footprint Y",
            "Model Height",
            "Volume",
            "Surface Area",
            "Bounding Volume",
            "Volume Fill Ratio",
            "Surface / Volume",
            "Connected Components",
            "Watertight",
            "Vertices",
            "Faces",
            "Estimated Floors",
            "Profile Consistency",
            "Top / Bottom Ratio",
        ],

        "Value": [

            f"{geometry['footprint_x_ft']:.2f} ft",

            f"{geometry['footprint_y_ft']:.2f} ft",

            f"{geometry['height_ft']:.2f} ft",

            f"{geometry['volume']:,.2f}",

            f"{geometry['surface_area']:,.2f}",

            f"{geometry['bounding_volume']:,.2f}",

            f"{geometry['volume_fill_ratio']:.3f}",

            f"{geometry['surface_volume_ratio']:.4f}",

            geometry[
                "connected_components"
            ],

            "Yes"
            if geometry[
                "watertight"
            ]
            else "No",

            f"{geometry['vertices']:,}",

            f"{geometry['faces']:,}",

            geometry[
                "estimated_floors"
            ],

            f"{geometry['vertical_profile']['profile_consistency']:.1f}%",

            f"{geometry['vertical_profile']['top_bottom_area_ratio']:.2f}×",
        ],
    }

    st.dataframe(
        pd.DataFrame(data),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ============================================================
# V2.7 BLUEPRINT WALL RECONSTRUCTION
# ============================================================

def preprocess_blueprint(image):
    """
    V2.6 Step 1:
    Create several diagnostic representations instead of relying
    on one aggressive adaptive-threshold image.

    The goal is to preserve architectural wall lines while
    suppressing furniture/decorative noise as much as possible.
    """

    img = np.asarray(image.convert("RGB"))

    # Limit processing size for predictable Streamlit performance.
    max_dim = 1800
    h, w = img.shape[:2]

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edges are more useful for architectural line drawings than
    # a raw threshold on this type of rendered blueprint.
    edges = cv2.Canny(gray, 50, 150)

    # Close small gaps in wall lines.
    line_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )
    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        line_kernel,
        iterations=2,
    )

    # A second representation is retained for diagnostics.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    small_kernel = np.ones((3, 3), np.uint8)
    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        small_kernel,
        iterations=1,
    )

    # Combine edge and adaptive evidence.
    combined = cv2.bitwise_or(
        closed_edges,
        adaptive,
    )

    return img, gray, edges, closed_edges, adaptive, combined


def detect_floorplan_bounds(binary):
    """
    Detect the most plausible central architectural plan boundary.

    We deliberately avoid simply selecting contours[0], because a
    rendered blueprint often contains an outer page frame, compass,
    furniture and decorative objects.
    """

    h, w = binary.shape[:2]
    image_area = float(h * w)

    # Ignore a small page margin so the outer image border is less
    # likely to become the floor-plan contour.
    margin_x = max(5, int(w * 0.03))
    margin_y = max(5, int(h * 0.03))

    roi = binary[
        margin_y:h - margin_y,
        margin_x:w - margin_x,
    ]

    contours, _ = cv2.findContours(
        roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.02:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)

        if cw < w * 0.12 or ch < h * 0.12:
            continue

        rect_area = float(cw * ch)
        fill_ratio = safe_div(area, rect_area)

        aspect = safe_div(cw, ch)

        # Most architectural floor plans are not tiny extreme
        # aspect-ratio objects.
        if aspect < 0.12 or aspect > 8.0:
            continue

        # Prefer large, reasonably filled, central candidates.
        center_x = x + cw / 2
        center_y = y + ch / 2

        roi_cx = roi.shape[1] / 2
        roi_cy = roi.shape[0] / 2

        center_distance = (
            safe_div(
                abs(center_x - roi_cx),
                max(roi.shape[1], 1),
            )
            +
            safe_div(
                abs(center_y - roi_cy),
                max(roi.shape[0], 1),
            )
        )

        size_score = min(
            1.0,
            rect_area / (image_area * 0.75),
        )

        fill_score = min(
            1.0,
            max(0.0, fill_ratio),
        )

        central_score = max(
            0.0,
            1.0 - center_distance,
        )

        score = (
            size_score * 0.55
            +
            fill_score * 0.20
            +
            central_score * 0.25
        )

        candidates.append(
            (
                score,
                x + margin_x,
                y + margin_y,
                cw,
                ch,
                area,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    score, x, y, cw, ch, area = candidates[0]

    return {
        "x": int(x),
        "y": int(y),
        "width_px": int(cw),
        "height_px": int(ch),
        "area_px": float(area),
        "score": float(score),
    }


def detect_architectural_lines(edges):
    """
    Detect long horizontal/vertical lines.

    These are not yet room walls. They are V2.6 structural
    candidates used to build the next reconstruction stage.
    """

    h, w = edges.shape[:2]

    min_horizontal = max(30, int(w * 0.08))
    min_vertical = max(30, int(h * 0.08))

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, min_horizontal // 4), 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(3, min_vertical // 4)),
    )

    horizontal = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    combined = cv2.bitwise_or(
        horizontal,
        vertical,
    )

    # Count detected line pixels as a simple diagnostic signal.
    line_pixels = int(
        np.count_nonzero(combined)
    )

    return horizontal, vertical, combined, line_pixels


def crop_blueprint_region(image_array, bounds, padding=8):
    """
    Crop the detected plan for downstream processing.
    """

    if bounds is None:
        return image_array

    h, w = image_array.shape[:2]

    x1 = max(
        0,
        bounds["x"] - padding,
    )
    y1 = max(
        0,
        bounds["y"] - padding,
    )
    x2 = min(
        w,
        bounds["x"]
        + bounds["width_px"]
        + padding,
    )
    y2 = min(
        h,
        bounds["y"]
        + bounds["height_px"]
        + padding,
    )

    return image_array[
        y1:y2,
        x1:x2,
    ]



def detect_wall_segments(plan_gray, min_length_ratio=0.035, max_segments=45):
    """V2.7.2 structural wall candidate detector.

    Hough lines are treated as raw geometric evidence, not walls. The
    detector scores candidates using length, continuity, intersections and
    proximity to a parallel line. This suppresses many furniture/detail
    strokes before they reach the 3D reconstruction stage.
    """
    gray = np.asarray(plan_gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    if h < 20 or w < 20:
        return [], np.zeros_like(gray)

    edges = cv2.Canny(gray, 50, 150)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    min_len = max(28, int(min(h, w) * min_length_ratio))
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(22, int(min(h, w) * 0.025)),
        minLineLength=min_len,
        maxLineGap=max(10, int(min(h, w) * 0.02)),
    )

    if raw is None:
        return [], edges

    raw = np.asarray(raw)
    if raw.size == 0:
        return [], edges
    try:
        raw_lines = raw.reshape(-1, 4)
    except ValueError:
        return [], edges

    horizontal = []
    vertical = []

    for line in raw_lines:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue

        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle

        if angle <= 6:
            horizontal.append(((y1 + y2) / 2.0, min(x1, x2), max(x1, x2)))
        elif abs(angle - 90) <= 6:
            vertical.append(((x1 + x2) / 2.0, min(y1, y2), max(y1, y2)))

    def merge_segments(items, axis_limit, coord_tol=7, gap_tol=14):
        if not items:
            return []
        items = sorted(items, key=lambda v: (v[0], v[1]))
        clusters = []
        current = [items[0]]
        for item in items[1:]:
            if abs(item[0] - np.mean([v[0] for v in current])) <= coord_tol:
                current.append(item)
            else:
                clusters.append(current)
                current = [item]
        clusters.append(current)

        output = []
        for cluster in clusters:
            coord = float(np.median([v[0] for v in cluster]))
            intervals = sorted((v[1], v[2]) for v in cluster)
            merged = []
            start, end = intervals[0]
            for aa, bb in intervals[1:]:
                if aa <= end + gap_tol:
                    end = max(end, bb)
                else:
                    if end - start >= max(30, axis_limit * 0.025):
                        merged.append((start, end))
                    start, end = aa, bb
            if end - start >= max(30, axis_limit * 0.025):
                merged.append((start, end))
            for start, end in merged:
                output.append((coord, float(start), float(end)))
        return output

    h_segments = merge_segments(horizontal, w)
    v_segments = merge_segments(vertical, h)

    candidates = []
    for y, x1, x2 in h_segments:
        length = x2 - x1
        if length >= min_len:
            candidates.append({
                "orientation": "horizontal",
                "x1": x1, "y1": y,
                "x2": x2, "y2": y,
                "length_px": float(length),
            })
    for x, y1, y2 in v_segments:
        length = y2 - y1
        if length >= min_len:
            candidates.append({
                "orientation": "vertical",
                "x1": x, "y1": y1,
                "x2": x, "y2": y2,
                "length_px": float(length),
            })

    if not candidates:
        return [], edges

    # Geometry-only structural scoring.
    def overlap_ratio(a1, a2, b1, b2):
        inter = max(0.0, min(a2, b2) - max(a1, b1))
        denom = max(1.0, min(a2 - a1, b2 - b1))
        return inter / denom

    for i, seg in enumerate(candidates):
        length = seg["length_px"]
        length_score = min(1.0, length / max(1.0, min(h, w) * 0.45))
        parallel_support = 0.0
        intersection_support = 0.0

        if seg["orientation"] == "horizontal":
            for j, other in enumerate(candidates):
                if i == j or other["orientation"] != "horizontal":
                    continue
                y_gap = abs(seg["y1"] - other["y1"])
                if 3 <= y_gap <= 18:
                    ov = overlap_ratio(seg["x1"], seg["x2"], other["x1"], other["x2"])
                    parallel_support = max(parallel_support, ov)
            for other in candidates:
                if other["orientation"] == "vertical":
                    if other["x1"] >= seg["x1"] - 2 and other["x1"] <= seg["x2"] + 2:
                        if other["y1"] <= seg["y1"] <= other["y2"]:
                            intersection_support += 1
        else:
            for j, other in enumerate(candidates):
                if i == j or other["orientation"] != "vertical":
                    continue
                x_gap = abs(seg["x1"] - other["x1"])
                if 3 <= x_gap <= 18:
                    ov = overlap_ratio(seg["y1"], seg["y2"], other["y1"], other["y2"])
                    parallel_support = max(parallel_support, ov)
            for other in candidates:
                if other["orientation"] == "horizontal":
                    if other["y1"] >= seg["y1"] - 2 and other["y1"] <= seg["y2"] + 2:
                        if other["x1"] <= seg["x1"] <= other["x2"]:
                            intersection_support += 1

        # Longer + connected + paired lines are stronger structural evidence.
        seg["structural_score"] = (
            0.58 * length_score
            + 0.27 * min(1.0, parallel_support)
            + 0.15 * min(1.0, intersection_support / 4.0)
        )
        seg["paired_support"] = float(parallel_support)

    # Keep strong candidates, but retain enough lines for complex plans.
    candidates.sort(key=lambda s: (s["structural_score"], s["length_px"]), reverse=True)
    strong = [s for s in candidates if s["structural_score"] >= 0.18]
    if len(strong) < 12:
        strong = candidates[:min(24, len(candidates))]
    else:
        strong = strong[:max_segments]

    # Remove near-duplicate lines that survive scoring.
    filtered = []
    for seg in strong:
        duplicate = False
        for kept in filtered:
            if seg["orientation"] != kept["orientation"]:
                continue
            if seg["orientation"] == "horizontal":
                coord_gap = abs(seg["y1"] - kept["y1"])
                ov = overlap_ratio(seg["x1"], seg["x2"], kept["x1"], kept["x2"])
            else:
                coord_gap = abs(seg["x1"] - kept["x1"])
                ov = overlap_ratio(seg["y1"], seg["y2"], kept["y1"], kept["y2"])
            if coord_gap <= 5 and ov >= 0.75:
                duplicate = True
                break
        if not duplicate:
            filtered.append(seg)

    overlay = np.zeros_like(gray)
    for seg in filtered:
        cv2.line(
            overlay,
            (int(seg["x1"]), int(seg["y1"])),
            (int(seg["x2"]), int(seg["y2"])),
            255,
            2,
        )

    return filtered, overlay

def reconstruct_wall_network(gray, segments):
    """V2.7.4: convert line candidates into a cleaner architectural wall graph.

    The previous stage selected Hough evidence. This stage deliberately works
    on that evidence instead of re-running Hough, and applies:
      1. orientation normalization
      2. close parallel-line consolidation
      3. interval merging
      4. endpoint extension to nearby perpendicular walls
      5. removal of short isolated fragments
      6. connected-component filtering on the resulting network

    The output is still geometric wall evidence, not semantic room recognition.
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    if not segments:
        return [], np.zeros_like(gray), np.zeros_like(gray)

    raw = []
    for seg in segments:
        try:
            x1, y1 = float(seg["x1"]), float(seg["y1"])
            x2, y2 = float(seg["x2"]), float(seg["y2"])
        except Exception:
            continue
        if seg.get("orientation") == "horizontal":
            y = (y1 + y2) / 2.0
            a, b = sorted((x1, x2))
            if b - a >= max(20, w * 0.025):
                raw.append({"orientation": "horizontal", "coord": y, "a": a, "b": b})
        else:
            x = (x1 + x2) / 2.0
            a, b = sorted((y1, y2))
            if b - a >= max(20, h * 0.025):
                raw.append({"orientation": "vertical", "coord": x, "a": a, "b": b})

    if not raw:
        return [], np.zeros_like(gray), np.zeros_like(gray)

    # Collapse very close parallel lines. If two detected lines overlap heavily,
    # they are more likely the two edges of the same wall than two separate walls.
    def consolidate_parallel(items, coord_tol=11, overlap_threshold=0.45):
        if not items:
            return []
        items = sorted(items, key=lambda q: (q["coord"], q["a"]))
        groups = []
        for item in items:
            placed = False
            for group in reversed(groups[-3:]):
                mean_coord = float(np.mean([g["coord"] for g in group]))
                if abs(item["coord"] - mean_coord) <= coord_tol:
                    group.append(item)
                    placed = True
                    break
            if not placed:
                groups.append([item])

        out = []
        for group in groups:
            # Split a coordinate cluster when intervals are clearly disjoint.
            intervals = sorted((g["a"], g["b"], g["coord"]) for g in group)
            merged_intervals = []
            for aa, bb, cc in intervals:
                if not merged_intervals:
                    merged_intervals.append([aa, bb, cc, 1])
                    continue
                prev = merged_intervals[-1]
                overlap = max(0.0, min(prev[1], bb) - max(prev[0], aa))
                denom = max(1.0, min(prev[1] - prev[0], bb - aa))
                if overlap / denom >= overlap_threshold or aa <= prev[1] + 16:
                    prev[0] = min(prev[0], aa)
                    prev[1] = max(prev[1], bb)
                    prev[2] = (prev[2] * prev[3] + cc) / (prev[3] + 1)
                    prev[3] += 1
                else:
                    merged_intervals.append([aa, bb, cc, 1])
            for aa, bb, cc, count in merged_intervals:
                out.append({"coord": cc, "a": aa, "b": bb, "support": count})
        return out

    hs = consolidate_parallel([r for r in raw if r["orientation"] == "horizontal"])
    vs = consolidate_parallel([r for r in raw if r["orientation"] == "vertical"])

    # Merge intervals that are on the same structural line.
    def merge_axis(items, coord_tol=9, gap_tol=22):
        if not items:
            return []
        items = sorted(items, key=lambda q: (q["coord"], q["a"]))
        clusters = []
        for item in items:
            if not clusters or abs(item["coord"] - np.mean([x["coord"] for x in clusters[-1]])) > coord_tol:
                clusters.append([item])
            else:
                clusters[-1].append(item)
        out = []
        for cluster in clusters:
            coord = float(np.median([x["coord"] for x in cluster]))
            ints = sorted((x["a"], x["b"], x.get("support", 1)) for x in cluster)
            cur_a, cur_b, cur_support = ints[0]
            for aa, bb, support in ints[1:]:
                if aa <= cur_b + gap_tol:
                    cur_b = max(cur_b, bb)
                    cur_support += support
                else:
                    out.append((coord, cur_a, cur_b, cur_support))
                    cur_a, cur_b, cur_support = aa, bb, support
            out.append((coord, cur_a, cur_b, cur_support))
        return out

    h_lines = merge_axis(hs)
    v_lines = merge_axis(vs)

    # Extend endpoints when a perpendicular structural line reaches the vicinity.
    refined = []
    extension = max(8.0, min(w, h) * 0.018)
    min_final = max(28.0, min(w, h) * 0.045)

    for y, x1, x2, support in h_lines:
        left = x1
        right = x2
        for x, y1, y2, _ in v_lines:
            if y1 - extension <= y <= y2 + extension:
                if abs(x - left) <= extension:
                    left = min(left, x)
                if abs(x - right) <= extension:
                    right = max(right, x)
        if right - left >= min_final:
            refined.append({
                "orientation": "horizontal", "x1": left, "y1": y,
                "x2": right, "y2": y, "support": int(support),
            })

    for x, y1, y2, support in v_lines:
        top = y1
        bottom = y2
        for y, x1, x2, _ in h_lines:
            if x1 - extension <= x <= x2 + extension:
                if abs(y - top) <= extension:
                    top = min(top, y)
                if abs(y - bottom) <= extension:
                    bottom = max(bottom, y)
        if bottom - top >= min_final:
            refined.append({
                "orientation": "vertical", "x1": x, "y1": top,
                "x2": x, "y2": bottom, "support": int(support),
            })

    # Remove near-duplicates after extension.
    final = []
    for seg in sorted(refined, key=lambda q: ((q.get("support", 1)),
                                                math.hypot(q["x2"]-q["x1"], q["y2"]-q["y1"])), reverse=True):
        duplicate = False
        for keep in final:
            if seg["orientation"] != keep["orientation"]:
                continue
            if seg["orientation"] == "horizontal":
                if abs(seg["y1"] - keep["y1"]) <= 9:
                    ov = max(0.0, min(seg["x2"], keep["x2"]) - max(seg["x1"], keep["x1"]))
                    denom = max(1.0, min(seg["x2"]-seg["x1"], keep["x2"]-keep["x1"]))
                    if ov / denom >= 0.70:
                        duplicate = True
                        break
            else:
                if abs(seg["x1"] - keep["x1"]) <= 9:
                    ov = max(0.0, min(seg["y2"], keep["y2"]) - max(seg["y1"], keep["y1"]))
                    denom = max(1.0, min(seg["y2"]-seg["y1"], keep["y2"]-keep["y1"]))
                    if ov / denom >= 0.70:
                        duplicate = True
                        break
        if not duplicate:
            final.append(seg)

    # Build two diagnostics: centerline network and a slightly dilated wall mask.
    centerline = np.zeros_like(gray)
    for seg in final:
        cv2.line(centerline,
                 (int(round(seg["x1"])), int(round(seg["y1"]))),
                 (int(round(seg["x2"])), int(round(seg["y2"]))),
                 255, 2)

    thickness = max(3, int(round(min(w, h) * 0.004)))
    wall_mask = cv2.dilate(centerline, np.ones((thickness, thickness), np.uint8), iterations=1)
    wall_mask = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    # Attach dimensions required by the existing 3D generator later.
    for seg in final:
        seg["plan_width_px"] = int(w)
        seg["plan_height_px"] = int(h)
        seg["length_px"] = float(math.hypot(seg["x2"]-seg["x1"], seg["y2"]-seg["y1"]))

    return final, centerline, wall_mask


def build_architectural_wall_mask(gray, segments, wall_px=5):
    """Compatibility wrapper for the V2.7.3 diagnostic wall mask."""
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg in segments or []:
        cv2.line(mask,
                 (int(seg["x1"]), int(seg["y1"])),
                 (int(seg["x2"]), int(seg["y2"])),
                 255, max(2, int(wall_px)))
    return mask

def detect_rooms_from_wall_network(
    gray,
    wall_centerline,
    plan_width_ft=None,
    plan_depth_ft=None,
):
    """V2.8.0: detect enclosed floor-plan spaces from the wall network.

    This stage deliberately detects *spaces*, not semantic room names.
    It uses the wall centerline as a barrier, closes small architectural
    openings such as door gaps, flood-fills the exterior, then extracts
    sufficiently large enclosed regions as room candidates.

    The result is an evidence-based room/space candidate map. Labels such
    as Bedroom, Kitchen, Toilet, etc. are reserved for a later OCR/semantic
    stage.
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return [], np.zeros((h, w), np.uint8), np.zeros((h, w, 3), np.uint8), {
            "room_count": 0,
            "candidate_count": 0,
            "min_area_px": 0,
            "wall_thickness_px": 0,
        }

    if wall_centerline is None or np.count_nonzero(wall_centerline) == 0:
        return [], np.zeros((h, w), np.uint8), np.zeros((h, w, 3), np.uint8), {
            "room_count": 0,
            "candidate_count": 0,
            "min_area_px": 0,
            "wall_thickness_px": 0,
        }

    # --------------------------------------------------------
    # Build a barrier from the consolidated wall centerlines.
    # Door gaps and tiny line breaks should not merge two rooms into
    # one giant connected free-space component, so use moderate
    # morphology rather than a very large dilation.
    # --------------------------------------------------------
    center = (wall_centerline > 0).astype(np.uint8) * 255

    wall_thickness_px = 5

    kernel_size = wall_thickness_px if wall_thickness_px % 2 == 1 else wall_thickness_px + 1
    barrier = cv2.dilate(
        center,
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1,
    )

    # Bridge small gaps caused by doors, anti-aliasing and imperfect Hough
    # reconstruction. Keep this intentionally modest to avoid swallowing
    # narrow rooms/corridors.
    gap_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7),
    )
    barrier = cv2.morphologyEx(
        barrier,
        cv2.MORPH_CLOSE,
        gap_kernel,
        iterations=1,
    )

    # Ensure the exterior boundary is closed. The existing network normally
    # contains the outer walls, but this small frame prevents a broken corner
    # from allowing flood-fill to leak into the building.
    frame_t = max(3, wall_thickness_px // 2)
    cv2.rectangle(
        barrier,
        (frame_t, frame_t),
        (w - 1 - frame_t, h - 1 - frame_t),
        255,
        frame_t,
    )

    # --------------------------------------------------------
    # Free-space connected components after removing the exterior.
    # --------------------------------------------------------
    free = (barrier == 0).astype(np.uint8)

    outside = free.copy()
    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8,
    )
    cv2.floodFill(
        outside,
        flood_mask,
        (0, 0),
        0,
        flags=8,
    )

    enclosed = (outside > 0).astype(np.uint8)

    # --------------------------------------------------------
    # Candidate threshold is relative to the plan size. For the supplied
    # architectural drawing this keeps normal rooms while rejecting tiny
    # furniture/text islands. It is deliberately not a fixed room count.
    # --------------------------------------------------------
    plan_area = float(h * w)
    min_area_px = max(
        500,
        int(plan_area * 0.00125),
    )
    max_area_px = int(plan_area * 0.45)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        enclosed,
        connectivity=8,
    )

    candidates = []

    for label_id in range(1, n_labels):
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < min_area_px or area_px > max_area_px:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        if bw < max(22, int(w * 0.035)) or bh < max(22, int(h * 0.015)):
            continue

        bbox_area = max(1, bw * bh)
        rectangularity = safe_div(area_px, bbox_area)
        aspect = safe_div(max(bw, bh), max(1, min(bw, bh)))

        # Extremely thin regions are more likely to be annotation remnants
        # or circulation slivers than rooms.
        if aspect > 12.0 and area_px < min_area_px * 2:
            continue

        cx, cy = centroids[label_id]

        if plan_width_ft and plan_depth_ft:
            px_to_ft_x = safe_div(plan_width_ft, w)
            px_to_ft_y = safe_div(plan_depth_ft, h)
            area_sqft = area_px * px_to_ft_x * px_to_ft_y
            width_ft = bw * px_to_ft_x
            depth_ft = bh * px_to_ft_y
        else:
            area_sqft = None
            width_ft = None
            depth_ft = None

        # A simple evidence score. This is NOT a semantic classifier.
        score = 50.0
        if rectangularity >= 0.55:
            score += 20
        elif rectangularity >= 0.35:
            score += 10
        if area_px >= min_area_px * 2:
            score += 15
        if aspect <= 8:
            score += 10
        if bw >= w * 0.05 and bh >= h * 0.03:
            score += 5
        score = max(0.0, min(100.0, score))

        candidates.append({
            "id": len(candidates) + 1,
            "label": int(label_id),
            "area_px": area_px,
            "x": x,
            "y": y,
            "width_px": bw,
            "height_px": bh,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "rectangularity": rectangularity,
            "aspect_ratio": aspect,
            "area_sqft": area_sqft,
            "width_ft": width_ft,
            "depth_ft": depth_ft,
            "confidence": score,
        })

    # Sort spatially from top to bottom, then left to right, and renumber.
    candidates.sort(key=lambda r: (r["centroid_y"], r["centroid_x"]))
    for idx, room in enumerate(candidates, 1):
        room["id"] = idx

    # --------------------------------------------------------
    # Diagnostic images.
    # --------------------------------------------------------
    room_mask = np.zeros((h, w), np.uint8)
    overlay = np.zeros((h, w, 3), np.uint8)

    for room in candidates:
        component = (labels == room["label"]).astype(np.uint8) * 255
        room_mask[component > 0] = 255
        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 255, 255),
            2,
        )
        cv2.circle(
            overlay,
            (int(round(room["centroid_x"])), int(round(room["centroid_y"]))),
            5,
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            overlay,
            str(room["id"]),
            (
                int(round(room["centroid_x"])) + 7,
                int(round(room["centroid_y"])) - 7,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Add the barrier itself to the diagnostic so room boundaries remain clear.
    overlay[barrier > 0] = (255, 255, 255)

    # Re-draw labels after the barrier so they remain visible.
    for room in candidates:
        cv2.circle(
            overlay,
            (int(round(room["centroid_x"])), int(round(room["centroid_y"]))),
            5,
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            overlay,
            str(room["id"]),
            (
                int(round(room["centroid_x"])) + 7,
                int(round(room["centroid_y"])) - 7,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    diagnostics = {
        "room_count": len(candidates),
        "candidate_count": len(candidates),
        "min_area_px": min_area_px,
        "wall_thickness_px": wall_thickness_px,
        "enclosed_pixel_area": int(np.count_nonzero(enclosed)),
    }

    return candidates, room_mask, overlay, diagnostics


def render_room_table(rooms):
    """Render V2.8 room/space candidates as a compact table."""
    if not rooms:
        return

    rows = []
    for room in rooms:
        rows.append({
            "Space": f"Space {room['id']}",
            "Area (sqft)": (
                round(room["area_sqft"], 1)
                if room.get("area_sqft") is not None
                else None
            ),
            "Width (ft)": (
                round(room["width_ft"], 1)
                if room.get("width_ft") is not None
                else None
            ),
            "Depth (ft)": (
                round(room["depth_ft"], 1)
                if room.get("depth_ft") is not None
                else None
            ),
            "Geometry Confidence": f"{room['confidence']:.0f}%",
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def create_wall_reconstruction_mesh(
    segments,
    plan_width_ft,
    plan_depth_ft,
    floor_number,
    floor_height,
    wall_thickness_in,
):
    """Create a 3D reconstruction for ONE uploaded floor plan.

    V2.7.3 intentionally does not duplicate one blueprint across all floors.
    Each uploaded blueprint represents one floor and is positioned at its
    declared floor number. Multi-floor stacking will be implemented later
    when multiple floor plans can be uploaded and aligned independently.
    """
    if not segments:
        return None

    wall_t = max(0.15, wall_thickness_in / 12.0)
    floor_idx = max(0, int(floor_number) - 1)
    z_base = floor_idx * floor_height
    parts = []

    for seg in segments:
        pw = float(seg.get("plan_width_px", 1))
        ph = float(seg.get("plan_height_px", 1))
        if pw <= 1 or ph <= 1:
            continue

        x1 = np.clip(seg["x1"], 0, pw)
        x2 = np.clip(seg["x2"], 0, pw)
        y1 = np.clip(seg["y1"], 0, ph)
        y2 = np.clip(seg["y2"], 0, ph)

        fx1 = (x1 / pw - 0.5) * plan_width_ft
        fx2 = (x2 / pw - 0.5) * plan_width_ft
        fy1 = (0.5 - y1 / ph) * plan_depth_ft
        fy2 = (0.5 - y2 / ph) * plan_depth_ft

        if seg["orientation"] == "horizontal":
            length = abs(fx2 - fx1)
            if length < 1.5:
                continue
            center = [(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0, z_base + floor_height / 2.0]
            wall = trimesh.creation.box(extents=[length, wall_t, floor_height])
        else:
            length = abs(fy2 - fy1)
            if length < 1.5:
                continue
            center = [(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0, z_base + floor_height / 2.0]
            wall = trimesh.creation.box(extents=[wall_t, length, floor_height])

        wall.apply_translation(center)
        parts.append(wall)
        if len(parts) >= 180:
            break

    if not parts:
        return None

    slab_t = 0.5
    slab = trimesh.creation.box(extents=[plan_width_ft, plan_depth_ft, slab_t])
    slab.apply_translation([0, 0, z_base])
    parts.append(slab)

    roof = trimesh.creation.box(extents=[plan_width_ft, plan_depth_ft, slab_t])
    roof.apply_translation([0, 0, z_base + floor_height])
    parts.append(roof)

    return trimesh.util.concatenate(parts)

# PROCEDURAL 3D BUILDING
# ============================================================

def create_building_mesh(
    width_ft,
    depth_ft,
    num_floors,
    floor_height,
    wall_thickness_in,
):

    wall_t = (
        wall_thickness_in /
        12.0
    )

    total_height = (
        num_floors *
        floor_height
    )

    parts = []

    # --------------------------------------------------------
    # Outer walls
    # --------------------------------------------------------

    front = trimesh.creation.box(
        extents=[
            width_ft,
            wall_t,
            total_height,
        ]
    )

    front.apply_translation(
        [
            0,
            -depth_ft / 2,
            total_height / 2,
        ]
    )

    parts.append(
        front
    )

    back = trimesh.creation.box(
        extents=[
            width_ft,
            wall_t,
            total_height,
        ]
    )

    back.apply_translation(
        [
            0,
            depth_ft / 2,
            total_height / 2,
        ]
    )

    parts.append(
        back
    )

    left = trimesh.creation.box(
        extents=[
            wall_t,
            depth_ft,
            total_height,
        ]
    )

    left.apply_translation(
        [
            -width_ft / 2,
            0,
            total_height / 2,
        ]
    )

    parts.append(
        left
    )

    right = trimesh.creation.box(
        extents=[
            wall_t,
            depth_ft,
            total_height,
        ]
    )

    right.apply_translation(
        [
            width_ft / 2,
            0,
            total_height / 2,
        ]
    )

    parts.append(
        right
    )

    # --------------------------------------------------------
    # Floor slabs
    # --------------------------------------------------------

    slab_thickness = 0.5

    for floor in range(
        1,
        num_floors + 1,
    ):

        slab = trimesh.creation.box(
            extents=[
                width_ft,
                depth_ft,
                slab_thickness,
            ]
        )

        slab.apply_translation(
            [
                0,
                0,
                floor *
                floor_height,
            ]
        )

        parts.append(
            slab
        )

    # --------------------------------------------------------
    # Roof
    # --------------------------------------------------------

    roof = trimesh.creation.box(
        extents=[
            width_ft,
            depth_ft,
            slab_thickness,
        ]
    )

    roof.apply_translation(
        [
            0,
            0,
            total_height,
        ]
    )

    parts.append(
        roof
    )

    # --------------------------------------------------------
    # Internal partition
    # --------------------------------------------------------

    if width_ft > 20:

        partition = trimesh.creation.box(
            extents=[
                wall_t,
                max(
                    1,
                    depth_ft -
                    2 * wall_t,
                ),
                total_height,
            ]
        )

        partition.apply_translation(
            [
                0,
                0,
                total_height / 2,
            ]
        )

        parts.append(
            partition
        )

    return trimesh.util.concatenate(
        parts
    )


def export_mesh_stl(mesh):

    buffer = io.BytesIO()

    mesh.export(
        buffer,
        file_type="stl",
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# ML FEATURE GENERATION
# ============================================================

def build_prediction_features(
    geometry,
    num_floors,
    floor_height,
    foundation_depth,
    wall_thickness,
    soil_condition,
    masonry_type,
    structural_intensity,
):

    floor_area = geometry[
        "footprint_area_sqft"
    ]

    built_up_area = (
        floor_area *
        num_floors
    )

    wall_length = (
        2 *
        (
            geometry[
                "footprint_x_ft"
            ]
            +
            geometry[
                "footprint_y_ft"
            ]
        )
        *
        num_floors
    )

    foundation_area = (
        floor_area
    )

    wall_height = (
        floor_height
    )

    slab_area = (
        floor_area
    )

    roof_area = (
        floor_area
    )

    num_rooms_per_floor = max(
        2,
        int(
            round(
                floor_area /
                250
            )
        ),
    )

    num_rooms = (
        num_rooms_per_floor *
        num_floors
    )

    door_area = (
        built_up_area *
        0.025
    )

    window_area = (
        built_up_area *
        0.12
    )

    # --------------------------------------------------------
    # Use geometry volume where possible.
    #
    # STL volume is assumed to be approximately m³ only
    # when the source model was constructed in those units.
    # This remains an engineering-prototyping approximation.
    # --------------------------------------------------------

    concrete_volume = (
        geometry["volume"]
        if geometry["volume"] > 0
        else
        built_up_area *
        0.18 /
        35.3147
    )

    features = {

        "floor_area_sqft":
            floor_area,

        "built_up_area_sqft":
            built_up_area,

        "num_floors":
            num_floors,

        "num_rooms":
            num_rooms,

        "wall_length_ft":
            wall_length,

        "wall_thickness_in":
            wall_thickness,

        "wall_height_ft":
            wall_height,

        "foundation_depth_ft":
            foundation_depth,

        "foundation_area_sqft":
            foundation_area,

        "door_area_sqft":
            door_area,

        "window_area_sqft":
            window_area,

        "slab_area_sqft":
            slab_area,

        "roof_area_sqft":
            roof_area,

        "concrete_volume_m3":
            concrete_volume,

        "slab_thickness_in":
            6.0,

        "soil_bearing_capacity_kpa":
            SOIL_CAPACITY[
                soil_condition
            ],

        "structural_intensity":
            structural_intensity,
    }

    return pd.DataFrame(
        [
            [
                features[f]
                for f in FEATURES
            ]
        ],
        columns=FEATURES,
    )


# ============================================================
# MATERIAL PREDICTION
# ============================================================

def predict_materials(
    feature_df
):

    predictions = {}

    for target, model in models.items():

        try:

            value = model.predict(
                feature_df
            )[0]

            predictions[
                target
            ] = max(
                0,
                float(value),
            )

        except Exception as e:

            predictions[
                target
            ] = None

            st.error(
                f"{target} prediction failed: {e}"
            )

    return predictions


# ============================================================
# ============================================================
# PAGE: BLUEPRINT → 3D
# ============================================================

if page == "Blueprint → 3D":

    st.header("🖼️ Blueprint → 3D Reconstruction")

    st.write(
        "V2.6 Step 1 analyzes the blueprint using computer vision "
        "to identify the primary plan region and long architectural "
        "line candidates before 3D reconstruction."
    )

    st.info(
        "🧱 V2.7 is a geometry-first reconstruction stage. "
        "It does not yet claim semantic room/door/furniture recognition; "
        "the detected wall network is an intermediate reconstruction."
    )

    blueprint = st.file_uploader(
        "Upload Blueprint / Floor Plan",
        type=["png", "jpg", "jpeg"],
        key="blueprint",
    )

    st.subheader("🏢 Floor Configuration")

    floor_number_input = st.number_input(
        "Which floor does this blueprint represent?",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        help=(
            "Each uploaded blueprint represents ONE floor. Enter its floor number; "
            "ArchMind will not duplicate this blueprint across other floors."
        ),
    )

    st.info(
        "This upload is treated as a single floor plan. Multiple floor uploads "
        "will be aligned and stacked in a later multi-floor reconstruction stage."
    )

    st.subheader("📏 Scale Calibration")

    calibration_mode = st.radio(
        "What real-world dimension do you know?",
        [
            "Overall Building Width",
            "Overall Building Depth",
            "No Overall Dimension Available",
        ],
        horizontal=True,
        help=(
            "Use the outside-to-outside dimension of the complete floor plan. "
            "Do not enter a room dimension such as 10 ft or 11 ft as the overall width."
        ),
    )

    reference_dimension = st.number_input(
        "Known Overall Dimension (ft)",
        min_value=5.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
        disabled=(calibration_mode == "No Overall Dimension Available"),
        help=(
            "Enter a dimension measured across the entire building footprint. "
            "For example, if the whole plan is 35 ft wide, enter 35—not the width of one room."
        ),
    )

    if calibration_mode != "No Overall Dimension Available":
        st.caption(
            "💡 Important: a room label like 11 ft × 15 ft is not the overall building width. "
            "Use the full exterior dimension of the floor plan."
        )

    if blueprint:

        image = Image.open(blueprint).convert("RGB")

        original, gray, edges, closed_edges, adaptive, combined = (
            preprocess_blueprint(image)
        )

        bounds = detect_floorplan_bounds(combined)

        st.subheader("1️⃣ Blueprint Analysis")

        st.image(
            image,
            caption="Uploaded Blueprint",
            use_container_width=True,
        )

        if bounds is None:

            st.error(
                "Could not detect a reliable central floor-plan "
                "region. Try a clearer blueprint with visible wall lines."
            )

        else:

            cropped = crop_blueprint_region(
                original,
                bounds,
                padding=10,
            )

            horizontal, vertical, line_mask, line_pixels = (
                detect_architectural_lines(edges)
            )

            cropped_pil = Image.fromarray(cropped)
            cropped_gray = cv2.cvtColor(
                np.asarray(cropped_pil),
                cv2.COLOR_RGB2GRAY,
            )
            wall_segments, wall_overlay = detect_wall_segments(
                cropped_gray
            )

            wall_mask_v273 = build_architectural_wall_mask(
                cropped_gray,
                wall_segments,
                wall_px=5,
            )

            refined_wall_segments, wall_centerline, wall_mask = reconstruct_wall_network(
                cropped_gray, wall_segments
            )
            wall_segments = refined_wall_segments

            st.success(
                "✅ Primary blueprint region detected."
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Plan Width",
                    f"{bounds['width_px']:,} px",
                )

            with c2:
                st.metric(
                    "Plan Height",
                    f"{bounds['height_px']:,} px",
                )

            with c3:
                st.metric(
                    "Boundary Score",
                    f"{bounds['score']:.2f}",
                )

            with c4:
                st.metric(
                    "Line Pixels",
                    f"{line_pixels:,}",
                )

            st.subheader("2️⃣ Computer Vision Diagnostics")

            d1, d2, d3, d4, d5 = st.columns(5)

            with d1:
                st.image(
                    cv2.cvtColor(
                        edges,
                        cv2.COLOR_GRAY2RGB,
                    ),
                    caption="Edge Map",
                    use_container_width=True,
                )

            with d2:
                st.image(
                    cv2.cvtColor(
                        line_mask,
                        cv2.COLOR_GRAY2RGB,
                    ),
                    caption="Long Architectural Line Candidates",
                    use_container_width=True,
                )

            with d3:
                st.image(
                    cropped,
                    caption="Detected Plan Region",
                    use_container_width=True,
                )

            with d4:
                st.image(
                    cv2.cvtColor(
                        wall_mask,
                        cv2.COLOR_GRAY2RGB,
                    ),
                    caption="V2.7.4 Architectural Wall Mask",
                    use_container_width=True,
                )

            with d5:
                st.image(
                    cv2.cvtColor(
                        wall_overlay,
                        cv2.COLOR_GRAY2RGB,
                    ),
                    caption=f"V2.7.4 Connected Wall Network ({len(wall_segments)})",
                    use_container_width=True,
                )

            if wall_segments:
                st.success(
                    f"🧱 V2.7.4 reconstructed {len(wall_segments)} connected structural wall segments from the V2.7.3 evidence. "
                    "The network is still geometric evidence; room/space detection is now introduced in V2.8; semantic labels come later."
                )
            else:
                st.warning(
                    "⚠️ No stable wall segments were detected. Try a clearer plan "
                    "or a blueprint with stronger architectural wall lines."
                )

            st.subheader("3️⃣ V2.8 Room / Space Detection")

            st.caption(
                "V2.8 detects enclosed geometric spaces from the wall network. "
                "It does not yet assign semantic names such as Bedroom, Kitchen, or Toilet."
            )

            rooms, room_mask, room_overlay, room_diag = detect_rooms_from_wall_network(
                cropped_gray,
                wall_centerline,
            )

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "Detected Spaces",
                    room_diag["room_count"],
                )

            with r2:
                st.metric(
                    "Wall Thickness Evidence",
                    f"{room_diag['wall_thickness_px']} px",
                )

            with r3:
                st.metric(
                    "Minimum Space Area",
                    f"{room_diag['min_area_px']:,} px²",
                )

            rd1, rd2, rd3 = st.columns(3)

            with rd1:
                st.image(
                    cv2.cvtColor(room_mask, cv2.COLOR_GRAY2RGB),
                    caption="V2.8 Enclosed Space Mask",
                    use_container_width=True,
                )

            with rd2:
                st.image(
                    room_overlay,
                    caption=f"V2.8 Space Candidates ({room_diag['room_count']})",
                    use_container_width=True,
                )

            with rd3:
                st.image(
                    cv2.cvtColor(wall_centerline, cv2.COLOR_GRAY2RGB),
                    caption="Input Wall Centerline Network",
                    use_container_width=True,
                )

            if rooms:
                st.success(
                    f"🧱 V2.8 identified {len(rooms)} enclosed geometric spaces from the wall network. "
                    "These are room/space candidates, not yet semantic room labels."
                )

                st.session_state["detected_rooms"] = rooms
                render_room_table(rooms)
            else:
                st.warning(
                    "⚠️ No reliable enclosed spaces were detected. The wall network needs further refinement before semantic room analysis."
                )

            st.subheader("4️⃣ Calibrated Scale Estimate")

            width_px = float(bounds["width_px"])
            height_px = float(bounds["height_px"])
            aspect_ratio = safe_div(height_px, width_px)

            if calibration_mode == "Overall Building Width":
                calibrated_width = float(reference_dimension)
                calibrated_depth = height_px * safe_div(calibrated_width, width_px)
                scale_basis = "overall width"

            elif calibration_mode == "Overall Building Depth":
                calibrated_depth = float(reference_dimension)
                calibrated_width = width_px * safe_div(calibrated_depth, height_px)
                scale_basis = "overall depth"

            else:
                calibrated_width = width_px
                calibrated_depth = height_px
                scale_basis = "pixel geometry (no real-world scale)"

            calibrated_footprint = calibrated_width * calibrated_depth

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Calibrated Width",
                    f"{calibrated_width:.1f} ft" if calibration_mode != "No Overall Dimension Available" else "Unscaled",
                )

            with c2:
                st.metric(
                    "Calibrated Depth",
                    f"{calibrated_depth:.1f} ft" if calibration_mode != "No Overall Dimension Available" else "Unscaled",
                )

            with c3:
                st.metric(
                    "Plan Aspect Ratio",
                    f"1 : {aspect_ratio:.2f}",
                )

            with c4:
                st.metric(
                    "Estimated Footprint",
                    f"{calibrated_footprint:,.0f} sqft" if calibration_mode != "No Overall Dimension Available" else "Unavailable",
                )

            if calibration_mode == "No Overall Dimension Available":
                st.warning(
                    "⚠️ No real-world dimension was supplied. ArchMind can analyze the "
                    "pixel geometry, but it will not claim a real-world footprint or "
                    "use this unscaled geometry for material estimation."
                )
            else:
                st.success(
                    f"✅ Scale calibrated using the {scale_basis}. "
                    "The reference dimension is applied to the complete detected plan boundary."
                )

            st.caption(
                "Automatic dimension/OCR extraction is still a future reconstruction stage. "
                "For now, use an actual overall exterior dimension when available."
            )

            if rooms and calibration_mode != "No Overall Dimension Available":
                px_to_ft_x = safe_div(calibrated_width, width_px)
                px_to_ft_y = safe_div(calibrated_depth, height_px)
                for room in rooms:
                    room["area_sqft"] = room["area_px"] * px_to_ft_x * px_to_ft_y
                    room["width_ft"] = room["width_px"] * px_to_ft_x
                    room["depth_ft"] = room["height_px"] * px_to_ft_y

                st.subheader("📐 V2.8 Calibrated Space Estimates")
                st.caption(
                    "Areas are pixel-derived from the calibrated overall plan dimensions. "
                    "They are geometric estimates until automatic dimension/OCR extraction is added."
                )
                render_room_table(rooms)

            st.subheader("5️⃣ V2.7 Wall-Network 3D Reconstruction")

            st.caption(
                "V2.7.4 converts filtered wall evidence into a consolidated connected wall network and then into 3D wall meshes. "
                "This upload is reconstructed as one floor only; it is not duplicated across the building."
            )

            if st.button(
                "🏗️ Reconstruct 3D Wall Network",
                type="primary",
            ):

                if calibration_mode == "No Overall Dimension Available":
                    st.error(
                        "❌ Provide an overall building width or depth before generating "
                        "a real-world 3D reconstruction."
                    )
                    st.stop()

                reconstructed_segments = [dict(seg) for seg in wall_segments]
                generated_mesh = create_wall_reconstruction_mesh(
                    reconstructed_segments,
                    calibrated_width,
                    calibrated_depth,
                    floor_number_input,
                    floor_height,
                    wall_thickness,
                )

                if generated_mesh is None:
                    st.error(
                        "❌ Wall reconstruction failed because no usable wall segments "
                        "were detected. The rectangular baseline is still available below."
                    )
                else:
                    st.session_state["generated_mesh"] = generated_mesh
                    st.success(
                        f"✅ V2.7.4 single-floor wall-network geometry generated from {len(reconstructed_segments)} "
                        f"candidate segments for Floor {int(floor_number_input)}."
                    )

            if "generated_mesh" in st.session_state:

                generated_mesh = (
                    st.session_state[
                        "generated_mesh"
                    ]
                )

                st.warning(
                    "V2.7 is an intermediate wall-network reconstruction. "
                    "It uses detected architectural line candidates; room semantics, "
                    "doors/windows, and exact wall topology are the next stage."
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Generated Width",
                        f"{generated_mesh.extents[0]:.1f} ft",
                    )

                with c2:
                    st.metric(
                        "Generated Depth",
                        f"{generated_mesh.extents[1]:.1f} ft",
                    )

                with c3:
                    st.metric(
                        "Generated Height",
                        f"{generated_mesh.extents[2]:.1f} ft",
                    )

                stl_bytes = export_mesh_stl(
                    generated_mesh
                )

                st.download_button(
                    "⬇️ Download Generated STL",
                    data=stl_bytes,
                    file_name="archmind_generated_building.stl",
                    mime="model/stl",
                )


# PAGE: STL MATERIAL PREDICTION
# ============================================================

elif page == "STL Material Prediction":

    st.header(
        "🧱 STL Material Prediction"
    )

    if uploaded_stl is None:

        st.info(
            "Upload an STL building model "
            "from the sidebar."
        )

    else:

        try:

            mesh = load_stl_mesh(
                uploaded_stl
            )

            geometry = analyze_geometry(
                mesh,
                num_floors_input,
                floor_height,
            )

            show_building_classification(
                geometry
            )

            st.divider()

            # ------------------------------------------------
            # Floor estimation
            # ------------------------------------------------

            estimated = geometry[
                "estimated_floors"
            ]

            selected = (
                num_floors_input
            )

            st.subheader(
                "🏢 Automatic Floor Estimation"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Model Height",
                    f"{geometry['height_ft']:.1f} ft",
                )

            with c2:

                st.metric(
                    "Selected Floors",
                    selected,
                )

            with c3:

                st.metric(
                    "Detected Floors",
                    estimated,
                )

            with c4:

                st.metric(
                    "Height Ratio",
                    f"{geometry['height_ratio']:.2f}×",
                )

            floor_ok = (
                handle_floor_confirmation(
                    geometry,
                    selected,
                )
            )

            st.divider()

            show_geometry_metrics(
                geometry
            )

            st.divider()

            # ------------------------------------------------
            # Final confirmed floors
            # ------------------------------------------------

            confirmed_floors = (
                st.session_state.get(
                    "confirmed_floors"
                )
            )

            if confirmed_floors is None:

                confirmed_floors = selected

            # ------------------------------------------------
            # Final height validation
            # ------------------------------------------------

            final_expected_height = (
                confirmed_floors *
                floor_height
            )

            final_ratio = safe_div(
                geometry["height_ft"],
                final_expected_height,
            )

            final_height_valid = (
                0.60 <=
                final_ratio <=
                1.80
            )

            building_ok = (
                geometry[
                    "classification"
                ]
                ==
                "Likely Building"
            )

            uncertain_ok = (
                geometry[
                    "classification"
                ]
                ==
                "Uncertain Structure"
            )

            # ------------------------------------------------
            # Prediction gate
            # ------------------------------------------------

            prediction_ready = (
                building_ok
                and
                final_height_valid
            )

            if building_ok:

                st.success(
                    "🏢 Building geometry accepted."
                )

            elif uncertain_ok:

                st.warning(
                    "⚠️ Geometry is uncertain. "
                    "Material prediction is paused "
                    "until the structure is manually "
                    "verified."
                )

            else:

                st.error(
                    "🚫 Geometry classified as "
                    "likely non-building."
                )

            if not final_height_valid:

                st.warning(
                    f"⚠️ Confirmed floor configuration "
                    f"expects approximately "
                    f"**{final_expected_height:.1f} ft**, "
                    f"while the model is "
                    f"**{geometry['height_ft']:.1f} ft**."
                )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            if prediction_ready:

                st.subheader(
                    "🤖 Material Intelligence"
                )

                st.info(
                    f"Using **{confirmed_floors} floors** "
                    "for material estimation."
                )

                feature_df = (
                    build_prediction_features(
                        geometry,
                        confirmed_floors,
                        floor_height,
                        foundation_depth,
                        wall_thickness,
                        soil_condition,
                        masonry_type,
                        structural_intensity,
                    )
                )

                with st.expander(
                    "🔎 ML Feature Preview"
                ):

                    st.dataframe(
                        feature_df,
                        use_container_width=True,
                    )

                if st.button(
                    "🚀 Run V2.5.1 Material Prediction",
                    type="primary",
                ):

                    if model_errors:

                        st.warning(
                            "Some ML models could not "
                            "be loaded."
                        )

                        for error in model_errors:

                            st.write(
                                error
                            )

                    else:

                        predictions = (
                            predict_materials(
                                feature_df
                            )
                        )

                        st.subheader(
                            "📊 Predicted Materials"
                        )

                        c1, c2 = st.columns(2)

                        with c1:

                            cement = predictions[
                                "cement_bags"
                            ]

                            steel = predictions[
                                "steel_tonnes"
                            ]

                            st.metric(
                                "Cement",
                                (
                                    f"{cement:,.0f} bags"
                                    if cement is not None
                                    else "Unavailable"
                                ),
                            )

                            st.metric(
                                "Steel",
                                (
                                    f"{steel:,.2f} tonnes"
                                    if steel is not None
                                    else "Unavailable"
                                ),
                            )

                        with c2:

                            bricks = predictions[
                                "brick_count"
                            ]

                            aac = predictions[
                                "aac_block_count"
                            ]

                            st.metric(
                                "Bricks",
                                (
                                    f"{bricks:,.0f}"
                                    if bricks is not None
                                    else "Unavailable"
                                ),
                            )

                            st.metric(
                                "AAC Blocks",
                                (
                                    f"{aac:,.0f}"
                                    if aac is not None
                                    else "Unavailable"
                                ),
                            )

                        st.info(
                            "These are predictions from "
                            "the ArchMind V2.2 synthetic "
                            "engineering-informed dataset "
                            "and are intended for ML "
                            "prototyping, not professional "
                            "structural estimation."
                        )

            else:

                st.info(
                    "🔒 Material prediction is currently "
                    "paused until building classification "
                    "and geometry configuration are valid."
                )

        except Exception as e:

            st.error(
                f"STL processing failed: {e}"
            )


# ============================================================
# PAGE: GEOMETRY ANALYSIS
# ============================================================

elif page == "Geometry Analysis":

    st.header(
        "📐 Geometry Intelligence"
    )

    if uploaded_stl is None:

        st.info(
            "Upload an STL model from the sidebar."
        )

    else:

        try:

            mesh = load_stl_mesh(
                uploaded_stl
            )

            geometry = analyze_geometry(
                mesh,
                num_floors_input,
                floor_height,
            )

            show_building_classification(
                geometry
            )

            st.divider()

            show_geometry_metrics(
                geometry
            )

            st.divider()

            st.subheader(
                "🏢 Floor Estimation"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Model Height",
                    f"{geometry['height_ft']:.1f} ft",
                )

            with c2:

                st.metric(
                    "Selected Floors",
                    num_floors_input,
                )

            with c3:

                st.metric(
                    "Detected Floors",
                    geometry[
                        "estimated_floors"
                    ],
                )

            if (
                geometry[
                    "estimated_floors"
                ]
                !=
                num_floors_input
            ):

                st.warning(
                    "⚠️ The geometry and selected "
                    "floor configuration do not match."
                )

            else:

                st.success(
                    "✅ Floor configuration is "
                    "consistent with geometry."
                )

        except Exception as e:

            st.error(
                f"Geometry analysis failed: {e}"
            )


# ============================================================
# PAGE: FEATURE PREVIEW
# ============================================================

elif page == "Feature Preview":

    st.header(
        "🔎 ML Feature Preview"
    )

    st.write(
        "Features currently supplied to the "
        "material prediction models."
    )

    if uploaded_stl is None:

        st.info(
            "Upload an STL model."
        )

    else:

        try:

            mesh = load_stl_mesh(
                uploaded_stl
            )

            geometry = analyze_geometry(
                mesh,
                num_floors_input,
                floor_height,
            )

            confirmed_floors = (
                st.session_state.get(
                    "confirmed_floors"
                )
            )

            if confirmed_floors is None:
                confirmed_floors = num_floors_input

            feature_df = (
                build_prediction_features(
                    geometry,
                    confirmed_floors,
                    floor_height,
                    foundation_depth,
                    wall_thickness,
                    soil_condition,
                    masonry_type,
                    structural_intensity,
                )
            )

            st.dataframe(
                feature_df.T.rename(
                    columns={
                        0: "Value"
                    }
                ),
                use_container_width=True,
            )

        except Exception as e:

            st.error(
                f"Feature generation failed: {e}"
            )


# ============================================================
# PAGE: ABOUT
# ============================================================

elif page == "About":

    st.header(
        "ℹ️ About ArchMind Pro"
    )

    st.markdown(
        """
## ArchMind Pro V2.6.0

ArchMind Pro is an AI-powered construction
material intelligence prototype.

### Current Pipeline

**Blueprint**
→ Computer Vision
→ Structural Footprint
→ Procedural 3D Geometry
→ Geometry Intelligence
→ Building Classification
→ Automatic Floor Estimation
→ User Confirmation
→ Custom ML Models
→ Material Prediction

### V2.6.0 Improvements

- Automatic floor estimation
- Building plausibility classification
- Vertical geometry profile analysis
- Taper detection
- Floor configuration confirmation
- Geometry validation
- STL processing
- V2.8 geometric room/space detection
[L26 - Step 1] Multi-stage blueprint preprocessing and architectural line diagnostics
- Existing material prediction models
- Hugging Face model loading

### Important Design Principle

Automatic floor estimation is treated as
a **geometric hypothesis**, not proof that
the object is a building.

Building classification considers:

- footprint
- volume density
- disconnected components
- surface-to-volume ratio
- vertical footprint consistency
- top-to-bottom footprint ratio
- geometric taper
- height-to-footprint relationship

### Limitations

This is an AI/geometry prototype.

It does not provide professional structural
engineering certification or final construction
material quantities.

The material prediction models were trained
using a synthetic engineering-informed dataset
for ML prototyping.

### Technology

Python  
Streamlit  
OpenCV  
Trimesh  
NumPy  
Pandas  
scikit-learn  
XGBoost  
Joblib  
Hugging Face Hub

### Roadmap

**V2.6.0**
Intelligent blueprint preprocessing + building geometry validation

**V3**
Better structural feature extraction

**V3.x**
Algorithm benchmarking

**Future**
Safety monitoring, manpower planning,
progress analysis and explainable AI.
"""
    )
