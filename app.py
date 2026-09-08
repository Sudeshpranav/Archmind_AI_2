import streamlit as st
import pandas as pd
import numpy as np
import trimesh
import joblib
import math

from huggingface_hub import hf_hub_download


# ============================================================
# ARCHMIND PRO V2.4
# Intelligent Construction Material Intelligence
# ============================================================

st.set_page_config(
    page_title="ArchMind Pro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO_ID = "AloneMrY/archmind-pro-v2-2-models"

TARGET_FILES = {
    "cement_bags": "cement_bags_model.joblib",
    "steel_tonnes": "steel_tonnes_model.joblib",
    "brick_count": "brick_count_model.joblib",
    "aac_block_count": "aac_block_count_model.joblib"
}


# Features expected by the V2.2 trained models
FEATURES = [
    "building_type",
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
    "slab_thickness_in",
    "concrete_volume_m3",
    "soil_condition",
    "soil_bearing_capacity_kpa",
    "structural_intensity",
    "masonry_type"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 20px;
    }

    .status-box {
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏗️ ArchMind Pro</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Construction Material Intelligence</div>',
    unsafe_allow_html=True
)

st.caption(
    "V2.4 — Intelligent building geometry extraction and "
    "ML-assisted material prediction"
)

st.markdown(
    '<div class="status-box">🟢 System Online</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models_from_huggingface():

    models = {}
    errors = []

    for target, filename in TARGET_FILES.items():

        try:

            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename
            )

            models[target] = joblib.load(model_path)

        except Exception as e:

            errors.append(
                f"{filename}: {str(e)}"
            )

    return models, errors


models, model_errors = load_models_from_huggingface()


# ============================================================
# STL LOADER
# ============================================================

def load_stl_mesh(uploaded_file):

    """
    Safely loads a Streamlit UploadedFile as an STL mesh.

    Explicit file_type='stl' prevents trimesh from trying
    to infer the file type from the Streamlit file object.
    """

    # Reset file pointer
    uploaded_file.seek(0)

    mesh = trimesh.load(
        uploaded_file,
        file_type="stl",
        force="mesh"
    )

    # Handle Scene objects just in case
    if isinstance(mesh, trimesh.Scene):

        if len(mesh.geometry) == 0:

            raise ValueError(
                "The uploaded STL contains no usable geometry."
            )

        mesh = trimesh.util.concatenate(
            tuple(mesh.geometry.values())
        )

    # Final validation
    if mesh is None:

        raise ValueError(
            "No mesh could be extracted from the STL."
        )

    if len(mesh.vertices) == 0:

        raise ValueError(
            "The STL contains no vertices."
        )

    if len(mesh.faces) == 0:

        raise ValueError(
            "The STL contains no faces."
        )

    return mesh


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Project Configuration")

page = st.sidebar.radio(
    "Navigation",
    [
        "Material Prediction",
        "Geometry Analysis",
        "Feature Preview",
        "About"
    ]
)


st.sidebar.markdown("---")


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
        "Institutional"
    ]
)


foundation_depth = st.sidebar.number_input(
    "Foundation Depth (ft)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)


wall_thickness = st.sidebar.number_input(
    "Wall Thickness (in)",
    min_value=4.0,
    max_value=18.0,
    value=9.0,
    step=0.5
)


num_floors_input = st.sidebar.number_input(
    "Number of Floors",
    min_value=1,
    max_value=100,
    value=1,
    step=1
)


floor_height = st.sidebar.number_input(
    "Floor Height (ft)",
    min_value=7.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)


soil_condition = st.sidebar.selectbox(
    "Soil Condition",
    [
        "good",
        "normal",
        "weak"
    ]
)


masonry_type = st.sidebar.selectbox(
    "Masonry Type",
    [
        "brick",
        "aac",
        "mixed"
    ]
)


structural_intensity = st.sidebar.selectbox(
    "Structural Intensity",
    [
        "low",
        "medium",
        "high"
    ]
)


st.sidebar.markdown("---")


uploaded_file = st.sidebar.file_uploader(
    "Upload STL Structural File",
    type=["stl"]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def soil_bearing_capacity(soil):

    mapping = {
        "good": 300,
        "normal": 200,
        "weak": 100
    }

    return mapping.get(
        soil,
        200
    )


# ============================================================
# GEOMETRY ANALYSIS
# ============================================================

def analyze_mesh(mesh):

    result = {}

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    dims = np.array(
        mesh.extents,
        dtype=float
    )

    dims_ft = dims

    dims_sorted = np.sort(
        dims_ft
    )

    width = float(
        dims_sorted[0]
    )

    depth = float(
        dims_sorted[1]
    )

    height = float(
        dims_sorted[2]
    )

    footprint_area = (
        width *
        depth
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    try:

        volume = abs(
            float(mesh.volume)
        )

    except Exception:

        volume = 0.0

    # --------------------------------------------------------
    # Surface area
    # --------------------------------------------------------

    try:

        surface_area = float(
            mesh.area
        )

    except Exception:

        surface_area = 0.0

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bounding_volume = (
        width *
        depth *
        height
    )

    if bounding_volume > 0:

        volume_fill_ratio = (
            volume /
            bounding_volume
        )

    else:

        volume_fill_ratio = 0.0

    # --------------------------------------------------------
    # Expected building height
    # --------------------------------------------------------

    expected_height = (
        float(num_floors_input) *
        float(floor_height)
    )

    if expected_height > 0:

        height_ratio = (
            height /
            expected_height
        )

    else:

        height_ratio = 0.0

    # --------------------------------------------------------
    # Height validation
    # --------------------------------------------------------

    height_valid = (
        0.60 <= height_ratio <= 1.80
    )

    # --------------------------------------------------------
    # Height / footprint ratio
    # --------------------------------------------------------

    smallest_footprint_dimension = min(
        width,
        depth
    )

    if smallest_footprint_dimension > 0:

        height_to_footprint_ratio = (
            height /
            smallest_footprint_dimension
        )

    else:

        height_to_footprint_ratio = 999.0

    # --------------------------------------------------------
    # Surface / volume ratio
    # --------------------------------------------------------

    if volume > 0:

        surface_volume_ratio = (
            surface_area /
            volume
        )

    else:

        surface_volume_ratio = 0.0

    # --------------------------------------------------------
    # Watertight
    # --------------------------------------------------------

    try:

        is_watertight = bool(
            mesh.is_watertight
        )

    except Exception:

        is_watertight = False

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    try:

        components = mesh.split(
            only_watertight=False
        )

        component_count = len(
            components
        )

    except Exception:

        component_count = 1

    # --------------------------------------------------------
    # Mesh complexity
    # --------------------------------------------------------

    vertex_count = len(
        mesh.vertices
    )

    face_count = len(
        mesh.faces
    )

    if footprint_area > 0:

        geometry_density = (
            face_count /
            footprint_area
        )

    else:

        geometry_density = 0.0

    # --------------------------------------------------------
    # Vertical distribution
    # --------------------------------------------------------

    lower_ratio = 0.0
    middle_ratio = 0.0
    upper_ratio = 0.0

    try:

        z_values = mesh.vertices[:, 2]

        z_min = float(
            np.min(z_values)
        )

        z_max = float(
            np.max(z_values)
        )

        z_range = (
            z_max -
            z_min
        )

        if z_range > 0:

            normalized_z = (
                z_values -
                z_min
            ) / z_range

            lower_vertices = np.sum(
                normalized_z < 0.15
            )

            middle_vertices = np.sum(
                (
                    normalized_z >= 0.15
                )
                &
                (
                    normalized_z <= 0.85
                )
            )

            upper_vertices = np.sum(
                normalized_z > 0.85
            )

            total_vertices = max(
                len(z_values),
                1
            )

            lower_ratio = (
                lower_vertices /
                total_vertices
            )

            middle_ratio = (
                middle_vertices /
                total_vertices
            )

            upper_ratio = (
                upper_vertices /
                total_vertices
            )

    except Exception:

        pass

    # --------------------------------------------------------
    # Building confidence
    # --------------------------------------------------------

    score = 0.0

    reasons = []

    # Footprint
    if footprint_area >= 250:

        score += 15

    else:

        reasons.append(
            "Very small footprint"
        )

    # Height
    if height >= 8:

        score += 10

    else:

        reasons.append(
            "Very small model height"
        )

    # Height / footprint
    if (
        0.15 <=
        height_to_footprint_ratio
        <= 4.0
    ):

        score += 15

    elif (
        height_to_footprint_ratio
        <= 8.0
    ):

        score += 5

    else:

        reasons.append(
            "Extreme height-to-footprint ratio"
        )

    # Volume fill
    if (
        0.05 <=
        volume_fill_ratio
        <= 0.90
    ):

        score += 15

    elif volume_fill_ratio > 0.90:

        score += 5

    else:

        reasons.append(
            "Very sparse geometry"
        )

    # Watertight
    if is_watertight:

        score += 15

    else:

        reasons.append(
            "Mesh is not watertight"
        )

    # Components
    if component_count <= 20:

        score += 10

    else:

        reasons.append(
            "Very high number of disconnected components"
        )

    # Mesh complexity
    if face_count >= 100:

        score += 10

    else:

        reasons.append(
            "Very low mesh complexity"
        )

    # Height consistency
    if height_valid:

        score += 10

    else:

        reasons.append(
            "Model height does not match selected floor configuration"
        )

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    building_confidence = max(
        0,
        min(
            100,
            score
        )
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if building_confidence >= 75:

        classification = (
            "Likely Building"
        )

    elif building_confidence >= 50:

        classification = (
            "Uncertain Structure"
        )

    else:

        classification = (
            "Likely Non-Building Object"
        )

    # --------------------------------------------------------
    # Extreme geometry
    # --------------------------------------------------------

    extreme_geometry = (
        height_to_footprint_ratio >
        8.0
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result = {

        "width_ft":
            width,

        "depth_ft":
            depth,

        "height_ft":
            height,

        "footprint_area_sqft":
            footprint_area,

        "volume":
            volume,

        "surface_area":
            surface_area,

        "bounding_volume":
            bounding_volume,

        "volume_fill_ratio":
            volume_fill_ratio,

        "expected_height":
            expected_height,

        "height_ratio":
            height_ratio,

        "height_valid":
            height_valid,

        "height_to_footprint_ratio":
            height_to_footprint_ratio,

        "surface_volume_ratio":
            surface_volume_ratio,

        "is_watertight":
            is_watertight,

        "component_count":
            component_count,

        "vertex_count":
            vertex_count,

        "face_count":
            face_count,

        "geometry_density":
            geometry_density,

        "lower_vertex_ratio":
            lower_ratio,

        "middle_vertex_ratio":
            middle_ratio,

        "upper_vertex_ratio":
            upper_ratio,

        "building_confidence":
            building_confidence,

        "classification":
            classification,

        "extreme_geometry":
            extreme_geometry,

        "reasons":
            reasons
    }

    return result


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def estimate_features(geometry):

    width = geometry[
        "width_ft"
    ]

    depth = geometry[
        "depth_ft"
    ]

    floor_area = (
        width *
        depth
    )

    # --------------------------------------------------------
    # Floors
    # --------------------------------------------------------

    num_floors = int(
        num_floors_input
    )

    num_floors = max(
        1,
        num_floors
    )

    # --------------------------------------------------------
    # Built-up area
    # --------------------------------------------------------

    built_up_area = (
        floor_area *
        num_floors *
        1.05
    )

    # --------------------------------------------------------
    # Wall length
    # --------------------------------------------------------

    perimeter = (
        2 *
        (
            width +
            depth
        )
    )

    internal_wall_length = (
        math.sqrt(
            max(
                floor_area,
                1
            )
        )
        *
        0.35
    )

    wall_length = (
        perimeter +
        internal_wall_length
    ) * num_floors

    # --------------------------------------------------------
    # Wall area
    # --------------------------------------------------------

    gross_wall_area = (
        wall_length *
        floor_height
    )

    # --------------------------------------------------------
    # Doors
    # --------------------------------------------------------

    door_area = (
        gross_wall_area *
        0.035
    )

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    window_area = (
        gross_wall_area *
        0.10
    )

    # --------------------------------------------------------
    # Opening constraint
    # --------------------------------------------------------

    max_opening_area = (
        gross_wall_area *
        0.30
    )

    total_openings = (
        door_area +
        window_area
    )

    if (
        total_openings >
        max_opening_area
    ):

        scale = (
            max_opening_area /
            max(
                total_openings,
                1
            )
        )

        door_area *= scale
        window_area *= scale

    # --------------------------------------------------------
    # Foundation
    # --------------------------------------------------------

    foundation_area = (
        floor_area *
        1.05
    )

    # --------------------------------------------------------
    # Slab
    # --------------------------------------------------------

    slab_area = (
        floor_area *
        num_floors
    )

    slab_thickness = 5.0

    # --------------------------------------------------------
    # Roof
    # --------------------------------------------------------

    roof_area = (
        floor_area *
        1.05
    )

    # --------------------------------------------------------
    # Concrete volume
    # --------------------------------------------------------

    slab_volume = (
        slab_area *
        (
            slab_thickness /
            12
        )
    )

    foundation_volume = (
        foundation_area *
        (
            foundation_depth /
            12
        )
    )

    frame_volume = (
        built_up_area *
        0.015
    )

    concrete_volume_ft3 = (
        slab_volume +
        foundation_volume +
        frame_volume
    )

    concrete_volume_m3 = (
        concrete_volume_ft3 *
        0.0283168
    )

    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    features = {

        "building_type":
            building_type,

        "floor_area_sqft":
            floor_area,

        "built_up_area_sqft":
            built_up_area,

        "num_floors":
            num_floors,

        "num_rooms":
            max(
                1,
                int(
                    floor_area /
                    250
                )
            ),

        "wall_length_ft":
            wall_length,

        "wall_thickness_in":
            wall_thickness,

        "wall_height_ft":
            floor_height,

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

        "slab_thickness_in":
            slab_thickness,

        "concrete_volume_m3":
            concrete_volume_m3,

        "soil_condition":
            soil_condition,

        "soil_bearing_capacity_kpa":
            soil_bearing_capacity(
                soil_condition
            ),

        "structural_intensity":
            structural_intensity,

        "masonry_type":
            masonry_type
    }

    return features


# ============================================================
# PREDICTION
# ============================================================

def predict_materials(features):

    # Explicit feature order
    df = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    predictions = {}

    for target, model in models.items():

        try:

            prediction = model.predict(
                df
            )[0]

            predictions[target] = max(
                0.0,
                float(prediction)
            )

        except Exception as e:

            predictions[target] = None

            st.error(
                f"Prediction failed for "
                f"{target}: {e}"
            )

    return predictions


# ============================================================
# MATERIAL PREDICTION PAGE
# ============================================================

if page == "Material Prediction":

    st.header(
        "📐 Material Prediction"
    )

    st.write(
        "Upload a structural STL model and ArchMind will "
        "extract geometry, validate building plausibility, "
        "generate ML features, and estimate construction materials."
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload an STL file from the sidebar to begin."
        )

        st.markdown(
            """
            ### V2.4 Intelligence Pipeline

            **STL**
            ↓

            **Geometry Extraction**
            ↓

            **Building Plausibility Analysis**
            ↓

            **Structural Validation**
            ↓

            **Feature Engineering**
            ↓

            **Custom ML Models**
            ↓

            **Material Prediction**
            """
        )

    else:

        # ----------------------------------------------------
        # Load mesh
        # ----------------------------------------------------

        try:

            mesh = load_stl_mesh(
                uploaded_file
            )

        except Exception as e:

            st.error(
                f"Unable to load STL file: {e}"
            )

            st.stop()

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        geometry = analyze_mesh(
            mesh
        )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        features = estimate_features(
            geometry
        )

        # ----------------------------------------------------
        # Geometry cards
        # ----------------------------------------------------

        st.subheader(
            "🔍 Geometry Analysis"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Footprint",
            f"{geometry['footprint_area_sqft']:,.0f} sqft"
        )

        col2.metric(
            "Floors",
            f"{num_floors_input}"
        )

        col3.metric(
            "Model Height",
            f"{geometry['height_ft']:,.1f} ft"
        )

        col4.metric(
            "Building Confidence",
            f"{geometry['building_confidence']:.0f}%"
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        classification = (
            geometry["classification"]
        )

        if classification == "Likely Building":

            st.success(
                f"🏢 Geometry classification: "
                f"**{classification}**"
            )

        elif classification == "Uncertain Structure":

            st.warning(
                f"⚠️ Geometry classification: "
                f"**{classification}**"
            )

        else:

            st.error(
                f"🚫 Geometry classification: "
                f"**{classification}**"
            )

        # ----------------------------------------------------
        # Detailed geometry
        # ----------------------------------------------------

        with st.expander(
            "🔎 View extracted geometry details"
        ):

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:

                st.write(
                    f"**Width:** "
                    f"{geometry['width_ft']:.2f} ft"
                )

                st.write(
                    f"**Depth:** "
                    f"{geometry['depth_ft']:.2f} ft"
                )

                st.write(
                    f"**Height:** "
                    f"{geometry['height_ft']:.2f} ft"
                )

                st.write(
                    f"**Volume:** "
                    f"{geometry['volume']:,.2f}"
                )

                st.write(
                    f"**Surface Area:** "
                    f"{geometry['surface_area']:,.2f}"
                )

                st.write(
                    f"**Bounding Volume:** "
                    f"{geometry['bounding_volume']:,.2f}"
                )

            with detail_col2:

                st.write(
                    f"**Volume Fill Ratio:** "
                    f"{geometry['volume_fill_ratio']:.3f}"
                )

                st.write(
                    f"**Height / Footprint:** "
                    f"{geometry['height_to_footprint_ratio']:.3f}"
                )

                st.write(
                    f"**Surface / Volume:** "
                    f"{geometry['surface_volume_ratio']:.5f}"
                )

                st.write(
                    f"**Connected Components:** "
                    f"{geometry['component_count']}"
                )

                st.write(
                    f"**Vertices:** "
                    f"{geometry['vertex_count']:,}"
                )

                st.write(
                    f"**Faces:** "
                    f"{geometry['face_count']:,}"
                )

                st.write(
                    f"**Watertight:** "
                    f"{'Yes' if geometry['is_watertight'] else 'No'}"
                )

        # ----------------------------------------------------
        # Structural validation
        # ----------------------------------------------------

        st.subheader(
            "🏗️ Structural Geometry Validation"
        )

        expected_height = (
            geometry["expected_height"]
        )

        height_ratio = (
            geometry["height_ratio"]
        )

        h1, h2, h3 = st.columns(3)

        h1.metric(
            "Model Height",
            f"{geometry['height_ft']:.1f} ft"
        )

        h2.metric(
            "Expected Height",
            f"{expected_height:.1f} ft"
        )

        h3.metric(
            "Height Ratio",
            f"{height_ratio:.2f}×"
        )

        if geometry["height_valid"]:

            st.success(
                "✅ Model height is reasonably consistent "
                "with the selected floor configuration."
            )

        else:

            st.warning(
                "⚠️ Model height does not closely match "
                "the selected floor count and floor height."
            )

        # ----------------------------------------------------
        # Geometry intelligence
        # ----------------------------------------------------

        st.subheader(
            "🧠 Geometry Intelligence"
        )

        g1, g2, g3, g4 = st.columns(4)

        g1.metric(
            "Volume Fill",
            f"{geometry['volume_fill_ratio']:.2f}"
        )

        g2.metric(
            "Components",
            geometry["component_count"]
        )

        g3.metric(
            "Watertight",
            "Yes"
            if geometry["is_watertight"]
            else "No"
        )

        g4.metric(
            "Faces",
            f"{geometry['face_count']:,}"
        )

        # ----------------------------------------------------
        # Validation observations
        # ----------------------------------------------------

        if geometry["reasons"]:

            with st.expander(
                "⚠️ Validation observations"
            ):

                for reason in geometry["reasons"]:

                    st.write(
                        f"• {reason}"
                    )

        # ----------------------------------------------------
        # Prediction gate
        # ----------------------------------------------------

        prediction_allowed = (
            geometry["height_valid"]
            and
            not geometry["extreme_geometry"]
            and
            geometry["building_confidence"] >= 50
        )

        st.markdown("---")

        if not prediction_allowed:

            st.error(
                """
                🚫 **ML prediction paused**

                ArchMind's geometry validation layer considers
                this model unreliable for material prediction.

                Please verify:

                - The STL represents a building structure
                - The floor count is correct
                - The floor height is correct
                - The STL dimensions are correctly scaled
                """
            )

        else:

            # ------------------------------------------------
            # Check models
            # ------------------------------------------------

            if len(models) < 4:

                st.error(
                    "Not all ML models could be loaded."
                )

                if model_errors:

                    with st.expander(
                        "Model loading errors"
                    ):

                        for error in model_errors:

                            st.write(
                                f"• {error}"
                            )

            else:

                if st.button(
                    "🚀 Predict Construction Materials",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Running ArchMind V2.4 intelligence pipeline..."
                    ):

                        predictions = predict_materials(
                            features
                        )

                    st.subheader(
                        "📊 Material Prediction"
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    cement = predictions.get(
                        "cement_bags"
                    )

                    steel = predictions.get(
                        "steel_tonnes"
                    )

                    bricks = predictions.get(
                        "brick_count"
                    )

                    aac = predictions.get(
                        "aac_block_count"
                    )

                    if cement is not None:

                        c1.metric(
                            "Cement",
                            f"{cement:,.0f} bags"
                        )

                    if steel is not None:

                        c2.metric(
                            "Steel",
                            f"{steel:,.2f} tonnes"
                        )

                    if bricks is not None:

                        c3.metric(
                            "Bricks",
                            f"{bricks:,.0f}"
                        )

                    if aac is not None:

                        c4.metric(
                            "AAC Blocks",
                            f"{aac:,.0f}"
                        )

                    st.success(
                        "✅ Material prediction completed."
                    )

                    st.caption(
                        "Predictions are generated using the "
                        "custom-trained ArchMind ML models."
                    )

                    st.warning(
                        """
                        ⚠️ Prototype disclaimer:
                        These predictions are based on a synthetic,
                        engineering-informed dataset created for ML
                        prototyping. They are not a substitute for
                        structural design, quantity surveying, or
                        professional engineering calculations.
                        """
                    )


# ============================================================
# GEOMETRY ANALYSIS PAGE
# ============================================================

elif page == "Geometry Analysis":

    st.header(
        "🧠 Intelligent Geometry Analysis"
    )

    st.write(
        "V2.4 analyzes the uploaded 3D geometry before "
        "allowing material prediction."
    )

    if uploaded_file is None:

        st.info(
            "Upload an STL file from the sidebar."
        )

    else:

        try:

            mesh = load_stl_mesh(
                uploaded_file
            )

            geometry = analyze_mesh(
                mesh
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            st.subheader(
                "Building Confidence"
            )

            confidence = int(
                geometry[
                    "building_confidence"
                ]
            )

            st.progress(
                confidence
            )

            st.write(
                f"**{confidence}% — "
                f"{geometry['classification']}**"
            )

            # ------------------------------------------------
            # Main metrics
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Width",
                f"{geometry['width_ft']:.1f} ft"
            )

            col2.metric(
                "Depth",
                f"{geometry['depth_ft']:.1f} ft"
            )

            col3.metric(
                "Height",
                f"{geometry['height_ft']:.1f} ft"
            )

            col4.metric(
                "Footprint",
                f"{geometry['footprint_area_sqft']:,.0f} sqft"
            )

            st.markdown("---")

            # ------------------------------------------------
            # Full table
            # ------------------------------------------------

            rows = {

                "Width (ft)":
                    geometry["width_ft"],

                "Depth (ft)":
                    geometry["depth_ft"],

                "Height (ft)":
                    geometry["height_ft"],

                "Footprint (sqft)":
                    geometry["footprint_area_sqft"],

                "Volume":
                    geometry["volume"],

                "Surface Area":
                    geometry["surface_area"],

                "Bounding Volume":
                    geometry["bounding_volume"],

                "Volume Fill Ratio":
                    geometry["volume_fill_ratio"],

                "Expected Height":
                    geometry["expected_height"],

                "Height Ratio":
                    geometry["height_ratio"],

                "Height / Footprint":
                    geometry["height_to_footprint_ratio"],

                "Surface / Volume":
                    geometry["surface_volume_ratio"],

                "Connected Components":
                    geometry["component_count"],

                "Vertices":
                    geometry["vertex_count"],

                "Faces":
                    geometry["face_count"],

                "Watertight":
                    geometry["is_watertight"],

                "Building Confidence":
                    geometry["building_confidence"],

                "Classification":
                    geometry["classification"]
            }

            geometry_df = pd.DataFrame(
                rows.items(),
                columns=[
                    "Property",
                    "Value"
                ]
            )

            st.dataframe(
                geometry_df,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Classification message
            # ------------------------------------------------

            if (
                geometry["classification"]
                ==
                "Likely Building"
            ):

                st.success(
                    "✅ The geometry passes the current "
                    "building plausibility checks."
                )

            elif (
                geometry["classification"]
                ==
                "Uncertain Structure"
            ):

                st.warning(
                    "⚠️ The geometry requires additional validation."
                )

            else:

                st.error(
                    "🚫 The geometry is unlikely to represent "
                    "a conventional building."
                )

        except Exception as e:

            st.error(
                f"Geometry analysis failed: {e}"
            )


# ============================================================
# FEATURE PREVIEW PAGE
# ============================================================

elif page == "Feature Preview":

    st.header(
        "🔬 ML Feature Preview"
    )

    st.write(
        "These are the features generated before being "
        "passed to the material prediction models."
    )

    if uploaded_file is None:

        st.info(
            "Upload an STL file to preview generated features."
        )

    else:

        try:

            mesh = load_stl_mesh(
                uploaded_file
            )

            geometry = analyze_mesh(
                mesh
            )

            features = estimate_features(
                geometry
            )

            feature_df = pd.DataFrame(
                {
                    "Feature":
                        list(features.keys()),

                    "Value":
                        list(features.values())
                }
            )

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                f"Feature extraction failed: {e}"
            )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.header(
        "🏗️ About ArchMind Pro"
    )

    st.markdown(
        """
        ## ArchMind Pro V2.4

        ArchMind Pro is an AI-powered construction material
        intelligence prototype.

        ### Core Technologies

        - 🧊 3D STL geometry processing
        - 🧠 Intelligent geometry validation
        - 🤖 Custom-trained machine learning models
        - 📊 Construction material prediction
        - ☁️ Streamlit deployment
        - 📦 Hugging Face model hosting

        ### V2.4 Intelligence Pipeline

        ```text
        STL Model
             ↓
        Geometry Extraction
             ↓
        Building Plausibility Analysis
             ↓
        Structural Geometry Validation
             ↓
        Feature Engineering
             ↓
        Custom ML Models
             ↓
        Material Prediction
        ```

        ### Current ML Outputs

        ArchMind currently estimates:

        - Cement bags
        - Steel tonnes
        - Brick count
        - AAC block count

        ### Important

        The current dataset is a synthetic,
        engineering-informed dataset created for ML prototyping.

        The system is not structural-design software and should
        not be used as a replacement for professional engineering,
        quantity surveying, or construction planning.

        ### Development Roadmap

        **V2.1** — Initial ML prototype

        **V2.2** — 50K-row material intelligence dataset and models

        **V2.3** — STL geometry validation

        **V2.4** — Intelligent building geometry extraction

        **V2.5** — Blueprint → 3D reconstruction

        **V3.0** — Improved ML feature generation and dataset

        **V3.x** — Algorithm competition and optimization

        **V4.0** — Full AI construction intelligence platform
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ArchMind Pro V2.4 • AI-powered construction material intelligence"
)
