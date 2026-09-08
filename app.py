import os
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import trimesh

from huggingface_hub import hf_hub_download


# ============================================================
# ARCHMIND PRO V2.2
# AI-POWERED CONSTRUCTION MATERIAL INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="ArchMind Pro V2.2",
    page_icon="🏗️",
    layout="wide"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        color: #FFFFFF !important;
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp {
        background-color: #0E1117;
    }

    section[data-testid="stSidebar"] {
        background-color: #161B22;
    }

    h1, h2, h3, h4 {
        color: #FFFFFF !important;
    }

    p, span, div, label {
        color: #E6EDF3 !important;
    }

    .stButton > button {
        font-weight: 700;
        border-radius: 10px;
    }

    .res-card {
        background-color: #1C2128;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363D;
        margin-bottom: 15px;
    }

    .status-card {
        background-color: #1C2128;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363D;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_REPO_ID = "AloneMrY/archmind-pro-v2-2-models"


TARGET_FILES = {
    "cement_bags": "cement_bags_model.joblib",
    "steel_tonnes": "steel_tonnes_model.joblib",
    "brick_count": "brick_count_model.joblib",
    "aac_block_count": "aac_block_count_model.joblib",
}


# ============================================================
# V2.2 MODEL FEATURES
# ============================================================

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
    "masonry_type",
]


# ============================================================
# SOIL PARAMETERS
# ============================================================

SOIL_BEARING = {
    "weak": 100.0,
    "normal": 200.0,
    "good": 300.0,
}


# ============================================================
# LOAD MODELS FROM HUGGING FACE
# ============================================================

@st.cache_resource(show_spinner=False)
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


# Load models
models, model_errors = load_models_from_huggingface()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏗️ ArchMind Pro")

    st.caption("V2.2 Material Intelligence")

    st.markdown("---")

    page = st.selectbox(
        "Navigation",
        [
            "Material Prediction",
            "Feature Preview",
            "About"
        ]
    )

    st.markdown("### Project Configuration")

    building_type = st.selectbox(
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
        ]
    )

    foundation_depth = st.slider(
        "Foundation Depth (ft)",
        2.0,
        15.0,
        5.0,
        0.1
    )

    wall_thickness = st.slider(
        "Wall Thickness (in)",
        4.0,
        12.0,
        6.0,
        0.5
    )

    num_floors = st.slider(
        "Number of Floors",
        1,
        12,
        1
    )

    floor_height = st.slider(
        "Floor Height (ft)",
        8.0,
        16.0,
        10.0,
        0.1
    )

    soil_condition = st.selectbox(
        "Soil Condition",
        [
            "weak",
            "normal",
            "good"
        ],
        index=1
    )

    masonry_type = st.selectbox(
        "Masonry Type",
        [
            "brick",
            "aac",
            "mixed"
        ],
        index=2
    )

    structural_intensity = st.slider(
        "Structural Intensity",
        0.70,
        1.50,
        1.00,
        0.01
    )

    st.markdown("---")

    st.caption(
        "Synthetic engineering-informed ML prototype. "
        "Outputs are planning estimates and not structural design certification."
    )


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def load_mesh(uploaded_file):

    temp_path = os.path.join(
        os.getcwd(),
        "_archmind_upload.stl"
    )

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:

        mesh = trimesh.load(
            temp_path,
            force="mesh"
        )

        if isinstance(mesh, trimesh.Scene):

            geometries = tuple(
                mesh.geometry.values()
            )

            if not geometries:
                raise ValueError(
                    "The uploaded STL scene contains no geometry."
                )

            mesh = trimesh.util.concatenate(
                geometries
            )

        return mesh

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# UNIT CONVERSION
# ============================================================

def convert_extents(extents, unit):

    factors = {

        "Feet": 1.0,

        "Inches": 1.0 / 12.0,

        "Meters": 3.280839895,

        "Millimeters": 0.003280839895,
    }

    if unit not in factors:
        raise ValueError(
            f"Unsupported unit: {unit}"
        )

    return (
        np.asarray(
            extents,
            dtype=float
        )
        * factors[unit]
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def estimate_features(mesh, unit):

    dims_ft = convert_extents(
        mesh.extents,
        unit
    )

    dims_ft = np.maximum(
        dims_ft,
        0.001
    )

    # --------------------------------------------------------
    # STL orientation is not guaranteed.
    # Use the two largest dimensions as footprint dimensions.
    # Number of floors is explicitly selected by the user.
    # --------------------------------------------------------

    sorted_dims = np.sort(dims_ft)

    footprint_x = sorted_dims[-1]

    footprint_y = sorted_dims[-2]

    floor_area = (
        footprint_x *
        footprint_y
    )

    # --------------------------------------------------------
    # Built-up area
    # --------------------------------------------------------

    built_up = (
        floor_area *
        num_floors *
        1.05
    )

    # --------------------------------------------------------
    # Building perimeter
    # --------------------------------------------------------

    perimeter = (
        2.0 *
        (
            footprint_x +
            footprint_y
        )
    )

    # --------------------------------------------------------
    # Estimate rooms
    # --------------------------------------------------------

    rooms_per_floor = max(
        1,
        round(
            (floor_area / 350.0) * 1.2
        )
    )

    num_rooms = max(
        1,
        rooms_per_floor * num_floors
    )

    # --------------------------------------------------------
    # Estimate wall length
    # --------------------------------------------------------

    wall_length = (
        perimeter * num_floors
        +
        (
            math.sqrt(
                max(
                    floor_area,
                    1.0
                )
            )
            * 0.35
            * num_floors
        )
    )

    wall_height = floor_height

    # --------------------------------------------------------
    # Openings
    # --------------------------------------------------------

    gross_wall_area = (
        wall_length *
        wall_height
    )

    door_area = (
        gross_wall_area *
        0.035
    )

    window_area = (
        gross_wall_area *
        0.12
    )

    # --------------------------------------------------------
    # Slab / roof
    # --------------------------------------------------------

    slab_area = (
        floor_area *
        num_floors
    )

    roof_area = floor_area

    # --------------------------------------------------------
    # Foundation
    # --------------------------------------------------------

    foundation_area = (
        floor_area *
        1.05
    )

    slab_thickness = 5.5

    # --------------------------------------------------------
    # Concrete estimation
    #
    # This is an engineering-informed approximation used
    # to generate model inputs.
    # --------------------------------------------------------

    slab_volume = (
        slab_area
        *
        (slab_thickness / 12.0)
        *
        0.0283168466
    )

    foundation_depth_m = (
        foundation_depth *
        0.3048
    )

    foundation_area_m2 = (
        foundation_area *
        0.092903
    )

    foundation_volume = (
        foundation_area_m2
        *
        foundation_depth_m
        *
        0.18
    )

    frame_volume = (
        built_up
        *
        0.092903
        *
        0.08
        *
        structural_intensity
    )

    concrete_volume = max(
        0.1,
        slab_volume
        +
        foundation_volume
        +
        frame_volume
    )

    # --------------------------------------------------------
    # Final V2.2 feature dictionary
    # --------------------------------------------------------

    return {

        "building_type": building_type,

        "floor_area_sqft": floor_area,

        "built_up_area_sqft": built_up,

        "num_floors": num_floors,

        "num_rooms": num_rooms,

        "wall_length_ft": wall_length,

        "wall_thickness_in": wall_thickness,

        "wall_height_ft": wall_height,

        "foundation_depth_ft": foundation_depth,

        "foundation_area_sqft": foundation_area,

        "door_area_sqft": door_area,

        "window_area_sqft": window_area,

        "slab_area_sqft": slab_area,

        "roof_area_sqft": roof_area,

        "slab_thickness_in": slab_thickness,

        "concrete_volume_m3": concrete_volume,

        "soil_condition": soil_condition,

        "soil_bearing_capacity_kpa":
            SOIL_BEARING[soil_condition],

        "structural_intensity":
            structural_intensity,

        "masonry_type":
            masonry_type,
    }


# ============================================================
# ABOUT PAGE
# ============================================================

if page == "About":

    st.title(
        "🏗️ ArchMind Pro V2.2"
    )

    st.write(
        "AI-powered construction material intelligence "
        "using 3D geometry, project parameters and "
        "custom-trained material prediction models."
    )

    st.markdown("---")

    st.subheader(
        "How ArchMind Works"
    )

    st.write(
        "1. User uploads a 3D STL structural model."
    )

    st.write(
        "2. ArchMind extracts geometry-related features."
    )

    st.write(
        "3. Project parameters are combined with the geometry."
    )

    st.write(
        "4. Custom-trained ML models estimate construction materials."
    )

    st.write(
        "5. Material predictions are presented as planning estimates."
    )

    st.markdown("---")

    st.subheader(
        "Material Prediction Targets"
    )

    st.write(
        "• Cement bags"
    )

    st.write(
        "• Steel rebar tonnes"
    )

    st.write(
        "• Brick count"
    )

    st.write(
        "• AAC block count"
    )

    st.markdown("---")

    st.info(
        "The V2.2 models were trained using a synthetic "
        "engineering-informed dataset. Predictions are "
        "intended for ML prototyping and material planning, "
        "not as a substitute for structural drawings, "
        "BOQs or licensed engineering decisions."
    )


# ============================================================
# MODEL ERROR HANDLING
# ============================================================

elif page in [
    "Material Prediction",
    "Feature Preview"
]:

    st.title(
        "Building Material Intelligence"
    )

    st.write(
        "Extracting structural features from 3D geometry."
    )

    # --------------------------------------------------------
    # Model loading status
    # --------------------------------------------------------

    if model_errors:

        st.error(
            "Some ArchMind V2.2 models could not be loaded "
            "from Hugging Face."
        )

        for error in model_errors:
            st.code(error)

        st.info(
            f"Expected Hugging Face repository: "
            f"{HF_REPO_ID}"
        )

        st.stop()

    # --------------------------------------------------------
    # All models successfully loaded
    # --------------------------------------------------------

    st.success(
        "✅ ArchMind V2.2 ML models loaded successfully."
    )

    # ========================================================
    # STL UPLOAD
    # ========================================================

    uploaded = st.file_uploader(
        "Upload STL Structural File",
        type=["stl"],
        help="Upload an STL model representing the structure."
    )

    # ========================================================
    # UNIT SELECTION
    # ========================================================

    unit = st.selectbox(
        "STL model unit",
        [
            "Feet",
            "Inches",
            "Meters",
            "Millimeters"
        ],
        help=(
            "STL files may not contain reliable unit metadata. "
            "Select the unit used when the model was exported."
        )
    )

    # ========================================================
    # PROCESS STL
    # ========================================================

    if uploaded:

        try:

            # ------------------------------------------------
            # Load mesh
            # ------------------------------------------------

            mesh = load_mesh(
                uploaded
            )

            if mesh.is_empty:

                st.error(
                    "The uploaded mesh is empty."
                )

                st.stop()

            # ------------------------------------------------
            # Extract features
            # ------------------------------------------------

            features = estimate_features(
                mesh,
                unit
            )

            dims = convert_extents(
                mesh.extents,
                unit
            )

            # ------------------------------------------------
            # Geometry information
            # ------------------------------------------------

            st.markdown(
                "### 📐 Geometry Analysis"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Footprint",
                f"{features['floor_area_sqft']:,.0f} sqft"
            )

            c2.metric(
                "Floors",
                f"{features['num_floors']}"
            )

            c3.metric(
                "Built-up Area",
                f"{features['built_up_area_sqft']:,.0f} sqft"
            )

            c4.metric(
                "Mesh Volume",
                f"{abs(mesh.volume):,.2f}"
            )

            st.caption(
                "Converted bounding dimensions: "
                f"{dims[0]:.2f} × "
                f"{dims[1]:.2f} × "
                f"{dims[2]:.2f} ft"
            )

            # =================================================
            # FEATURE PREVIEW
            # =================================================

            if page == "Feature Preview":

                st.subheader(
                    "Features Sent to V2.2 ML"
                )

                feature_df = (
                    pd.DataFrame(
                        [features]
                    )[FEATURES]
                    .T
                    .rename(
                        columns={
                            0: "value"
                        }
                    )
                )

                st.dataframe(
                    feature_df,
                    use_container_width=True
                )

                st.info(
                    "These features are generated from the "
                    "uploaded STL geometry and project configuration."
                )

            # =================================================
            # MATERIAL PREDICTION
            # =================================================

            else:

                st.markdown(
                    "### Ready for ML Prediction"
                )

                if st.button(
                    "🚀 RUN ARCHMIND V2.2",
                    use_container_width=True,
                    type="primary"
                ):

                    # -----------------------------------------
                    # Prepare input
                    # -----------------------------------------

                    input_df = (
                        pd.DataFrame(
                            [features]
                        )[FEATURES]
                    )

                    # -----------------------------------------
                    # Prediction
                    # -----------------------------------------

                    predictions = {}

                    for target, model in models.items():

                        prediction = model.predict(
                            input_df
                        )

                        value = float(
                            np.asarray(
                                prediction
                            )
                            .reshape(-1)[0]
                        )

                        predictions[target] = max(
                            0.0,
                            value
                        )

                    # =================================================
                    # RESULTS
                    # =================================================

                    st.markdown(
                        "### 📊 Resource Allocation"
                    )

                    a, b = st.columns(2)

                    with a:

                        st.markdown(
                            f"""
                            <div class="res-card">
                                <h4>🧱 Cement</h4>
                                <h2>
                                    {predictions["cement_bags"]:,.0f}
                                    bags
                                </h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with b:

                        st.markdown(
                            f"""
                            <div class="res-card">
                                <h4>🔩 Steel Rebar</h4>
                                <h2>
                                    {predictions["steel_tonnes"]:,.2f}
                                    tonnes
                                </h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    c, d = st.columns(2)

                    with c:

                        st.markdown(
                            f"""
                            <div class="res-card">
                                <h4>🧱 Bricks</h4>
                                <h2>
                                    {predictions["brick_count"]:,.0f}
                                </h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with d:

                        st.markdown(
                            f"""
                            <div class="res-card">
                                <h4>⬜ AAC Blocks</h4>
                                <h2>
                                    {predictions["aac_block_count"]:,.0f}
                                </h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # =================================================
                    # INPUT SUMMARY
                    # =================================================

                    st.subheader(
                        "Prediction Input Summary"
                    )

                    st.dataframe(
                        input_df.T.rename(
                            columns={
                                0: "value"
                            }
                        ),
                        use_container_width=True
                    )

                    # =================================================
                    # DISCLAIMER
                    # =================================================

                    st.warning(
                        "⚠️ These are ML planning estimates generated "
                        "from the V2.2 synthetic engineering-informed "
                        "training distribution. They are not a substitute "
                        "for structural drawings, BOQs, quantity surveys "
                        "or professional engineering."
                    )

        except Exception as e:

            st.error(
                f"Could not process this STL: {e}"
            )

            st.exception(e)