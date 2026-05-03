import streamlit as st
import numpy as np

# --- 1. PhenoAge Constants and Configuration ---

# List of biomarkers required for the PhenoAge calculation
BIOMARKER_KEYS = [
    'Albumin', 'Creatinine', 'Glucose', 'CRP', 'Lymph', 'MCV', 'RDW', 'ALK', 'WBC'
]

# Approximate normal ranges for user guidance (These are the common UI input units)
NORMAL_RANGES = {
    'Albumin': (3.5, 5.5, 'g/dL'),          # UI unit
    'ALK': (44, 147, 'U/L'),                # UI unit
    'Creatinine': (0.6, 1.3, 'mg/dL'),      # UI unit
    'Glucose': (70, 100, 'mg/dL (Fasting)'),# UI unit
    'CRP': (0.0, 3.0, 'mg/L'),              # UI unit
    'Lymph': (20.0, 40.0, '%'),
    'MCV': (80.0, 100.0, 'fL'),
    'RDW': (11.5, 14.5, '%'),
    'WBC': (4.5, 11.0, '1000/uL'),
}

# --- 2. PhenoAge Calculation Functions ---

def _compute_phenoage_paper_units(paper_units: dict, chronological_age: float) -> tuple[float, float, float]:
    """
    Core function: Calculates the PhenoAge Linear Predictor (xb) and converts it to 
    PhenoAge in years using the corrected Gompertz transformation.
    
    IMPORTANT: This function expects biomarker inputs to be in the paper's required units:
        Albumin: g/L, Creatinine: µmol/L, Glucose: mmol/L, CRP: mg/dL (for the log term)
        
    Returns: (phenoage_years, age_gap, xb_score)
    """
    try:
        # Check for non-positive CRP, which requires natural log transformation
        if paper_units.get('CRP', 0) <= 0:
            st.error("Converted CRP value must be positive for the PhenoAge calculation.")
            return 0.0, 0.0, 0.0

        # Linear Predictor ("xb") using the coefficients from the published formula
        # Note: All inputs are now in the necessary paper units (g/L, µmol/L, mmol/L, etc.)
        xb = (
            -19.907
            - 0.0336 * paper_units['Albumin']
            + 0.0095 * paper_units['Creatinine']
            + 0.1953 * paper_units['Glucose']
            + 0.0954 * np.log(paper_units['CRP'])  # ln(CRP, now in mg/dL)
            - 0.0120 * paper_units['Lymph']
            + 0.0268 * paper_units['MCV']
            + 0.3306 * paper_units['RDW']
            + 0.00188 * paper_units['ALK']
            + 0.0554 * paper_units['WBC']
            + 0.0804 * chronological_age
        )

        # --- Corrected Gompertz transformation (using intermediate M term) ---
        
        if xb > 700:
            st.error("Calculation boundary exceeded: The Linear Predictor (xb) is too high.")
            return 0.0, 0.0, 0.0

        # 1. Calculate the intermediate exponent term (E)
        exponent_term = -1.51714 * np.exp(xb) / 0.0076927

        # 2. Calculate the intermediate M term (Mortality Hazard)
        M = 1 - np.exp(exponent_term)

        # Check for numerical stability of M
        if M >= 1.0 or M <= 0.0:
            st.error("Calculation boundary exceeded: Intermediate M term is out of range. Check input values.")
            return 0.0, 0.0, 0.0
            
        # 3. Calculate the outer log argument
        arg_outer_log = -0.00553 * np.log(1 - M)

        if arg_outer_log <= 0:
            st.error("Calculation boundary exceeded: Intermediate log result is non-positive. Check input values.")
            return 0.0, 0.0, 0.0

        # 4. Final PhenoAge calculation
        phenoage_years = 141.50 + np.log(arg_outer_log) / 0.09165

        # --- End of Corrected Transformation ---
        
        # 5. Calculate Age Gap
        age_gap = phenoage_years - chronological_age
        
        # Check for physically impossible results
        if phenoage_years < 1:
            st.error("The calculated PhenoAge is not a plausible age. Input values may be extreme.")
            return 0.0, 0.0, 0.0

        return phenoage_years, age_gap, xb

    except KeyError as e:
        st.error(f"Missing internal biomarker: {e}. This indicates a code issue.")
        return 0.0, 0.0, 0.0
    except Exception as e:
        st.error(f"An unexpected error occurred during core calculation: {e}")
        return 0.0, 0.0, 0.0

def calculate_phenoage_years(biomarkers: dict, chronological_age: float) -> tuple[float, float, float]:
    """
    Wrapper function: Takes common US lab units (g/dL, mg/dL, mg/L) from the UI, 
    converts them to the paper's required units (g/L, µmol/L, mmol/L, mg/dL), 
    and then calls the core calculation.
    """
    try:
        # 1. Input Validation (Ensure all keys exist)
        for key in BIOMARKER_KEYS:
            if key not in biomarkers:
                raise KeyError(key)

        # 2. Unit Conversions (From common UI units to Paper units)
        paper_units = {
            # Albumin: g/dL -> g/L (Multiply by 10)
            'Albumin': biomarkers['Albumin'] * 10.0,
            # Creatinine: mg/dL -> µmol/L (Multiply by 88.4)
            'Creatinine': biomarkers['Creatinine'] * 88.4,
            # Glucose: mg/dL -> mmol/L (Divide by 18.016)
            'Glucose': biomarkers['Glucose'] / 18.016,
            # CRP: mg/L -> mg/dL (Divide by 10, as the formula's log term uses mg/dL)
            'CRP': biomarkers['CRP'] / 10.0,
            
            # The following units (%, fL, U/L, 1000/uL) are assumed to be consistent 
            # with the units used in the paper's coefficients (Lymph, MCV, RDW, ALK, WBC)
            'Lymph': biomarkers['Lymph'],
            'MCV': biomarkers['MCV'],
            'RDW': biomarkers['RDW'],
            'ALK': biomarkers['ALK'],
            'WBC': biomarkers['WBC'],
        }

        # 3. Call the core calculation function
        return _compute_phenoage_paper_units(paper_units, chronological_age)

    except KeyError as e:
        st.error(f"Missing biomarker input: {e}. Please ensure all 9 inputs are present.")
        return 0.0, 0.0, 0.0
    except Exception as e:
        st.error(f"An unexpected error occurred during unit conversion: {e}")
        return 0.0, 0.0, 0.0

# --- 3. Streamlit UI Layout ---

st.set_page_config(
    page_title="PhenoAge Biomarker Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 PhenoAge Predictor (Corrected Gompertz Model with Unit Conversion)")
st.markdown("""
Estimate biological age (**PhenoAge**) based on 9 common blood biomarkers and your Chronological Age.
The difference between your Chronological Age and PhenoAge is the **Age Gap**.
A positive gap suggests accelerated aging.
""")

# Setup two columns for the input layout
col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("1. Enter Biomarker Values")
    st.markdown("Please enter the values from your most recent blood panel. **Note: Units are converted internally to match the paper's formula.**")

    # Create a dictionary to hold user inputs
    biomarkers = {}

    # Layout for inputs using columns
    input_cols = st.columns(3)

    # Chronological Age Input (Needed for calculation)
    st.subheader("Personal Information")
    chronological_age = st.number_input(
        "Chronological Age (Years)",
        min_value=18.0,
        max_value=120.0,
        value=40.0,
        step=1.0,
        format="%.1f",
        help="Your actual age in years. This is used as a variable in the PhenoAge formula."
    )
    st.markdown("---")
    st.subheader("Biomarker Inputs")

    # Dynamic input generation with normal ranges shown in tooltips/help
    input_keys = BIOMARKER_KEYS
    
    for i, key in enumerate(input_keys):
        # Determine the column for the current input
        col_index = i % 3
        with st.container():
            # Get the range and unit for the current biomarker
            low, high, unit = NORMAL_RANGES.get(key, (None, None, ''))
            
            help_text = f"Normal Range: {low} - {high} {unit}"
            label = f"{key} ({unit})"
            
            # Set a typical default value within the range
            default_value = (low + high) / 2 if low and high else 1.0

            # Adjust step and format based on expected precision
            step = 0.1 if key in ['Albumin', 'Creatinine', 'CRP'] else 1.0
            format_str = "%.2f" if step == 0.1 else "%.1f"

            biomarkers[key] = input_cols[col_index].number_input(
                label,
                min_value=0.01,
                value=default_value,
                step=step,
                format=format_str,
                help=help_text
            )


with col_right:
    st.header("2. Normal Ranges")
    st.markdown("Reference values for the biomarkers (approximate, in common reporting units):")
    
    # Display the normal ranges in the right column
    range_table = []
    for key, (low, high, unit) in NORMAL_RANGES.items():
        range_table.append({
            'Biomarker': key,
            'Range': f"{low} - {high}",
            'Unit': unit
        })
    
    st.table(range_table)
    
    # Optional section for the formula
    st.subheader("Model Overview")
    st.info("This prediction uses the corrected Gompertz proportional hazards model. Your input units (g/dL, mg/dL, mg/L) are automatically converted to the units used in the paper's formula (g/L, µmol/L, mmol/L, mg/dL) prior to calculation.")

st.markdown("---")

# --- 4. Calculation and Results ---
if st.button("Calculate PhenoAge", type="primary"):
    
    phenoage, age_gap, xb_score = calculate_phenoage_years(biomarkers, chronological_age)
    
    if phenoage > 0: # Only proceed if calculation was successful
        
        st.header("3. Prediction Results")
        
        # Setup columns for metrics display
        result_col1, result_col2, result_col3 = st.columns(3)

        # Metric 1: Chronological Age
        result_col1.metric(
            label="Chronological Age",
            value=f"{chronological_age:.1f} years"
        )
        
        # Metric 2: PhenoAge
        delta_color = "inverse" if age_gap < 0 else "off"
        result_col2.metric(
            label="Predicted PhenoAge",
            value=f"{phenoage:.1f} years",
            delta=f"{age_gap:.1f} Age Gap",
            delta_color=delta_color
        )
        
        # Metric 3: Interpretation
        interpretation_text = ""
        if age_gap < -5:
            interpretation_text = "Highly decelerated aging (significantly younger biological age)."
            st.balloons()
        elif age_gap < 0:
            interpretation_text = "Decelerated aging (younger biological age)."
        elif age_gap < 5:
            interpretation_text = "Biological age is close to chronological age."
        else:
            interpretation_text = "Accelerated aging (older biological age)."

        result_col3.metric(
            label="Age Acceleration Status",
            value="Interpretation",
            delta=interpretation_text,
            delta_color="off"
        )
        
        st.markdown("---")
        
        st.subheader("Detailed Result")
        st.info(f"The raw Linear Predictor (xb) score is **{xb_score:.3f}**.")
        st.warning("Disclaimer: This tool provides a scientific estimate for educational purposes only and is not a substitute for professional medical advice or official diagnostic testing.")
