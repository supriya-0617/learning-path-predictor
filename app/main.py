"""Streamlit entry point: `streamlit run app/main.py`"""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core import COURSE_CATALOGUE, recommendation_details


st.set_page_config(page_title="Learning Path Predictor", layout="wide")
st.title("Learning Path Predictor")
st.write(
    "Select courses you have completed. The fine-tuned language model will rank "
    "valid next-course recommendations."
)

catalogue_rows = [
    {
        "Course ID": course_id,
        "Course name": details["name"],
        "Category": details["category"],
    }
    for course_id, details in COURSE_CATALOGUE.items()
]
with st.expander("Available course catalogue"):
    st.table(pd.DataFrame(catalogue_rows))

options = [
    f"{course_id} - {details['name']}"
    for course_id, details in COURSE_CATALOGUE.items()
]
selected = st.multiselect("Completed courses", options)
top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if not selected:
    st.info("Select at least one completed course to generate recommendations.")
else:
    completed_ids = [choice.split(" - ", 1)[0] for choice in selected]
    try:
        recommendations = recommendation_details(completed_ids, top_n)
        if recommendations:
            st.subheader("Recommended next courses")
            st.table(pd.DataFrame(recommendations))
        else:
            st.info("No additional courses are available.")
    except (FileNotFoundError, RuntimeError, OSError) as error:
        st.error(f"The recommendation model is unavailable: {error}")
