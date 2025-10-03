import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
st.write("✅ Streamlit started: app.py running")

try:
    from meddx.serve.ui_app import main
    st.write("✅ Imported ui_app successfully")
    main()
except Exception as e:
    st.error(f"❌ Error in app: {e}")
    raise
