import streamlit as st
from streamlit_gsheets import GSheetsConnection

# Create a connection object.
# conn = st.connection("gsheets", type=GSheetsConnection)

# df = conn.read(
    # worksheet="database",
    # ttl="10m",
    # usecols=[0, 1, 2],
    # nrows=4,
#)

#for row in df.itertuples():
    #st.write(f"{row.fullname} {row.financexpertise} : {row.lightfinance}:")


@st.cache_data(ttl=600)
def load_consultant_data():
    try:
        from streamlit_gsheets import GSheetsConnection
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="database", ttl="10m", usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], nrows=1000)
        return df
    except Exception as e:
        st.error(f"❌ Error loading consultant data: {e}")
        return None
df = load_consultant_data()
st.dataframe(df)